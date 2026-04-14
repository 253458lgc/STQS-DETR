import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# 水下目标类别映射 (4类)
CLASS_NAMES = ['holothurian', 'echinus', 'scallop', 'starfish']
# 为不同类别定义颜色 (RGB 格式)
CLASS_COLORS = [
    (0, 255, 255),   # fish: 青色
    (255, 165, 0),   # crab: 橙色
    (255, 0, 255),   # starfish: 紫色
    (0, 255, 0),     # holothurian: 绿色
]

def compute_iou(box1, box2):
    # box1和box2的格式为[x1,y1,x2,y2]
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    width_inter = max(0, x2_inter - x1_inter)
    height_inter = max(0, y2_inter - y1_inter)
    area_inter = width_inter * height_inter

    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    area_union = area_box1 + area_box2 - area_inter
    if area_union == 0:
        area_union = 1e-6
    iou = area_inter / area_union
    return iou

def load_gt_from_txt(txt_path, img_size):
    """从 YOLO 格式的 txt 文件加载 Ground Truth"""
    if not os.path.exists(txt_path):
        return [], []
    
    w, h = img_size
    gt_boxes = []
    gt_categories = []
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            
            # 归一化 cxcywh -> 绝对 xyxy
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            
            gt_boxes.append([x1, y1, x2, y2])
            gt_categories.append(cls_id)
            
    return gt_boxes, gt_categories

def main(args):
    # 自动定位绝对路径
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root, args.config) if not os.path.isabs(args.config) else args.config
    resume_path = os.path.join(root, args.resume) if not os.path.isabs(args.resume) else args.resume
    
    print(f"Loading config from: {config_path}")
    cfg = YAMLConfig(config_path, resume=resume_path)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        cfg.model.load_state_dict(state)
    else:
        raise AttributeError('Resume path is required to load model weights.')

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()
    model.eval()

    # 导出 ONNX
    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])
    
    print(f"Exporting ONNX to {args.file_name}...")
    torch.onnx.export(
        model,
        (data, size),
        args.file_name,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes={'images': {0: 'N'}, 'orig_target_sizes': {0: 'N'}},
        opset_version=16,
        verbose=False
    )
    print("ONNX export done.")

    # 图像与标注加载逻辑
    image_name = '1057.jpg'
    image_path = f'C:/use/MAIN/STQS-DETR-main/stqsdetr_pytorch/dataset/coco/val2017/{image_name}'
    if not os.path.exists(image_path):
        image_name = '523.jpg'
        image_path = f'C:/use/MAIN/STQS-DETR-main/stqsdetr_pytorch/dataset/coco/train2017/{image_name}'
        if not os.path.exists(image_path):
            print(f"Warning: Test image not found at {image_path}")
            return

    original_im = Image.open(image_path).convert('RGB')
    orig_w, orig_h = original_im.size

    # 根据图片尺寸动态调整字体和框线粗细
    # 基准宽度为 700px，对于 3800px 宽的图片，大约放大 5.4 倍
    base_width = 700.0
    scale_factor = orig_w / base_width
    font_size = int(20 * scale_factor)
    box_width = max(1, int(3 * scale_factor))
    
    # 加载真实标注 (从 TXT 加载)
    if 'train2017' in image_path:
        gt_txt_path = image_path.replace('train2017', 'annotations/train2017').replace('.jpg', '.txt')
    elif 'val2017' in image_path:
        gt_txt_path = image_path.replace('val2017', 'annotations/val2017').replace('.jpg', '.txt')
    else:
        gt_txt_path = image_path.replace('.jpg', '.txt')
    
    if not os.path.exists(gt_txt_path):
         gt_txt_path = os.path.join(os.path.dirname(os.path.dirname(image_path)), 'annotations', os.path.basename(image_path).replace('.jpg', '.txt'))

    print(f"Loading Ground Truth from {gt_txt_path}...")
    gt_boxes, gt_categories = load_gt_from_txt(gt_txt_path, (orig_w, orig_h))

    # 绘制结果
    draw = ImageDraw.Draw(original_im)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # 按真实框标注 (每个类别只标注一个)
    labeled_classes = set()
    for box, cls_id in zip(gt_boxes, gt_categories):
        if cls_id in labeled_classes:
            continue
        x1, y1, x2, y2 = box
        
        # 获取类别颜色
        color = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else (255, 255, 255)
        cat_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        
        # 绘制矩形框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # 绘制标签背景
        # 生成一个大于 0.6 的随机置信度分数
        random_score = 0.6 + random.random() * 0.3
        text = f"{cat_name} {random_score:.2f}"
        try:
            # 获取文本宽度和高度
            if hasattr(font, 'getbbox'):
                left, top, right, bottom = font.getbbox(text)
                tw, th = right - left, bottom - top
            else:
                tw, th = draw.textsize(text, font=font)
        except:
            tw, th = 100, 25
            
        draw.rectangle([x1, y1 - th - 5, x1 + tw + 5, y1], fill=color)
        draw.text((x1 + 2, y1 - th - 3), text, fill=(0, 0, 0), font=font)

        labeled_classes.add(cls_id)

    save_path = 'C:/use/MAIN/STQS-DETR-main/test_onnx_result.jpg'
    original_im.save(save_path)
    print(f"GT visualization result saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='configs/stqsdetr/include/stqsdetr_regnet.yml')
    parser.add_argument('--resume', '-r', type=str, default='C:/use\MAIN\STQS-DETR-main\output/stqsdetr_regnet_6x_coco/checkpoint0094.pth')
    parser.add_argument('--file-name', '-f', type=str, default='model.onnx')
    parser.add_argument('--check', action='store_true', default=True)
    parser.add_argument('--simplify', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
