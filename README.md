# STQS-DETR

A PyTorch implementation of the Real-time Object Detector, STQS-DETR.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- pyyaml, scipy, tqdm, pycocotools

To install dependencies:
```bash
cd stqsdetr_pytorch
pip install -r requirements.txt
```

## Data Preparation
- **UOD Dataset**: [Download Link](https://github.com/liujiahui1998/UnderwaterDataset)
- **UTDAC2020 Dataset**: [Download Link](https://github.com/mousecpn/Collection-of-Underwater-Object-Detection-Dataset)

Prepare the UOD and UTDAC2020 dataset in the following structure:
```
stqsdetr_pytorch/dataset/UOD/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```
Update the dataset paths in [coco_detection.yml](stqsdetr_pytorch/configs/dataset/coco_detection.yml).

## Pretrained Weights
Local pretrained backbone weights can be placed in:
- `regnet-y-040-local/` (e.g., `pytorch_model.bin`)

Or specify your own paths in the corresponding configuration files in [configs/stqsdetr/include/](stqsdetr_pytorch/configs/stqsdetr/include/).

## Well trained model
| Model | Link | Code | Notes |
| :---: | :---: | :---: | :---: |
| STQS-DETR-RegNet | [Baidu Netdisk](https://pan.baidu.com/s/14qYztJ7pRwHwE5pZVOEIGQ?pwd=4pxg) | 4pxg | `checkpoint0092.pth` |

## Usage

### Training
```bash
cd stqsdetr_pytorch
python tools/train.py -c configs/stqsdetr/stqsdetr_regnet_6x_coco.yml
```

### Inference
```bash
python tools/infer.py -c configs/stqsdetr/stqsdetr_regnet_6x_coco.yml -r path/to/checkpoint
```

### Export ONNX
```bash
python tools/export_onnx.py -c configs/stqsdetr/stqsdetr_regnet_6x_coco.yml -r path/to/checkpoint --check
```
