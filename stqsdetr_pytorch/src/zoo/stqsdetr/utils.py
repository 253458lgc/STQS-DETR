"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


import math 
def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

def grad_cam_plus_plus(model, input_tensor, target_class):
    """
    计算Grad-CAM++热力图

    参数:
        model: STQSDETR模型实例
        input_tensor: 输入图像张量，形状为[1, 3, H, W]
        target_class: 目标类别索引，例如COCO数据集中的类别索引

    返回:
        cam: 热力图，形状为[H, W]，值范围为0-1
    """
    # 确保模型处于评估模式
    model.eval()
    
    # 钩子用于捕获特征图和梯度
    features_list = []
    gradients_list = []
    
    def save_features_hook(module, input, output):
        features_list.append(output)
        
    def save_gradients_hook(module, grad_input, grad_output):
        gradients_list.append(grad_output[0])
        
    # 注册钩子到HybridEncoder的最后一层
    # 注意：根据具体模型结构调整，这里假设使用encoder的输出
    target_layer = model.encoder.encoder[-1]
    handle_features = target_layer.register_forward_hook(save_features_hook)
    handle_gradients = target_layer.register_full_backward_hook(save_gradients_hook)

    # 前向传播
    output = model(input_tensor)

    # 获取目标类别的分数
    if isinstance(output, dict):
        # STQSDETR通常输出是一个字典，包含'pred_logits'和'pred_boxes'
        # pred_logits形状: [batch, num_queries, num_classes]
        scores = output['pred_logits']
    else:
        scores = output

    # 我们需要最大化目标类别的分数
    # 这里简单地取所有query中该类别的最大分数
    # scores: [1, 300, 4] -> 取第0个batch -> [300, 4]
    batch_scores = scores[0] 
    
    # 找到该类别得分最高的query
    class_scores = batch_scores[:, target_class]
    max_score_idx = torch.argmax(class_scores)
    target_score = class_scores[max_score_idx]
    
    # 反向传播
    model.zero_grad()
    target_score.backward(retain_graph=True)
    
    # 移除钩子
    handle_features.remove()
    handle_gradients.remove()

    # 获取特征图和梯度
    if not features_list or not gradients_list:
        raise ValueError("无法获取特征图或梯度，请检查目标层是否正确")
        
    features = features_list[0]   # [B, N, C] = [1, 2560, 256] (sequence length, channels)
    gradients = gradients_list[0] # [B, N, C]
    
    # 对于Transformer输出，我们需要将其重塑回空间维度
    # HybridEncoder输出通常是展平的特征序列
    # 我们需要知道空间形状。STQSDETR encoder输出通常是多尺度的
    # 这里为了简化，我们尝试重塑为近似的正方形或根据输入尺寸推断
    # 但更稳妥的是使用最后一层特征图（如果它保持了空间结构）
    
    # 如果是Transformer Encoder输出，通常是 (B, L, C)
    # 我们需要将其转换回 (B, C, H, W) 格式来进行Grad-CAM++计算
    # 这里做一个简单的假设：L = H * W
    B, L, C = features.shape
    H_feat = int(math.sqrt(L))
    W_feat = L // H_feat
    
    if H_feat * W_feat != L:
         # 如果不能完美开方，尝试使用固定的stride (例如32)
         # 输入640x640 -> stride 32 -> 20x20 = 400
         # STQSDETR encoder输出是多尺度特征拼接
         # 这里我们只取最高层级的特征进行可视化
         pass

    # 由于STQSDETR encoder输出结构复杂，我们改用backbone的最后一层进行可视化
    # 重新注册钩子到backbone最后一层
    features_list.clear()
    gradients_list.clear()
    
    # 获取backbone的最后一层
    # RegNet或PResNet的结构不同，需通用处理
    # 假设model.backbone存在且有body或直接是层列表
    if hasattr(model.backbone, 'body'): # IntermediateLayerGetter
         target_layer = model.backbone.body.layer4 # ResNet-like
    elif hasattr(model.backbone, 'model'): # RegNet
         # RegNet structure: model.model.encoder.stages[3]
         # 需要根据具体实现调整
         try:
             # HuggingFace RegNetModel output is (last_hidden_state, hidden_states)
             # But here we are hooking into nn.Module
             # RegNetModel structure: model.model.encoder.stages[3].layers[-1]
             target_layer = model.backbone.model.encoder.stages[2].layers[-1] # RegNet stage 3 (index 2)
         except:
             # Fallback: try to find the last Conv2d layer
             modules = list(model.backbone.modules())
             for m in reversed(modules):
                 if isinstance(m, nn.Conv2d):
                     target_layer = m
                     break
    else:
         # 尝试直接获取最后一层
         modules = list(model.backbone.modules())
         for m in reversed(modules):
             if isinstance(m, nn.Conv2d):
                 target_layer = m
                 break
                 
    handle_features = target_layer.register_forward_hook(save_features_hook)
    handle_gradients = target_layer.register_full_backward_hook(save_gradients_hook)
    
    # 再次前向传播以获取backbone特征
    model.zero_grad()
    output = model(input_tensor)
    scores = output['pred_logits'][0]
    class_scores = scores[:, target_class]
    target_score = class_scores[torch.argmax(class_scores)]
    target_score.backward(retain_graph=True)
    
    handle_features.remove()
    handle_gradients.remove()
    
    features = features_list[0]
    gradients = gradients_list[0]
    
    # 计算Grad-CAM++
    # features: [B, C, H, W]
    # gradients: [B, C, H, W]
    B, C, H, W = features.shape
    
    # 梯度
    grads = gradients
    
    # 权重计算 (Grad-CAM++ 公式)
    score_exp = torch.exp(target_score)
    grads_2 = grads ** 2
    grads_3 = grads ** 3
    
    sum_activations = torch.sum(features, dim=(2, 3))
    
    aij = grads_2 / (2 * grads_2 + sum_activations[:, :, None, None] * grads_3 + 1e-7)
    weights = torch.sum(aij * torch.relu(grads), dim=(2, 3))
    
    # 生成热力图
    cam = torch.sum(weights[:, :, None, None] * features, dim=1)
    
    # ReLU
    cam = F.relu(cam)
    
    # 归一化
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-7)
    
    # 调整大小
    cam = F.interpolate(cam.unsqueeze(1), size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze(1)
    
    return cam

