import torch
import torch.nn as nn
import yaml

class DiceLoss(nn.Module):
    """
    Dice Loss for medical image segmentation
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) or (B, C, D, H, W)
        # target: (B, H, W) or (B, D, H, W)
        
        # Convert target to one-hot
        C = pred.shape[1]
        # Ensure torch.eye is on the same device as target to avoid device mismatch
        eye = torch.eye(C, device=target.device)
        target_one_hot = eye[target.squeeze(1) if target.ndim > pred.ndim-1 else target]
        
        # Reshape to (B, C, -1)
        if pred.ndim == 4: # 2D
            target_one_hot = target_one_hot.permute(0, 3, 1, 2).contiguous()
        else: # 3D
            target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).contiguous()
            
        pred = torch.softmax(pred, dim=1)
        
        dims = tuple(range(2, pred.ndim))
        intersection = torch.sum(pred * target_one_hot, dim=dims)
        cardinality = torch.sum(pred + target_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth + 1e-8)
        return 1. - torch.mean(dice_score)

def get_loss_function(deep_supervision=False):
    dice_loss = DiceLoss()
    ce_loss = nn.CrossEntropyLoss()
    
    def loss_fn(pred, target):
        if deep_supervision and isinstance(pred, (list, tuple)):
            # nnU-Net V2 Deep Supervision Loss
            # 通常对不同尺度的输出赋予不同的权重
            weights = [1 / (2**i) for i in range(len(pred))]
            weights = [w / sum(weights) for w in weights]
            
            total_loss = 0
            for i, p in enumerate(pred):
                # 对 target 进行下采样以匹配输出尺度
                if p.shape[2:] != target.shape[1:]:
                    curr_target = F.interpolate(target.unsqueeze(1).float(), size=p.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    curr_target = target
                
                total_loss += weights[i] * (dice_loss(p, curr_target) + ce_loss(p, curr_target))
            return total_loss
        else:
            return dice_loss(pred, target) + ce_loss(pred, target)
            
    return loss_fn

import torch.nn.functional as F

def load_config(config_path):
    with open(config_path, 'r') as f:
        # Loader=yaml.FullLoader 是为了安全加载
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
