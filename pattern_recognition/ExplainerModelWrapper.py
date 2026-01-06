import torch 
import torch.nn as nn
import torch.nn.functional as F

class ExplainerModelWrapper(nn.Module):
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model
        self.model.eval()
        # 用于存储 Grad-CAM 所需的中间变量
        self.gradients = None
        self.activations = None

    def forward(self, x):
        # 1. 得到原始输出
        logits = self.model(x)
        
        # 处理 nnU-Net V2 可能存在的多尺度输出 (Deep Supervision)
        # 通常推理时只取最高分辨率的输出 (第一个元素)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
            
        # 2. 必须输出 Softmax，用于 SHAP 和 LIME
        # 对于分割任务，dim=1 是类别维度 (B, C, H, W)
        probs = torch.softmax(logits, dim=1)
        
        # 如果是用于 SHAP 解释，且输出是多维的 (分割图)，
        # SHAP 的 GradientExplainer 要求输出是 (B, C) 格式的标量聚合
        # 我们返回每个类别的全局平均概率，作为该类别在全图上的“存在感”得分
        if self.training or not torch.is_grad_enabled():
            return probs
        
        # 在计算梯度时，聚合空间维度
        spatial_dims = tuple(range(2, probs.ndim))
        return torch.mean(probs, dim=spatial_dims)

    def get_entropy(self, probs):
        """
        计算像素级熵作为不确定性指标 (Uncertainty Index)
        probs: (B, C, H, W) 或 (B, C, D, H, W)
        return: (B, H, W) 或 (B, D, H, W)
        """
        # 熵公式: -sum(p * log(p))，用于衡量预测的不确定性
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return entropy

    # --- Grad-CAM 支持 ---
    def register_hooks(self, target_layer):
        """
        为 Grad-CAM 注册 forward 和 backward hooks
        target_layer: 模型中的目标卷积层 (例如 Encoder 的最后一层)
        """
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] 是对于输出的梯度
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        # 使用 register_full_backward_hook 保证在较新版本 PyTorch 中的兼容性
        target_layer.register_full_backward_hook(backward_hook)

    def get_gradcam(self):
        """
        计算 Grad-CAM 热力图
        注意：调用此方法前，必须先进行 forward() 并对目标输出执行 backward()
        """
        if self.gradients is None or self.activations is None:
            return None
        
        # 1. 计算通道权重 alpha: 对梯度进行全局平均池化 (支持 2D/3D)
        # 维度从 2 开始是空间维度 (H, W) 或 (D, H, W)
        spatial_dims = tuple(range(2, self.gradients.ndim))
        weights = torch.mean(self.gradients, dim=spatial_dims, keepdim=True)
        
        # 2. 加权组合激活图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # 3. ReLU 激活，只关注对正向预测有贡献的特征
        cam = F.relu(cam)
        
        # 4. 归一化到 [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-10)
        
        return cam

    # --- 定量指标 (Overlap@k, Recall@k) ---
    @staticmethod
    def get_top_k_mask(score_map, k_ratio):
        """
        获取得分图 (如解释图或不确定性图) 前 k% 高分区域的掩码
        score_map: (H, W) 或 (D, H, W)
        k_ratio: 比例系数 (0 < k_ratio <= 1)
        """
        flat_map = score_map.flatten()
        k = int(len(flat_map) * k_ratio)
        if k <= 0: k = 1
        
        # 找到前 k 个最大值
        topk_values, _ = torch.topk(flat_map, k)
        threshold = topk_values[-1]
        
        # 生成二进制掩码
        mask = (score_map >= threshold).float()
        return mask

    @staticmethod
    def calculate_overlap_at_k(explanation_map, uncertainty_map, k_ratio=0.1):
        """
        计算 Overlap@k: 解释结果与不确定性指标在 top-k 区域的重合度
        explanation_map: 解释工具输出的热力图 (如 SHAP, Grad-CAM)
        uncertainty_map: 不确定性指标图 (如 Entropy)
        """
        # 确保在同一设备上
        uncertainty_map = uncertainty_map.to(explanation_map.device)
        
        mask_exp = ExplainerModelWrapper.get_top_k_mask(explanation_map, k_ratio)
        mask_unc = ExplainerModelWrapper.get_top_k_mask(uncertainty_map, k_ratio)
        
        intersection = torch.sum(mask_exp * mask_unc)
        total_k = torch.sum(mask_exp) # 实际的 k 个像素点
        
        overlap = intersection / (total_k + 1e-10)
        return overlap.item()

    @staticmethod
    def calculate_recall_at_k(explanation_map, uncertainty_map, k_ratio=0.1):
        """
        计算 Recall@k: 解释结果的 top-k 区域中有多少比例落在了不确定性的 top-k 区域内
        """
        # 在 k 相同的情况下，Recall@k 的计算逻辑与 Overlap@k 相同
        return ExplainerModelWrapper.calculate_overlap_at_k(explanation_map, uncertainty_map, k_ratio)
