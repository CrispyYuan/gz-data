import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
from lime import lime_image
from tqdm import tqdm

# 导入自定义工具
import utils

# Ensure project root is on sys.path so we can import pattern_recognition
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入外部的 ExplainerModelWrapper 模块（位于 pattern_recognition/ExplainerModelWrapper.py）
from pattern_recognition import ExplainerModelWrapper

# ==========================================
def load_trained_model(model_path, device):
    """
    加载训练好的模型并用外部的 ExplainerModelWrapper 包装。
    兼容情况：
      - 保存为 state_dict 的 checkpoint
      - 保存为完整的 model 对象（torch.save(model))
    会尝试若干候选架构：nnUNetV2 (BraTS/ACDC) 与 UNet (BraTS/ACDC)
    """
    # 尝试加载 checkpoint（支持 map_location）
    ckpt = torch.load(model_path, map_location=device)

    # 如果直接存了 model 对象（非 dict），尝试直接使用
    if not isinstance(ckpt, dict):
        try:
            model = ckpt
            model.to(device)
            model.eval()
            wrapped = ExplainerModelWrapper.ExplainerModelWrapper(model)
            wrapped.to(device)
            wrapped.eval()
            return wrapped
        except Exception:
            pass

    # 支持多种 state_dict 格式（可能包含 'state_dict' 键）
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        state_dict = None

    # 载入候选架构
    from pattern_recognition.models.UNet import get_unet_acdc, get_unet_brats
    from pattern_recognition.models.nnUNetV2 import get_nnunet_acdc, get_nnunet_brats

    candidates = [
        get_nnunet_brats(),
        get_nnunet_acdc(),
        get_unet_brats(),
        get_unet_acdc(),
    ]

    model = None
    last_err = None
    if state_dict is None:
        raise RuntimeError(f"无法识别的 checkpoint 格式: {model_path}")

    for cand in candidates:
        try:
            cand.load_state_dict(state_dict)
            model = cand
            break
        except Exception as e:
            last_err = e
            continue

    if model is None:
        raise RuntimeError(f"无法将 state_dict 匹配到候选模型结构，请检查模型文件。最新错误: {last_err}")

    model.to(device)
    model.eval()

    # 使用外部提供的 ExplainerModelWrapper 进行包装
    wrapped = ExplainerModelWrapper.ExplainerModelWrapper(model)
    wrapped.to(device)
    wrapped.eval()
    return wrapped

# ==========================================
# 2. 核心：LIME 预测包装器 (Prediction Wrapper)
# ==========================================
def segmentation_prediction_wrapper(images):
    """
    LIME 需要一个函数：输入 numpy 图像列表 -> 输出 预测概率(N, classes)。
    
    对于分割任务，我们需要将 '分割图' 转换为 '分类概率' 形式供 LIME 理解。
    策略：计算图像中属于'肿瘤'类别的概率总和（或平均值）。
    LIME 将解释：哪些超像素块导致了'肿瘤总概率'的增加。
    """
    global MODEL, DEVICE

    # LIME 输入可能为 (H, W, C) 或 (N, H, W, C) 或 (H, W) 等
    arr = np.array(images)

    # Normalize per-image
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim == 3:
        # single image (H,W,C) -> batch
        arr = arr[np.newaxis, ...]

    # Ensure float
    arr = arr.astype(np.float32)

    # Convert RGB->grayscale if necessary (take first channel)
    if arr.shape[-1] == 3:
        # convert to single channel by averaging
        arr = np.mean(arr, axis=-1, keepdims=True)

    # transpose to (N, C, H, W) or (N, C, D, H, W) later
    N, H, W, C = arr.shape

    # Detect model expected input channels and dimensionality by inspecting modules
    model_obj = MODEL.model if hasattr(MODEL, 'model') else MODEL
    input_dim = 2
    in_ch = 1
    found3 = False
    found2 = False
    in_ch3 = None
    in_ch2 = None
    for m in model_obj.modules():
        if isinstance(m, torch.nn.Conv3d) and in_ch3 is None:
            found3 = True
            in_ch3 = m.in_channels
        if isinstance(m, torch.nn.Conv2d) and in_ch2 is None:
            found2 = True
            in_ch2 = m.in_channels
    if found3:
        input_dim = 3
        in_ch = in_ch3
    elif found2:
        input_dim = 2
        in_ch = in_ch2

    # Prepare tensor according to model's expected dim and channels
    if input_dim == 2:
        # Create tensor (N, in_ch, H, W)
        img_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        if in_ch != img_tensor.shape[1]:
            # repeat channels
            img_tensor = img_tensor.repeat(1, int(np.ceil(in_ch / img_tensor.shape[1])), 1, 1)
            img_tensor = img_tensor[:, :in_ch, :, :]
    else:
        # 3D model: create depth=1 volume: (N, C, D=1, H, W)
        img_tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()  # (N, C, H, W)
        img_tensor = img_tensor.unsqueeze(2)  # (N, C, 1, H, W)
        if in_ch != img_tensor.shape[1]:
            img_tensor = img_tensor.repeat(1, int(np.ceil(in_ch / img_tensor.shape[1])), 1, 1, 1)
            img_tensor = img_tensor[:, :in_ch, :, :, :]

        # If model has pooling in depth dimension, ensure depth >= 2^num_pools
        min_depth = 1
        try:
            if hasattr(model_obj, 'encoder'):
                num_pools = max(0, len(model_obj.encoder) - 1)
                min_depth = 2 ** num_pools
        except Exception:
            min_depth = 8

        if img_tensor.shape[2] < min_depth:
            # repeat along depth axis to reach min_depth
            reps = int(np.ceil(min_depth / img_tensor.shape[2]))
            img_tensor = img_tensor.repeat(1, 1, reps, 1, 1)
            img_tensor = img_tensor[:, :, :min_depth, :, :]

    img_tensor = img_tensor.to(DEVICE)

    # Run model
    with torch.no_grad():
        out = MODEL(img_tensor)

    # If model returned aggregated class scores (B, C), forward may have reduced spatial dims
    if isinstance(out, torch.Tensor):
        output = out
    elif isinstance(out, (list, tuple)):
        # take first element if list (deep supervision)
        output = out[0]
    else:
        # unexpected type
        output = torch.tensor(np.array(out)).to(DEVICE)

    # If output is (B, C) already, assume these are class scores
    if output.ndim == 2:
        scores = output.cpu().numpy()
        return scores

    # Now output should be (B, C, ...) with spatial dims
    # Sum probabilities of non-background classes as foreground score
    if output.ndim >= 3:
        # Ensure probs: if logits, apply softmax along channel dim
        if output.max() > 1.0 or output.min() < 0.0:
            probs = torch.softmax(output, dim=1)
        else:
            probs = output

        # foreground defined as sum over channels 1..end
        if probs.shape[1] == 1:
            fg = probs[:, 0, ...]
        else:
            fg = torch.sum(probs[:, 1:, ...], dim=1)

        # aggregate spatially to a scalar score per image
        # if 3D: sum over all spatial dims
        spatial_dims = tuple(range(1, fg.ndim))
        tumor_score = torch.sum(fg, dim=spatial_dims).cpu().numpy()

        # Build two-class output for LIME: [background_score, tumor_score]
        zeros = np.zeros_like(tumor_score)
        res = np.stack([zeros, tumor_score], axis=1)
        return res

    # fallback
    return np.zeros((N, 2))

# ==========================================
# 3. 主流程
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1 & 2: LIME Analysis for Segmentation")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input MRI (.nii.gz)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to ground truth mask (.nii.gz)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--case_id", type=str, default="Case_001", help="ID for the patient case")
    args = parser.parse_args()

    # 配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"--- Processing {args.case_id} ---")

    # 1. 加载数据
    # 注意：这里加载的是 3D 数据
    print("Loading data...")
    volume_data = utils.load_nifti_image(args.img_path) # (H, W, D) 或 (C, H, W, D)
    mask_data = utils.load_nifti_image(args.mask_path)
    
    # 处理多模态情况：如果数据是 (C, H, W, D)，取特定模态（例如 T1c 或 FLAIR）
    if volume_data.ndim == 4:
        volume_data = volume_data[0, ...] # 取第一个通道，根据实际情况修改
        
    # 2. 选择最佳 2D 切片
    # LIME 在 3D 上非常慢，通常做法是对含有最大肿瘤面积的切片进行解释
    best_slice_idx = utils.get_best_slice(mask_data)
    print(f"Selected Slice Index: {best_slice_idx}")
    
    img_slice = volume_data[:, :, best_slice_idx]
    mask_slice = mask_data[:, :, best_slice_idx]
    
    # 预处理：标准化并转为 RGB 格式 (LIME 默认处理 RGB，虽然可以是灰度，但转为 RGB 兼容性最好)
    img_slice_norm = utils.normalize_image(img_slice)
    # 扩展为 (H, W, 3) 供 LIME 处理，或者 (H, W) 如果用灰度模式
    # 这里保持 (H, W) 并在 wrapper 里处理维度
    
    # 3. 加载模型
    print("Loading model...")
    MODEL = load_trained_model(args.model_path, DEVICE)
    
    # 4. 初始化 LIME Explainer
    print("Initializing LIME Explainer...")
    explainer = lime_image.LimeImageExplainer()
    
    # 5. 生成解释 (最耗时的一步)
    # segmentation_fn='slic' 是产生超像素的方法
    # num_samples 是扰动次数，次数越多越准但越慢 (建议测试时设为 100，正式跑设为 1000)
    print("Running LIME (this may take a while)...")
    explanation = explainer.explain_instance(
        image=img_slice_norm, 
        classifier_fn=segmentation_prediction_wrapper, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000,
        batch_size=16 # 根据显存调整
    )
    
    # 6. 获取结果并可视化
    print("Generating visualizations...")
    # 获取解释叠加图
    temp, mask_lime = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=5, 
        hide_rest=False
    )
    
    # 生成完整的 Heatmap (用于 Step 3 定量分析)
    heatmap = utils.generate_heatmap_from_lime(explanation, img_slice_norm)
    
    # 7. 保存结果
    # 保存图像
    vis_path = os.path.join(args.output_dir, f"{args.case_id}_step2_panel.png")
    utils.save_comparison_plot(img_slice_norm, mask_slice, heatmap, mask_lime, vis_path)
    
    # 保存数据矩阵 (重要：为了 Step 3 的 Overlap@k 计算)
    # 保存内容：LIME热力图，原始GT Mask，预测 Mask(可选)
    np.save(os.path.join(args.output_dir, f"{args.case_id}_lime_heatmap.npy"), heatmap)
    np.save(os.path.join(args.output_dir, f"{args.case_id}_gt_mask.npy"), mask_slice)
    
    print(f"Done! Results saved to {args.output_dir}")