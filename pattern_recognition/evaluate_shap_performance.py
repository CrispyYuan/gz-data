import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import nibabel as nib
from train_utils import load_config

def calculate_dice_iou(mask1, mask2):
    """ 计算两个二进制掩码之间的 Dice 和 IoU """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    sum_masks = mask1.sum() + mask2.sum()
    
    dice = (2. * intersection) / (sum_masks + 1e-10)
    iou = intersection / (union + 1e-10)
    return dice, iou

def get_top_k_mask(attr, k_ratio=0.1):
    """ 获取贡献度最高的前 k% 区域作为解释掩码 """
    # attr 形状可能是 (C_in, D, H, W) 或 (C_in, H, W)
    # 先取绝对值并对输入通道求和
    total_attr = np.abs(attr).sum(axis=0)
    flat_attr = total_attr.flatten()
    
    k = int(len(flat_attr) * k_ratio)
    if k <= 0: k = 1
    
    threshold = np.partition(flat_attr, -k)[-k]
    mask = (total_attr >= threshold).astype(np.uint8)
    return mask

def evaluate_shap():
    config = load_config("config.yaml")
    model_type = config.get('interpret', {}).get('model', 'unet')
    data_root = config['path']['data_path']
    
    results = []

    # --- 1. 处理 BraTS 结果 ---
    brats_dir = f"shap/{model_type}_brats"
    if os.path.exists(brats_dir):
        print(f"Evaluating BraTS results in {brats_dir}...")
        for f in os.listdir(brats_dir):
            if f.startswith("metrics_") and f.endswith(".json"):
                with open(os.path.join(brats_dir, f), 'r') as jf:
                    meta = json.load(jf)
                
                pid = meta['patient_id']
                # 兼容不同模型的文件名后缀
                shap_path = os.path.join(brats_dir, f"shap_values_{pid}.npy")
                if not os.path.exists(shap_path):
                    shap_path = os.path.join(brats_dir, f"shap_values_{pid}_{model_type}.npy")
                
                if not os.path.exists(shap_path): continue
                
                # 加载 SHAP 值并生成掩码
                attr = np.load(shap_path)
                # 对于 3D BraTS，肿瘤占比极小，使用更小的 k_ratio (2%) 更合理
                exp_mask = get_top_k_mask(attr, k_ratio=0.02)
                
                # 加载对应的 Ground Truth (3D)
                # 注意：SHAP 是在 32x32x32 空间计算的，GT 也需要下采样
                gt_path = os.path.join(data_root, "BraTS_Processed/val", pid, f"{pid}_seg.nii.gz")
                gt_nii = nib.load(gt_path)
                gt_data = np.asanyarray(gt_nii.dataobj)
                # 映射标签 (4 -> 3)
                gt_data[gt_data == 4] = 3
                # 只关注目标类别 (Class 3)
                gt_binary = (gt_data == 3).astype(np.float32)
                
                # 下采样 GT 到 32x32x32 以匹配 SHAP 空间
                gt_tensor = torch.from_numpy(gt_binary).unsqueeze(0).unsqueeze(0)
                gt_sub = torch.nn.functional.interpolate(gt_tensor, size=(32, 32, 32), mode='nearest').numpy()[0, 0]
                
                dice, iou = calculate_dice_iou(exp_mask, gt_sub)
                results.append({"dataset": "BraTS", "id": pid, "dice": dice, "iou": iou})
                print(f"  {pid}: Dice={dice:.4f}, IoU={iou:.4f}")

    # --- 2. 处理 ACDC 结果 ---
    acdc_dir = f"shap/acdc_{model_type}"
    if os.path.exists(acdc_dir):
        print(f"Evaluating ACDC results in {acdc_dir}...")
        for f in os.listdir(acdc_dir):
            if f.startswith("metrics_") and f.endswith(".json"):
                with open(os.path.join(acdc_dir, f), 'r') as jf:
                    meta = json.load(jf)
                
                frame_name = meta['frame']
                shap_path = os.path.join(acdc_dir, f"shap_values_{frame_name}.npy")
                gt_path = os.path.join(acdc_dir, f"gt_{frame_name}.npy")
                
                if not (os.path.exists(shap_path) and os.path.exists(gt_path)): continue
                
                attr = np.load(shap_path)
                gt_data = np.load(gt_path)
                
                exp_mask = get_top_k_mask(attr, k_ratio=0.1)
                # ACDC 目标类别是 3 (LV)
                gt_binary = (gt_data == 3).astype(np.uint8)
                
                dice, iou = calculate_dice_iou(exp_mask, gt_binary)
                results.append({"dataset": "ACDC", "id": frame_name, "dice": dice, "iou": iou})
                print(f"  {frame_name}: Dice={dice:.4f}, IoU={iou:.4f}")

    if not results:
        print("No results found to evaluate.")
        return

    # --- 3. 绘图 ---
    datasets = sorted(list(set(r['dataset'] for r in results)))
    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 6), squeeze=False)
    
    for i, ds_name in enumerate(datasets):
        ds_results = [r for r in results if r['dataset'] == ds_name]
        ids = [r['id'] for r in ds_results]
        dices = [r['dice'] for r in ds_results]
        ious = [r['iou'] for r in ds_results]
        
        x = np.arange(len(ids))
        width = 0.35
        
        axes[0, i].bar(x - width/2, dices, width, label='Dice', color='skyblue')
        axes[0, i].bar(x + width/2, ious, width, label='IoU', color='salmon')
        
        axes[0, i].set_title(f"{ds_name} SHAP Quality ({model_type})")
        axes[0, i].set_xticks(x)
        axes[0, i].set_xticklabels(ids, rotation=45, ha='right')
        axes[0, i].set_ylim(0, 1.0)
        axes[0, i].legend()
        axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    output_path = f"shap/dice_iou_comparison_{model_type}.png"
    plt.savefig(output_path)
    print(f"\nComparison plot saved to {output_path}")
    
    # 保存汇总数据
    with open(f"shap/summary_metrics_{model_type}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate_shap()
