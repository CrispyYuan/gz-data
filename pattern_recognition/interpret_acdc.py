import os
import torch
import numpy as np
import shap
import json
import matplotlib.pyplot as plt
import nibabel as nib
from data_loader import ACDCDataset
from models.UNet import get_unet_acdc
from models.nnUNetV2 import get_nnunet_acdc
from ExplainerModelWrapper import ExplainerModelWrapper
from train_utils import load_config

def interpret_acdc_shap():
    # 1. 加载配置与设备
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型
    model_type = config.get('interpret', {}).get('model', 'unet')
    if model_type == "unet":
        model = get_unet_acdc()
        checkpoint_path = "./checkpoints/unet/acdc/unet_best.pth"
    else:
        model = get_nnunet_acdc()
        checkpoint_path = "./checkpoints/nnunet/acdc/nnunet_acdc_best.pth"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 使用包装器，确保输出为 Softmax 概率
    wrapped_model = ExplainerModelWrapper(model)

    # 3. 准备数据
    data_path = os.path.join(config['path']['data_path'], "ACDC_Processed")
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    # 背景数据集 (用于 SHAP 参考)
    # ACDC 是 2D 切片，计算量相对较小
    train_ds = ACDCDataset(train_dir)
    background_indices = np.random.choice(len(train_ds), 10, replace=False)
    background_images = []
    for idx in background_indices:
        img, _ = train_ds[idx]
        background_images.append(img)
    
    background_data = torch.stack(background_images).to(device)

    # 4. 初始化 SHAP 解释器
    explainer = shap.GradientExplainer(wrapped_model, background_data)

    # 5. 遍历指定的病例进行解释
    patient_ids = config.get('interpret', {}).get('acdc_patient_ids', [])
    if not patient_ids:
        print("No ACDC patient IDs specified in config.yaml")
        return

    output_dir = f"shap/acdc_{model_type}"
    os.makedirs(output_dir, exist_ok=True)

    for pid in patient_ids:
        print(f"\nProcessing ACDC Patient: {pid}")
        patient_folder = os.path.join(val_dir, pid)
        if not os.path.exists(patient_folder):
            print(f"Warning: Folder {patient_folder} not found, skipping.")
            continue

        # 加载该病例的所有切片
        # 识别包含影像的 frame 文件 (排除 _gt 和 _4d)
        files = sorted([f for f in os.listdir(patient_folder) if '_frame' in f and f.endswith('.nii') and '_gt' not in f and '_4d' not in f])
        
        for frame_file in files:
            frame_name = frame_file.replace('.nii', '')
            print(f"  Explaining {frame_name}...")
            
            img_nii = nib.load(os.path.join(patient_folder, frame_file))
            data = img_nii.get_fdata() # (128, 128, 10)
            
            # 选取中间切片进行解释
            mid_slice = data.shape[2] // 2
            test_img = torch.from_numpy(data[:, :, mid_slice]).float().unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 128, 128)

            # 计算 SHAP 值
            # ACDC 有 4 个类别: 0:BG, 1:RV, 2:MYO, 3:LV
            # 我们解释 Class 3 (LV - 左心室)
            target_class = 3
            shap_values = explainer.shap_values(test_img, nsamples=50)
            
            if isinstance(shap_values, list):
                attr = shap_values[target_class][0] # (1, 128, 128)
            else:
                # (B, C_in, H, W, C_out)
                attr = shap_values[0, :, :, :, target_class]

            # 6. 可视化
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # 原始图像
            axes[0].imshow(test_img[0, 0].cpu().numpy(), cmap='gray')
            axes[0].set_title(f"Original ({frame_name})")
            axes[0].axis('off')
            
            # SHAP 热力图
            v_max = np.abs(attr).max()
            im = axes[1].imshow(attr[0], cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            axes[1].set_title(f"SHAP (Class {target_class})")
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])

            # 分割结果与不确定性
            with torch.no_grad():
                torch.set_grad_enabled(False)
                probs_full = wrapped_model(test_img)
                torch.set_grad_enabled(True)
                pred_mask = torch.argmax(probs_full, dim=1)[0]
                entropy_map = wrapped_model.get_entropy(probs_full)[0]

            axes[2].imshow(pred_mask.cpu().numpy(), cmap='jet')
            axes[2].set_title("Segmentation")
            axes[2].axis('off')

            im_ent = axes[3].imshow(entropy_map.cpu().numpy(), cmap='hot')
            axes[3].set_title("Uncertainty (Entropy)")
            axes[3].axis('off')
            plt.colorbar(im_ent, ax=axes[3])

            # 7. 定量分析
            total_attr_torch = torch.from_numpy(np.abs(attr)).to(device)
            overlap = wrapped_model.calculate_overlap_at_k(total_attr_torch[0], entropy_map, k_ratio=0.1)
            print(f"    Overlap@0.1: {overlap:.4f}")

            plt.tight_layout()
            output_fig = os.path.join(output_dir, f"shap_{frame_name}.png")
            plt.savefig(output_fig)
            plt.close()

            # 8. 保存数据
            shap_save_path = os.path.join(output_dir, f"shap_values_{frame_name}.npy")
            np.save(shap_save_path, attr)
            
            # 保存 Ground Truth 切片用于后续 Dice/IoU 计算
            gt_path = os.path.join(patient_folder, frame_file.replace('.nii', '_gt.nii'))
            gt_slice = nib.load(gt_path).get_fdata()[:, :, mid_slice]
            gt_save_path = os.path.join(output_dir, f"gt_{frame_name}.npy")
            np.save(gt_save_path, gt_slice)

            metrics = {
                "patient_id": pid,
                "frame": frame_name,
                "target_class": target_class,
                "overlap_at_0.1": overlap,
                "shap_values_shape": list(attr.shape),
                "mean_abs_shap": float(np.abs(attr).mean())
            }
            with open(os.path.join(output_dir, f"metrics_{frame_name}.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            
            print(f"    Data saved to {output_dir}")

if __name__ == "__main__":
    interpret_acdc_shap()
