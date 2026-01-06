import os
import torch
import numpy as np
import shap
import json
import matplotlib.pyplot as plt
import nibabel as nib
from data_loader import BraTSDataset
from models.UNet import get_unet_brats
from models.nnUNetV2 import get_nnunet_brats
from ExplainerModelWrapper import ExplainerModelWrapper
from train_utils import load_config

def interpret_brats_shap():
    # 1. 加载配置与设备
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载模型
    # model = get_unet_brats()
    model = get_nnunet_brats()
    # checkpoint_path = "./checkpoints/unet/brats/unet_best.pth"
    checkpoint_path = "./checkpoints/nnunet/brats/nnunet_best.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 使用包装器，确保输出为 Softmax 概率
    wrapped_model = ExplainerModelWrapper(model)

    # 3. 准备数据
    data_path = os.path.join(config['path']['data_path'], "BraTS_Processed")
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")

    # 背景数据集 (用于 SHAP 参考)
    train_ds = BraTSDataset(train_dir, cache=False)
    background_indices = np.random.choice(len(train_ds), 2, replace=False)
    background_images = []
    target_size = (32, 32, 32)
    for idx in background_indices:
        img, _ = train_ds[idx]
        img_sub = torch.nn.functional.interpolate(img.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False)
        background_images.append(img_sub.squeeze(0))
    
    background_data = torch.stack(background_images).to(device)

    # 4. 初始化 SHAP 解释器
    explainer = shap.GradientExplainer(wrapped_model, background_data)

    # 5. 遍历指定的病例进行解释
    patient_ids = config.get('interpret', {}).get('patient_ids', [])
    if not patient_ids:
        print("No patient IDs specified in config.yaml")
        return

    for pid in patient_ids:
        print(f"\nProcessing Patient: {pid}")
        patient_folder = os.path.join(val_dir, pid)
        if not os.path.exists(patient_folder):
            print(f"Warning: Folder {patient_folder} not found, skipping.")
            continue

        # 加载特定病例数据
        # 这里我们直接使用 BraTSDataset 的逻辑来加载
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        for mod in modalities:
            path_nii = os.path.join(patient_folder, f"{pid}_{mod}.nii.gz")
            img_data = np.asanyarray(nib.load(path_nii).dataobj).astype(np.float32)
            images.append(img_data)
        
        test_img = torch.from_numpy(np.stack(images, axis=0)).permute(0, 3, 1, 2)
        test_img_input = torch.nn.functional.interpolate(test_img.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).to(device)

        # 计算 SHAP 值
        target_class = 3
        print(f"Calculating SHAP values for Class {target_class}...")
        shap_values = explainer.shap_values(test_img_input, nsamples=50)
        
        if isinstance(shap_values, list):
            attr = shap_values[target_class][0]
        else:
            attr = shap_values[0, :, :, :, :, target_class]

        # 6. 可视化
        slice_idx = 16 
        mod_names = ['T1', 'T1ce', 'T2', 'FLAIR']
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        for i in range(4):
            axes[0, i].imshow(test_img_input[0, i, slice_idx].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f"Original {mod_names[i]}")
            axes[0, i].axis('off')
            
            shap_slice = attr[i, slice_idx]
            v_max = np.abs(shap_slice).max()
            im = axes[1, i].imshow(shap_slice, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            axes[1, i].set_title(f"SHAP {mod_names[i]}")
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

        with torch.no_grad():
            torch.set_grad_enabled(False)
            probs_full = wrapped_model(test_img_input)
            torch.set_grad_enabled(True)
            pred_mask = torch.argmax(probs_full, dim=1)[0]
            entropy_map = wrapped_model.get_entropy(probs_full)[0]

        axes[2, 0].imshow(pred_mask[slice_idx].cpu().numpy(), cmap='jet')
        axes[2, 0].set_title("Segmentation Result")
        axes[2, 0].axis('off')

        im_ent = axes[2, 1].imshow(entropy_map[slice_idx].cpu().numpy(), cmap='hot')
        axes[2, 1].set_title("Uncertainty (Entropy)")
        axes[2, 1].axis('off')
        plt.colorbar(im_ent, ax=axes[2, 1], fraction=0.046, pad=0.04)

        axes[2, 2].axis('off')
        axes[2, 3].axis('off')

        # 7. 定量分析
        total_attr = np.abs(attr).sum(axis=0)
        total_attr_torch = torch.from_numpy(total_attr).to(device)
        overlap = wrapped_model.calculate_overlap_at_k(total_attr_torch, entropy_map, k_ratio=0.1)
        print(f"Overlap@0.1 for {pid}: {overlap:.4f}")

        plt.tight_layout()
        output_fig = f"./shap/shap_result_{pid}_nnunet.png"
        plt.savefig(output_fig)
        plt.close()
        print(f"Result saved to {output_fig}")

        # 8. 保存数据到本地
        model_type = config.get('interpret', {}).get('model', 'unet')
        output_dir = f"shap/{model_type}_brats"
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 SHAP 值 (numpy 格式)
        shap_save_path = os.path.join(output_dir, f"shap_values_{pid}.npy")
        np.save(shap_save_path, attr)
        
        # 保存定量分析数值 (JSON 格式)
        metrics = {
            "patient_id": pid,
            "target_class": target_class,
            "overlap_at_0.1": overlap,
            "shap_values_shape": list(attr.shape),
            "mean_abs_shap": float(np.abs(attr).mean())
        }
        metrics_save_path = os.path.join(output_dir, f"metrics_{pid}.json")
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        # 同时移动可视化图片到该目录
        os.rename(output_fig, os.path.join(output_dir, output_fig))
        
        print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    interpret_brats_shap()
