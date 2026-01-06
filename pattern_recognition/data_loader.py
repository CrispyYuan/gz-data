import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch.nn.functional as F

class ACDCDataset(Dataset):
    """
    ACDC 数据集加载器 (适配已预处理的数据: 重采样, Z-score, ROI 裁剪)
    直接加载指定目录 (如 train/val/test) 下的所有病例，并按切片返回 (2D)
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        if not os.path.exists(data_dir):
            return

        # 遍历目录下所有病例文件夹
        file_names = [d for d in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, d))]

        for patient in file_names:
            patient_dir = os.path.join(data_dir, patient)
            for f in sorted(os.listdir(patient_dir)):
                # 排除 _gt 和 _4d 文件，支持 .nii, .nii.gz, .npy
                if (f.endswith(".nii.gz") or f.endswith(".nii") or f.endswith(".npy")) and "_gt" not in f and "_4d" not in f:
                    img_path = os.path.join(patient_dir, f)
                    if f.endswith(".nii.gz"):
                        lab_path = img_path.replace(".nii.gz", "_gt.nii.gz")
                    elif f.endswith(".nii"):
                        lab_path = img_path.replace(".nii", "_gt.nii")
                    else:
                        lab_path = img_path.replace(".npy", "_gt.npy")
                    
                    if os.path.exists(lab_path):
                        try:
                            # 获取切片数量
                            if f.endswith(".npy"):
                                data_shape = np.load(img_path, mmap_mode='r').shape
                            else:
                                data_shape = nib.load(img_path).shape
                            
                            num_slices = data_shape[2] if len(data_shape) == 3 else 1
                            for s in range(num_slices):
                                self.images.append((img_path, s))
                                self.labels.append((lab_path, s))
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, slice_idx = self.images[idx]
        lab_path, _ = self.labels[idx]
        
        if img_path.endswith(".nii.gz") or img_path.endswith(".nii"):
            data = nib.load(img_path).get_fdata()
            label_data = nib.load(lab_path).get_fdata()
        else:
            data = np.load(img_path)
            label_data = np.load(lab_path)
        
        if data.ndim == 3:
            image = data[:, :, slice_idx]
            label = label_data[:, :, slice_idx]
        else:
            image = data
            label = label_data
        
        # 转换为 Tensor: (1, H, W)
        image = torch.from_numpy(image).float().unsqueeze(0)
        label = torch.from_numpy(label).long()
        
        if self.transform:
            image = self.transform(image)
        return image, label

class BraTSDataset(Dataset):
    """
    BraTS 数据集加载器 (适配已预处理的数据: 重采样, Z-score, ROI 裁剪)
    直接加载指定目录 (如 train/val/test) 下的所有病例 (3D)
    支持内存缓存以加速训练
    """
    def __init__(self, data_dir, transform=None, cache=True):
        self.data_dir = data_dir
        self.transform = transform
        self.cache = cache
        self.cached_data = []
        
        if os.path.exists(data_dir):
            self.folders = [os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir)) 
                            if os.path.isdir(os.path.join(data_dir, d))]
        else:
            self.folders = []

        if self.cache and self.folders:
            print(f"Caching BraTS data from {data_dir} into RAM...")
            for idx in range(len(self.folders)):
                self.cached_data.append(self._load_item(idx))
            print(f"Caching complete. Total items: {len(self.cached_data)}")

    def _load_item(self, idx):
        folder = self.folders[idx]
        base_name = os.path.basename(folder)
        
        # 模态顺序对齐 models/UNet.py: (T1, T1c, T2, FLAIR)
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []
        for mod in modalities:
            path_nii = os.path.join(folder, f"{base_name}_{mod}.nii.gz")
            path_npy = os.path.join(folder, f"{base_name}_{mod}.npy")
            
            if os.path.exists(path_nii):
                # 使用 dataobj 并转换为 numpy 数组通常比 get_fdata() 快
                img = nib.load(path_nii)
                data = np.asanyarray(img.dataobj).astype(np.float32)
            elif os.path.exists(path_npy):
                data = np.load(path_npy).astype(np.float32)
            else:
                raise FileNotFoundError(f"Missing modality {mod} in {folder}")
            images.append(data)
        
        # 堆叠模态: (4, H, W, D) -> (4, D, H, W)
        image = np.stack(images, axis=0)
        image = torch.from_numpy(image).permute(0, 3, 1, 2)
        
        seg_path_nii = os.path.join(folder, f"{base_name}_seg.nii.gz")
        seg_path_npy = os.path.join(folder, f"{base_name}_seg.npy")
        
        if os.path.exists(seg_path_nii):
            img_seg = nib.load(seg_path_nii)
            label = np.asanyarray(img_seg.dataobj)
        elif os.path.exists(seg_path_npy):
            label = np.load(seg_path_npy)
        else:
            raise FileNotFoundError(f"Missing segmentation in {folder}")
            
        # BraTS 2021 标签通常为 [0, 1, 2, 4]，将 4 映射为 3 以适配 4 类别输出
        label = label.copy() # 避免修改原始数据（如果是 mmap）
        label[label == 4] = 3
            
        # (H, W, D) -> (D, H, W)
        label = torch.from_numpy(label).long().permute(2, 0, 1)
        
        return image, label

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        if self.cache:
            image, label = self.cached_data[idx]
        else:
            image, label = self._load_item(idx)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
