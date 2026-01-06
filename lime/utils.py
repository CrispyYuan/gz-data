import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.segmentation import mark_boundaries

def load_nifti_image(path):
    """
    读取 NIfTI 格式 (.nii.gz) 的医学影像。
    返回 numpy 数组 (H, W, D) 或 (C, H, W, D)。
    """
    img = nib.load(path)
    data = img.get_fdata()
    return data

def normalize_image(image):
    """
    Z-score 标准化，与 Step 0 文档中的预处理保持一致。
    """
    mean = np.mean(image)
    std = np.std(image)
    if std == 0:
        return image
    return (image - mean) / std

def get_best_slice(mask_volume):
    """
    自动寻找肿瘤面积最大的切片索引，用于可视化展示。
    输入: 3D Mask (H, W, D)
    返回: 切片索引 index
    """
    # 假设 mask 中非 0 值为肿瘤
    slice_areas = np.sum(mask_volume > 0, axis=(0, 1))
    best_slice = np.argmax(slice_areas)
    return best_slice

def save_comparison_plot(image_slice, mask_slice, lime_heatmap, lime_boundaries, save_path):
    """
    生成并保存 Step 2 所需的可视化面板。
    包含：原始图像、Ground Truth、LIME 热力图、LIME 解释边界图。
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. 原始图像
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title("Original MRI Slice")
    axes[0].axis('off')

    # 2. 分割结果 (Ground Truth)
    axes[1].imshow(image_slice, cmap='gray')
    axes[1].imshow(mask_slice, cmap='jet', alpha=0.5) # 叠加显示
    axes[1].set_title("Ground Truth Segmentation")
    axes[1].axis('off')

    # 3. LIME Heatmap (重要性热力图)
    # 红色代表正向贡献，蓝色代表负向贡献
    im = axes[2].imshow(lime_heatmap, cmap='RdBu_r', vmin=-np.max(np.abs(lime_heatmap)), vmax=np.max(np.abs(lime_heatmap)))
    axes[2].set_title("LIME Importance Heatmap")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. LIME Boundaries (超像素解释)
    # 显示解释结果的前 3 个最重要区域
    axes[3].imshow(mark_boundaries(image_slice, lime_boundaries))
    axes[3].set_title("LIME Superpixel Explanations")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"可视化面板已保存至: {save_path}")

def generate_heatmap_from_lime(explanation, original_image):
    """
    将 LIME 的 (superpixel_index, weight) 转换为像素级的 heatmap。
    这对 Step 3 的 Overlap@k 计算至关重要。
    """
    # 获取最上面解释的标签（通常是肿瘤类，label=1）
    # LIME Explanation 对象的 local_exp 是一个字典 {label: [(feature_idx, weight), ...]}
    ind = explanation.top_labels[0]
    
    # 获取解释权重
    dict_heatmap = dict(explanation.local_exp[ind])
    
    # 初始化 heatmap
    heatmap = np.zeros(original_image.shape[:2])
    
    # explanation.segments 存储了超像素的分割掩码
    segments = explanation.segments
    
    for superpixel_idx, weight in dict_heatmap.items():
        heatmap[segments == superpixel_idx] = weight
        
    return heatmap