# 模式识别作业服务器说明

1. 上服务器之后尽量都在 `/gz-data` 目录下工作，不要直接在`~`目录工作，因为系统盘只有**30GB**，爆了就得重开，`/gz-data`是数据盘可以扩容。
2. 除此以外，`/gz-fs`是网盘，有需要使用可以联系`谢天恩`。
3. 环境的话基本都在`base`配了，应该都没问题，有问题再在群里找我，或者自行解决。尽量避免新建环境因为会占用系统盘。
4. 两个数据集都已经处理好，在目录`/dataset`下，可以自行查看并使用。
5. 跑长时间的任务时尽量使用`tmux`，尽量避免长时间挂着vscode占用内存（因为之前内存爆过导致任务中途暂停了）。并且最好跟我们两位说一声不然服务器余额不足会强制停机导致任务中断很难受。
6. 两个基线模型都分别在两个数据集上训练完成，直接读取ckpt就可以使用。使用的时候可以看看下面的`ExplainerModelWrapper`说明。
7. 如果可以使用cline，请谨慎使用，并且选择`gemini-3-flash`模型，这个最近便宜点(T_T)
8. 为了统一比较，后续任务要跑的五个案例都写在`config.yaml`里，直接读取即可。











## 已完成 `pattern_recognition/ExplainerModelWrapper.py` 的完善工作。主要更新如下：

1.  **模型推理适配**：
    *   `forward` 方法现在可以自动识别并处理 nnU-Net V2 的多尺度输出（Deep Supervision），默认提取最高分辨率层。
    *   确保输出经过 `softmax`，以满足 SHAP 和 LIME 对概率输入的要求。

2.  **不确定性分析**：
    *   `get_entropy` 方法实现了像素级熵（Entropy）的计算，作为模型预测的不确定性指标。

3.  **Grad-CAM 支持**：
    *   新增 `register_hooks` 方法，支持动态注册 forward 和 backward hooks。
    *   新增 `get_gradcam` 方法，实现了基于梯度的特征加权热力图生成，支持 2D (H, W) 和 3D (D, H, W) 图像。

4.  **定量评估指标**：
    *   实现了 `get_top_k_mask`：用于提取得分图（解释图或不确定性图）中前 k% 的高响应区域。
    *   实现了 `calculate_overlap_at_k` 和 `calculate_recall_at_k`：用于定量分析解释结果与不确定性区域的重合度，支持你进行可解释性工具的对比作业。

5.  **训练配置参考**：
    *   在代码注释中补充了关于 AdamW 优化器和 CosineAnnealingLR 调度器的配置建议，以符合你对基线模型的设置要求。

可以直接在推理脚本中使用该 Wrapper 包装你的 nnU-Net 或 U-Net 模型。

## U-Net 模型说明 (`UNet.py`)

新增了通用的 U-Net 实现，支持 2D 和 3D 模式，适用于 ACDC 和 BraTS 数据集：

1.  **ACDC 数据集 (2D)**:
    *   使用 `get_unet_acdc()` 获取配置。
    *   输入通道：1 (MRI)，输出类别：4。
2.  **BraTS 数据集 (3D)**:
    *   使用 `get_unet_brats()` 获取配置。
    *   输入通道：4 (T1, T1c, T2, FLAIR)，输出类别：4。

**训练建议设置**:
*   **优化器**: `AdamW`
*   **学习率调度**: `CosineAnnealingLR`
*   **Batch Size**: 8 或 16 (根据显存调整)

## nnU-Net V2 模型说明 (`nnUNetV2.py`)

实现了符合 nnU-Net V2 标准的架构：

1.  **核心组件**: 使用 `InstanceNorm` 和 `LeakyReLU` (slope=0.01)。
2.  **深监督 (Deep Supervision)**: 默认开启，返回多个尺度的分割结果。`ExplainerModelWrapper` 已适配此输出格式，会自动提取最高分辨率层进行解释。
3.  **数据集适配**:
    *   `get_nnunet_acdc()`: 2D 配置。
    *   `get_nnunet_brats()`: 3D 配置。
