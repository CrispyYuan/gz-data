import torch
import torch.nn as nn
import torch.nn.functional as F

class nnUNetBlock(nn.Module):
    """
    nnU-Net 标准卷积块: (Conv -> InstanceNorm -> LeakyReLU) * 2
    nnU-Net 默认使用 InstanceNorm 和 LeakyReLU (slope=0.01)
    """
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        if dim == 2:
            conv_layer = nn.Conv2d
            norm_layer = nn.InstanceNorm2d
        else:
            conv_layer = nn.Conv3d
            norm_layer = nn.InstanceNorm3d

        self.block = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            norm_layer(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            conv_layer(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            norm_layer(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class nnUNetV2(nn.Module):
    """
    nnU-Net V2 架构实现
    特点: 
    1. 使用 InstanceNorm + LeakyReLU
    2. 支持 Deep Supervision (深监督)，在推理时可关闭
    3. 灵活支持 2D/3D
    """
    def __init__(self, in_channels, out_channels, dim=2, features=[32, 64, 128, 256, 512], deep_supervision=True):
        super().__init__()
        self.dim = dim
        self.deep_supervision = deep_supervision
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.seg_heads = nn.ModuleList() # 用于深监督的分割头
        
        if dim == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            up_conv = nn.ConvTranspose2d
            seg_layer = nn.Conv2d
        else:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            up_conv = nn.ConvTranspose3d
            seg_layer = nn.Conv3d

        # --- Encoder ---
        curr_channels = in_channels
        for feature in features:
            self.encoder.append(nnUNetBlock(curr_channels, feature, dim=dim))
            curr_channels = feature

        # --- Decoder ---
        # 从最底层向上构建
        for feature in reversed(features[:-1]):
            # 上采样层
            self.decoder.append(up_conv(feature * 2, feature, kernel_size=2, stride=2))
            # 卷积块
            self.decoder.append(nnUNetBlock(feature * 2, feature, dim=dim))
            # 深监督分割头 (每个尺度一个)
            if self.deep_supervision:
                self.seg_heads.append(seg_layer(feature, out_channels, kernel_size=1))

        # 如果不开启深监督，或者作为最后的输出头
        if not self.deep_supervision or len(self.seg_heads) == 0:
            self.final_seg_head = seg_layer(features[0], out_channels, kernel_size=1)
        else:
            # 最后一个分割头（最高分辨率）
            self.final_seg_head = self.seg_heads[-1]

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                skip_connections.append(x)
                x = self.pool(x)
        
        # Decoder
        skip_connections = skip_connections[::-1]
        outputs = []
        
        for i in range(0, len(self.decoder), 2):
            # 1. Upsample
            x = self.decoder[i](x)
            skip = skip_connections[i // 2]
            
            # 2. Concat
            x = torch.cat((skip, x), dim=1)
            
            # 3. Conv
            x = self.decoder[i+1](x)
            
            # 4. Deep Supervision Output
            if self.deep_supervision and (i // 2) < len(self.seg_heads):
                outputs.append(self.seg_heads[i // 2](x))
        
        # nnU-Net V2 的输出顺序通常是从高分辨率到低分辨率
        # 我们这里反转一下，让 outputs[0] 始终是最高分辨率
        if self.deep_supervision:
            outputs = outputs[::-1]
            return outputs
        else:
            return self.final_seg_head(x)

def get_nnunet_acdc():
    """ ACDC 数据集 (2D) """
    return nnUNetV2(in_channels=1, out_channels=4, dim=2, features=[32, 64, 128, 256, 512])

def get_nnunet_brats():
    """ BraTS 数据集 (3D) """
    return nnUNetV2(in_channels=4, out_channels=4, dim=3, features=[32, 64, 128, 256])

if __name__ == "__main__":
    # 测试 nnU-Net V2 2D + Deep Supervision
    model_2d = get_nnunet_acdc()
    x2d = torch.randn((2, 1, 128, 128))
    y2d = model_2d(x2d)
    print(f"nnU-Net 2D Output type: {type(y2d)}")
    if isinstance(y2d, list):
        print(f"Number of DS outputs: {len(y2d)}")
        for i, out in enumerate(y2d):
            print(f"  Scale {i} shape: {out.shape}")

    # 测试 nnU-Net V2 3D
    model_3d = get_nnunet_brats()
    x3d = torch.randn((1, 4, 64, 64, 64))
    y3d = model_3d(x3d)
    print(f"nnU-Net 3D Output (Highest Scale) shape: {y3d[0].shape}")
