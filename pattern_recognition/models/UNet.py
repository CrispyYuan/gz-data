import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    双层卷积块: (Conv -> BatchNorm -> ReLU) * 2
    支持 2D 和 3D 模式
    """
    def __init__(self, in_channels, out_channels, dim=2):
        super().__init__()
        if dim == 2:
            conv_layer = nn.Conv2d
            norm_layer = nn.BatchNorm2d
        elif dim == 3:
            conv_layer = nn.Conv3d
            norm_layer = nn.BatchNorm3d
        else:
            raise ValueError("dim must be 2 or 3")

        self.block = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            conv_layer(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """
    通用 U-Net 实现，支持 2D (ACDC) 和 3D (BraTS) 数据集
    """
    def __init__(self, in_channels, out_channels, dim=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.dim = dim
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        if dim == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            up_conv = nn.ConvTranspose2d
            final_conv = nn.Conv2d
        else:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            up_conv = nn.ConvTranspose3d
            final_conv = nn.Conv3d

        # Encoder (Downsampling)
        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature, dim=dim))
            in_channels = feature

        # Decoder (Upsampling)
        for feature in reversed(features):
            self.ups.append(
                up_conv(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature, dim=dim))

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dim=dim)
        
        # Final Layer
        self.final_conv = final_conv(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            # 1. Upsample
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # 2. Handle potential size mismatch (due to odd input dimensions)
            if x.shape != skip_connection.shape:
                # 对齐空间维度
                diff = [skip_connection.size(i) - x.size(i) for i in range(2, x.ndim)]
                pad = []
                for d in reversed(diff):
                    pad.extend([d // 2, d - d // 2])
                x = F.pad(x, pad)

            # 3. Concatenate and Conv
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def get_unet_acdc():
    """
    针对 ACDC 数据集的 U-Net 配置 (通常为 2D)
    输入: 1 通道 (MRI)
    输出: 4 类别 (Background, RV, MYO, LV)
    """
    return UNet(in_channels=1, out_channels=4, dim=2)

def get_unet_brats():
    """
    针对 BraTS 数据集的 U-Net 配置 (3D)
    输入: 4 通道 (T1, T1c, T2, FLAIR)
    输出: 4 类别 (Background, ET, TC, WT)
    """
    return UNet(in_channels=4, out_channels=4, dim=3)

if __name__ == "__main__":
    # 测试 2D U-Net (ACDC)
    model_2d = get_unet_acdc()
    x2d = torch.randn((8, 1, 224, 224))
    y2d = model_2d(x2d)
    print(f"2D UNet Output Shape: {y2d.shape}") # Expected: [8, 4, 224, 224]

    # 测试 3D U-Net (BraTS)
    model_3d = get_unet_brats()
    x3d = torch.randn((1, 4, 64, 64, 64)) # Batch size 1 for testing memory
    y3d = model_3d(x3d)
    print(f"3D UNet Output Shape: {y3d.shape}") # Expected: [1, 4, 64, 64, 64]
