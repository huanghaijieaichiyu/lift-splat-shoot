import torch.nn as nn

from utils.RepVit import RepViTBlock
from utils.common import Conv, SPPF, C2f


class CamEncoder_3d(nn.Module):

    def __init__(self, D, C, downsample):
        super(CamEncoder_3d, self).__init__()
        self.D = D
        self.C = C

        # 使用纯Conv构建的backbone，避免通道不匹配问题
        self.backbone = nn.Sequential(
            Conv(3, 32, 3, 2),  # 下采样2倍
            RepViTBlock(32, 64, 3, 2),  # 下采样2倍
            SPPF(64, 64),
            RepViTBlock(64, 64, 3, 2),  # 下采样2倍
            C2f(64, 128, 1),
            Conv(128, 256, 1),
            Conv(256, 512, 1)
        )

        # 深度预测和特征生成头
        self.depth_head = Conv(512, D, k=1)
        self.feat_head = Conv(512, C, k=1)

    def get_depth_dist(self, x):
        """计算深度分布"""
        return x.softmax(dim=1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B*N, 3, H, W] - 相机图像
        Returns:
            features: [B*N, C, D, H//downsample, W//downsample] - 3D特征体素
        """
        # 提取图像特征
        features = self.backbone(x)

        # 预测深度分布
        depth = self.depth_head(features)
        depth = self.get_depth_dist(depth)  # [B*N, D, H', W']

        # 预测特征
        feats = self.feat_head(features)  # [B*N, C, H', W']

        # 通过深度分布加权特征，得到3D特征
        # 将特征与每个深度平面相乘，得到3D特征体素
        output_feats = depth.unsqueeze(
            1) * feats.unsqueeze(2)  # [B*N, C, D, H', W']
        print(output_feats.size())

        return output_feats


def create_cam_encoder(D, C, downsample):
    """
    创建相机编码器的工厂函数
    Args:
        D: 深度平面数量
        C: 特征通道数
        downsample: 下采样因子
    Returns:
        ViTCamEncoder实例
    """
    return CamEncoder_3d(D, C, downsample)
