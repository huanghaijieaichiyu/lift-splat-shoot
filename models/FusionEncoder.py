import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RepVit import RepViTBlock
from .common import Conv, PSA, SPPF, C2f, C2fCIB, Gencov, EMA, ChannelAttention
from .CamEncoder_3d import CamEncoder_3d


class LidarEncoder(nn.Module):
    """LiDAR特征编码器"""

    def __init__(self, in_channels, out_channels):
        super(LidarEncoder, self).__init__()

        # LiDAR点云特征提取网络
        self.backbone = nn.Sequential(
            Conv(in_channels, 64, 3, 2),
            RepViTBlock(64, 128, 3, 2),
            SPPF(128, 128),
            RepViTBlock(128, 256, 3, 2),
            C2f(256, 512, 1)
        )

        # 特征增强

        # 输出头
        self.out_conv = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W] - 点云BEV特征图
        Returns:
            features: [B, out_channels, H', W'] - 编码后的LiDAR特征
        """
        feat = self.backbone(x)
        return self.out_conv(feat)


class CrossAttention(nn.Module):
    """跨模态注意力模块"""

    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.q_conv = nn.Conv2d(dim, dim, 1)
        self.k_conv = nn.Conv2d(dim, dim, 1)
        self.v_conv = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5

    def forward(self, x1, x2):
        """
        计算跨模态注意力
        Args:
            x1: [B, C, H, W] - 第一个模态特征
            x2: [B, C, H, W] - 第二个模态特征
        Returns:
            out: [B, C, H, W] - 融合后的特征
        """
        B, C, H, W = x1.shape

        q = self.q_conv(x1).view(B, C, -1)
        k = self.k_conv(x2).view(B, C, -1)
        v = self.v_conv(x2).view(B, C, -1)

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))
        return out.view(B, C, H, W)


class FusionEncoder(nn.Module):
    """
    相机和LiDAR特征融合编码器
    """

    def __init__(self, D, C, downsample, lidar_channels=64):
        super(FusionEncoder, self).__init__()

        # 相机编码器
        self.cam_encoder = CamEncoder_3d(D, C, downsample)

        # LiDAR编码器
        self.lidar_encoder = LidarEncoder(lidar_channels, C)

        # 跨模态注意力
        self.cross_attn = CrossAttention(C)

        # 特征融合
        self.fusion_conv = nn.Sequential(
            Conv(C*2, C, 1),
            PSA(C, C),
            EMA(C)
        )

    def forward(self, imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev):
        """
        前向传播
        Args:
            imgs: [B*N, 3, H, W] - 相机图像
            rots: [B*N, 3, 3] - 旋转矩阵
            trans: [B*N, 3] - 平移向量
            intrins: [B*N, 3, 3] - 相机内参
            post_rots: [B*N, 3, 3] - 后处理旋转
            post_trans: [B*N, 3] - 后处理平移
            lidar_bev: [B, C, H, W] - LiDAR BEV特征图
        Returns:
            fused_feat: [B, C, D, H', W'] - 融合后的3D特征体素
        """
        # 1. 提取相机特征
        cam_feat = self.cam_encoder(
            imgs, rots, trans, intrins, post_rots, post_trans)
        B, C, D, H, W = cam_feat.shape
        cam_feat = cam_feat.mean(dim=2)  # 压缩深度维度 [B, C, H, W]

        # 2. 提取LiDAR特征
        lidar_feat = self.lidar_encoder(lidar_bev)

        # 3. 跨模态注意力
        cam_enhanced = self.cross_attn(cam_feat, lidar_feat)
        lidar_enhanced = self.cross_attn(lidar_feat, cam_feat)

        # 4. 特征融合
        fused = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion_conv(fused)

        # 5. 恢复深度维度
        fused = fused.unsqueeze(2).repeat(1, 1, D, 1, 1)

        return fused


def create_fusion_encoder(D, C, downsample, lidar_channels=64):
    """
    创建融合编码器的工厂函数
    Args:
        D: 深度平面数量
        C: 特征通道数
        downsample: 下采样因子
        lidar_channels: LiDAR输入通道数
    Returns:
        FusionEncoder实例
    """
    return FusionEncoder(D, C, downsample, lidar_channels)
