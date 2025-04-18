from models.common import (
    Conv, C2f, SPPF, PSA, EMA,
    C2fCIB, ChannelAttention, Gencov, SPPELAN
)
from models.RepVit import RepViTBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从common导入相关组件


class XFeatCamEncoder(nn.Module):
    """
    基于XFeat设计的相机特征编码器
    结合了XFeat的轻量高效设计和LSS模型结构
    特点：
    1. 轻量级卷积结构
    2. 高效特征提取
    3. 多尺度特征融合
    """

    def __init__(self, D, C, downsample=16, enhance_features=True):
        super(XFeatCamEncoder, self).__init__()

        self.D = D  # 深度维度
        self.C = C  # 特征通道数
        self.downsample = downsample
        self.enhance_features = enhance_features

        # 特征提取主干网络 - 使用配置更适合BEV任务的轻量级结构
        # 第一层 - 高效初始特征提取
        self.stage1 = nn.Sequential(
            Conv(3, 32, k=3, s=2, p=1),  # 下采样2倍
            RepViTBlock(32, 32, 3, 1)    # 保持维度
        )

        # 第二层 - 特征下采样和增强
        self.stage2 = nn.Sequential(
            Conv(32, 64, k=3, s=2, p=1),  # 下采样2倍
            C2f(64, 64, 1)                # 特征增强
        )

        # 第三层 - 进一步下采样和特征增强
        self.stage3 = nn.Sequential(
            Conv(64, 128, k=3, s=2, p=1),  # 下采样2倍
            PSA(128, 128)                  # 空间注意力
        )

        # 第四层 - 高级特征提取
        self.stage4 = nn.Sequential(
            SPPELAN(128, 256, 128),       # 空间金字塔特征增强
            RepViTBlock(256, 256, 1, 1)   # 维度不变
        )

        # 注意力增强模块 - 提高特征质量
        if enhance_features:
            self.channel_attn = ChannelAttention(256)
            self.ema = EMA(256)

        # 深度网络 - 预测深度分布
        self.depth_net = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, self.D, k=1)  # 只预测深度通道
        )

        # 特征网络 - 预测特征
        self.feat_net = nn.Sequential(
            Conv(256, 256, k=3, p=1),
            Conv(256, self.C, k=1)  # 只预测特征通道
        )

    def get_depth_dist(self, x, eps=1e-6):
        """
        获取深度分布 - 使用softmax并确保数值稳定性
        Args:
            x: 输入张量 [B, D, H, W]
            eps: 数值稳定性的小常数
        """
        # 对深度维度应用softmax
        return F.softmax(x, dim=1) + eps

    def forward(self, x):
        """
        前向传播 - 精确按照LSS模型结构处理特征和深度
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 特征张量 [B, C, D, H', W']，与体素投影兼容
        """
        B, _, H, W = x.shape

        # 核心特征提取
        x1 = self.stage1(x)         # [B, 32, H/2, W/2]
        x2 = self.stage2(x1)        # [B, 64, H/4, W/4]
        x3 = self.stage3(x2)        # [B, 128, H/8, W/8]
        x4 = self.stage4(x3)        # [B, 256, H/8, W/8]

        # 应用特征增强
        if self.enhance_features:
            x4 = x4 * self.channel_attn(x4)
            x4 = self.ema(x4)

        # 计算深度分布
        depth = self.depth_net(x4)  # [B, D, H/8, W/8]
        depth = self.get_depth_dist(depth)  # [B, D, H/8, W/8]

        # 计算特征
        feat = self.feat_net(x4)    # [B, C, H/8, W/8]

        # 确保深度和特征尺寸一致
        H_feat, W_feat = feat.shape[2:]
        if depth.shape[2:] != (H_feat, W_feat):
            depth = F.interpolate(depth, (H_feat, W_feat),
                                  mode='bilinear', align_corners=False)

        # 检查尺寸是否符合预期的下采样率
        target_h, target_w = H // self.downsample, W // self.downsample
        if H_feat != target_h or W_feat != target_w:
            # 确保特征图尺寸正确
            feat = F.interpolate(feat, (target_h, target_w),
                                 mode='bilinear', align_corners=False)
            depth = F.interpolate(
                depth, (target_h, target_w), mode='bilinear', align_corners=False)

        # 创建最终的特征体积 - 精确按照LSS的方式
        # 变换为 [B, C, D, H', W']
        # 使用广播机制: 将feat [B, C, H', W'] 乘以 depth [B, D, H', W']
        feat_volume = feat.unsqueeze(
            2) * depth.unsqueeze(1)  # [B, C, D, H', W']

        # 数值稳定性检查
        if torch.isnan(feat_volume).any() or torch.isinf(feat_volume).any():
            print(f"警告: 特征体积包含NaN或Inf，使用nan_to_num替换")
            feat_volume = torch.nan_to_num(
                feat_volume, nan=0.0, posinf=0.0, neginf=0.0)

        return feat_volume  # [B, C, D, H', W']


class XFeatBEVEncoder(nn.Module):
    """
    基于XFeat设计的BEV特征编码器
    轻量高效的BEV特征提取网络
    """

    def __init__(self, inC, outC):
        super(XFeatBEVEncoder, self).__init__()

        # 特征缩放系数
        s = 0.5
        c1 = int(64 * s)
        c2 = int(128 * s)
        c3 = int(256 * s)

        # 第一阶段 - 初始特征提取
        self.stage1 = nn.Sequential(
            Conv(inC, c1, k=3, s=1, p=1),
            RepViTBlock(c1, c1, 3, 2),
            C2f(c1, c1, 1)
        )

        # 第二阶段 - 特征下采样和增强
        self.stage2 = nn.Sequential(
            RepViTBlock(c1, c2, 3, 2),
            PSA(c2, c2)
        )

        # 第三阶段 - 特征增强
        self.stage3 = nn.Sequential(
            SPPF(c2, c2),
            ChannelAttention(c2)
        )

        # 上采样路径 - 恢复分辨率
        self.up1 = nn.Sequential(
            C2fCIB(c2, c3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # 最终输出层 - 预测头
        self.head = nn.Sequential(
            Conv(c3, c3, k=3, s=1, p=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(c3, outC, k=1, act=False, bn=False)  # 最终输出层无BN和激活
        )

        # 输出检查点
        self.use_checkpointing = False

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入BEV特征 [B, inC, H, W]
        Returns:
            预测结果 [B, outC, H, W]
        """
        # 使用torch.utils.checkpoint可以节省内存，但会增加计算时间
        if self.use_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            x1 = checkpoint(self.stage1, x)
            x2 = checkpoint(self.stage2, x1)
            x3 = checkpoint(self.stage3, x2)
            x = checkpoint(self.up1, x3)
            x = checkpoint(self.head, x)
        else:
            x1 = self.stage1(x)
            x2 = self.stage2(x1)
            x3 = self.stage3(x2)
            x = self.up1(x3)
            x = self.head(x)

        return x


class XFeatMultiModalEncoder(nn.Module):
    """
    多模态融合编码器 - 结合相机和LiDAR特征
    基于XFeat设计理念，提供高效特征融合
    """

    def __init__(self, camera_channels, lidar_channels, fusion_channels, out_channels):
        super(XFeatMultiModalEncoder, self).__init__()

        # 相机特征处理
        self.camera_conv = Conv(camera_channels, fusion_channels//2, k=1)

        # LiDAR特征处理
        self.lidar_conv = Conv(lidar_channels, fusion_channels//2, k=1)

        # 特征融合模块
        self.fusion_module = nn.Sequential(
            Conv(fusion_channels, fusion_channels, k=3, p=1),
            SPPELAN(fusion_channels, fusion_channels, fusion_channels//2),
            PSA(fusion_channels, fusion_channels)
        )

        # 输出头
        self.output_head = Conv(fusion_channels, out_channels, k=1)

        # 注意力模块 - 增强重要特征
        self.camera_attention = ChannelAttention(fusion_channels//2)
        self.lidar_attention = ChannelAttention(fusion_channels//2)

    def forward(self, camera_features, lidar_features):
        """
        前向传播
        Args:
            camera_features: 相机特征 [B, camera_channels, H, W]
            lidar_features: LiDAR特征 [B, lidar_channels, H, W]
        Returns:
            fused_features: 融合特征 [B, out_channels, H, W]
        """
        # 确保输入特征尺寸一致
        if camera_features.shape[2:] != lidar_features.shape[2:]:
            # 调整LiDAR特征尺寸匹配相机特征
            lidar_features = F.interpolate(
                lidar_features,
                size=camera_features.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 处理相机特征
        cam_feats = self.camera_conv(camera_features)
        cam_feats = cam_feats * self.camera_attention(cam_feats)

        # 处理LiDAR特征
        lidar_feats = self.lidar_conv(lidar_features)
        lidar_feats = lidar_feats * self.lidar_attention(lidar_feats)

        # 融合特征
        fused = torch.cat([cam_feats, lidar_feats], dim=1)
        fused = self.fusion_module(fused)

        # 输出
        output = self.output_head(fused)

        return output
