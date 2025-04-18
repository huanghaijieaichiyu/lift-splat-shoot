import torch
import torch.nn as nn
import math


class XFeatExtractor(nn.Module):
    """
    XFeat风格的轻量级特征提取网络，基于CVPR'24 XFeat论文
    用于高效图像特征提取的网络架构
    """

    def __init__(self, in_channels=3, out_channels=64, width_factor=1.0):
        super(XFeatExtractor, self).__init__()

        # 缩放通道数
        c1 = int(32 * width_factor)   # 第一层通道数
        c2 = int(64 * width_factor)   # 第二层通道数
        c3 = int(128 * width_factor)  # 第三层通道数
        c4 = int(256 * width_factor)  # 第四层通道数

        # 主干网络 - 受XFeat启发的轻量级架构
        self.conv1 = self._make_layer(
            in_channels, c1, kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            c1, c2, kernel_size=3, stride=2, padding=1)
        self.conv3 = self._make_layer(
            c2, c3, kernel_size=3, stride=2, padding=1)
        self.conv4 = self._make_layer(
            c3, c4, kernel_size=3, stride=1, padding=1)

        # 特征金字塔网络 - 用于多尺度特征融合
        self.lateral3 = nn.Conv2d(c3, c4, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(c2, c3, kernel_size=1, stride=1, padding=0)
        self.lateral1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # 最终输出层
        self.final_conv = nn.Conv2d(
            c4, out_channels, kernel_size=1, stride=1, padding=0)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        """创建卷积、BN、ReLU的基本块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _initialize_weights(self):
        """使用XFeat风格的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播，包含特征金字塔处理"""
        # 主干网络前向传播
        c1 = self.conv1(x)        # 1/2 分辨率
        c2 = self.conv2(c1)       # 1/4 分辨率
        c3 = self.conv3(c2)       # 1/8 分辨率
        c4 = self.conv4(c3)       # 1/8 分辨率 (保持分辨率)

        # 特征金字塔自顶向下路径
        p4 = c4
        p3 = self.lateral3(c3) + self.upsample(p4)
        p2 = self.lateral2(c2) + self.upsample(p3)

        # 最终特征
        out = self.final_conv(p4)

        return out


class XFeatDetector(nn.Module):
    """
    特征点检测器，基于XFeat架构，包括特征提取和兴趣点检测
    """

    def __init__(self, in_channels=3, feat_channels=64, num_keypoints=4096):
        super(XFeatDetector, self).__init__()

        # 特征提取网络
        self.extractor = XFeatExtractor(in_channels, feat_channels)

        # 检测头 - 预测关键点响应图
        self.detector = nn.Sequential(
            nn.Conv2d(feat_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # 描述子头 - 用于特征描述
        self.descriptor = nn.Sequential(
            nn.Conv2d(feat_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        self.num_keypoints = num_keypoints

    def detect_keypoints(self, score_map, k=None, nms_radius=4):
        """
        从响应图中检测关键点
        Args:
            score_map: 关键点响应图，形状为[B, 1, H, W]
            k: 每张图像提取的关键点数量，默认为self.num_keypoints
            nms_radius: 非极大值抑制的半径
        Returns:
            keypoints: 关键点坐标，形状为[B, k, 2]
            scores: 关键点分数，形状为[B, k]
        """
        if k is None:
            k = self.num_keypoints

        batch_size = score_map.shape[0]

        # 应用非极大值抑制
        # 简单方法：使用最大池化进行NMS
        nms_scores = score_map.clone()
        max_pool = nn.MaxPool2d(kernel_size=nms_radius,
                                stride=1, padding=nms_radius//2)

        is_max = (score_map == max_pool(score_map))
        nms_scores = nms_scores * is_max.float()

        # 展平响应图并获取前k个关键点
        nms_scores = nms_scores.view(batch_size, -1)
        topk_scores, topk_indices = torch.topk(
            nms_scores, k=min(k, nms_scores.shape[1]), dim=1)

        # 转换索引为坐标
        h, w = score_map.shape[2], score_map.shape[3]
        topk_y = (topk_indices // w).float()
        topk_x = (topk_indices % w).float()

        # 组合坐标
        keypoints = torch.stack([topk_x, topk_y], dim=2)

        return keypoints, topk_scores

    def forward(self, x, top_k=None):
        """
        模型前向传播
        Args:
            x: 输入图像，形状为[B, C, H, W]
            top_k: 每张图像提取的关键点数量
        Returns:
            keypoints: 关键点坐标，形状为[B, k, 2]
            scores: 关键点分数，形状为[B, k]
            descriptors: 关键点描述子，形状为[B, k, D]
        """
        if top_k is None:
            top_k = self.num_keypoints

        # 提取特征
        features = self.extractor(x)

        # 检测关键点
        score_map = self.detector(features)
        keypoints, scores = self.detect_keypoints(score_map, k=top_k)

        # 提取描述子
        desc_map = self.descriptor(features)

        # 为每个关键点提取描述子
        batch_size = x.shape[0]
        descriptors = []

        for i in range(batch_size):
            # 对于每个批次的图像
            kpts = keypoints[i]  # [k, 2]

            # 归一化坐标到[-1, 1]范围，用于grid_sample
            h, w = desc_map.shape[2], desc_map.shape[3]
            normalized_kpts = torch.zeros_like(kpts)
            normalized_kpts[:, 0] = 2 * kpts[:, 0] / (w - 1) - 1  # x坐标
            normalized_kpts[:, 1] = 2 * kpts[:, 1] / (h - 1) - 1  # y坐标

            # 重塑为grid_sample期望的格式 [1, k, 1, 2]
            grid = normalized_kpts.view(1, -1, 1, 2)

            # 采样描述子
            desc = torch.nn.functional.grid_sample(
                desc_map[i:i+1], grid, mode='bilinear', align_corners=True)

            # 重塑为 [k, D]
            desc = desc.permute(0, 2, 3, 1).reshape(-1, desc_map.shape[1])

            # 归一化描述子
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)

            descriptors.append(desc)

        descriptors = torch.stack(descriptors)

        return {
            'keypoints': keypoints,        # [B, k, 2]
            'scores': scores,              # [B, k]
            'descriptors': descriptors     # [B, k, D]
        }

# 多尺度特征融合模块 - 用于增强特征表示


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块，用于融合不同尺度的特征映射"""

    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleFeatureFusion, self).__init__()

        self.convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1))

        self.fusion_conv = nn.Conv2d(
            out_channels * len(in_channels_list), out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, features_list):
        """
        前向传播
        Args:
            features_list: 不同尺度特征的列表
        Returns:
            fused_features: 融合后的特征
        """
        # 调整每个特征图的通道数
        aligned_features = []
        target_size = features_list[0].shape[2:]

        for i, features in enumerate(features_list):
            # 调整通道数
            x = self.convs[i](features)

            # 调整空间尺寸
            if x.shape[2:] != target_size:
                x = nn.functional.interpolate(
                    x, size=target_size, mode='bilinear', align_corners=True)

            aligned_features.append(x)

        # 拼接所有特征
        fused = torch.cat(aligned_features, dim=1)

        # 融合通道
        fused = self.fusion_conv(fused)
        fused = self.norm(fused)
        fused = self.act(fused)

        return fused
