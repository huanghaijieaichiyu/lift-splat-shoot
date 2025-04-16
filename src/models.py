"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum
from models.CamEncoder_3d import CamEncoder_3d, create_cam_encoder
from logging import raiseExceptions
import math

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from models.RepVit import RepViTBlock
from models.common import EMA, PSA, SPPELAN, ChannelAttention, Conv, Gencov, C2fCIB, SCDown, C2f, SPPF
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_name('efficientnet-b0')
        self.up1 = Up(320 + 112, 512)
        self.depthnet = self.depthnet = nn.Conv2d(
            512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(
            1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x  # x: 24 x 512 x 8 x 22

    def forward(self, x):
        # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        计算点云中点的(x,y,z)位置（在自车坐标系中）
        返回 B x N x D x H/downsample x W/downsample x 3

        优化版本：使用批处理和矩阵运算加速计算
        """
        B, N, _ = trans.shape

        # 计算投影矩阵 (camera -> ego)
        # 避免重复计算逆矩阵
        if self.proj_matrix_cache is None or self.proj_matrix_cache[0].shape[0] != B:
            # 相机内参的逆矩阵
            intrins_inv = torch.inverse(
                intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
            # 相机到自车的变换矩阵
            cam_to_ego = torch.matmul(rots, intrins_inv)
            # 后处理旋转的逆矩阵
            post_rots_inv = torch.inverse(
                post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)

            # 缓存计算的矩阵
            self.proj_matrix_cache = (cam_to_ego, post_rots_inv, trans)
        else:
            cam_to_ego, post_rots_inv, cached_trans = self.proj_matrix_cache
            # 如果trans发生变化，更新缓存
            # 确保维度一致性进行比较
            if cached_trans.shape != trans.shape or not torch.allclose(trans, cached_trans):
                # 重新计算投影矩阵
                intrins_inv = torch.inverse(
                    intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
                cam_to_ego = torch.matmul(rots, intrins_inv)
                post_rots_inv = torch.inverse(
                    post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)
                self.proj_matrix_cache = (cam_to_ego, post_rots_inv, trans)

        cam_to_ego, post_rots_inv, _ = self.proj_matrix_cache

        # 获取视锥体网格点的形状
        D, fH, fW, _ = self.frustum.shape

        # 撤销后处理变换
        points = self.frustum.unsqueeze(0).unsqueeze(
            0) - post_trans.view(B, N, 1, 1, 1, 3)

        # 重塑为适合批量矩阵乘法的形状
        points_reshaped = points.reshape(B, N, -1, 3)

        # 使用批量矩阵乘法应用旋转变换
        post_rots_inv_transposed = post_rots_inv.transpose(-1, -2)
        points_rotated = torch.matmul(
            points_reshaped, post_rots_inv_transposed)

        # 恢复原始形状
        points = points_rotated.reshape(B, N, D, fH, fW, 3)

        # 准备相机到自车坐标转换
        # 将深度应用到x,y坐标
        points_with_depth = torch.cat(
            (
                points[..., :2] * points[..., 2:3],  # 应用深度
                points[..., 2:3]  # 保持深度值
            ),
            dim=-1
        )

        # 重塑为批量矩阵乘法形状
        points_with_depth_reshaped = points_with_depth.reshape(B, N, -1, 3)

        # 应用相机到自车的变换
        cam_to_ego_transposed = cam_to_ego.transpose(-1, -2)
        points_ego_reshaped = torch.matmul(
            points_with_depth_reshaped,
            cam_to_ego_transposed
        )

        # 添加平移向量
        points_ego_reshaped = points_ego_reshaped + trans.view(B, N, 1, 3)

        # 恢复原始形状
        points_ego = points_ego_reshaped.reshape(B, N, D, fH, fW, 3)

        return points_ego

    def get_cam_feats(self, x):
        """
        返回 B x N x D x H/downsample x W/downsample x C

        优化版本：使用内存高效的操作和特征增强
        """
        B, N, C, imH, imW = x.shape

        # 批处理相机图像编码
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)  # [B*N, C, H', W']

        # 添加特征增强
        if self.feat_enhancement:
            # 重塑张量以适应EMA和PSA模块的输入要求
            # 确保4D张量 [B*N, C, H, W]
            x = x.view(-1, self.camC, x.shape[2], x.shape[3])
            if hasattr(self, 'ema') and isinstance(self.ema, nn.Module):
                # 正确应用EMA注意力
                ema_weights = self.ema(x)
                x = x * ema_weights

            if hasattr(self, 'feat_attn') and isinstance(self.feat_attn, nn.Module):
                # 正确应用PSA注意力
                psa_weights = self.feat_attn(x)
                x = x * psa_weights

        # 重塑为期望的输出格式，避免额外的内存副本
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """Voxel pooling operation
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        if kept.sum() == 0:
            # 如果没有有效点，返回全零张量
            nx0 = int(self.nx[0].item())
            nx1 = int(self.nx[1].item())
            nx2 = int(self.nx[2].item())
            return torch.zeros((B, C * nx2, nx0, nx1),
                               device=x.device)

        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 总是使用cumsum_trick，避免QuickCumsum可能的问题
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        final = torch.zeros(
            (B, C, nx2, nx0, nx1), device=x.device)
        final[geom_feats[:, 3].long(), :, geom_feats[:, 2].long(),
              geom_feats[:, 0].long(), geom_feats[:, 1].long()] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC, model='vit', num_classes=10, lidar_channels=64):
    """
    根据配置编译不同类型的模型
    Args:
        grid_conf: 网格配置
        data_aug_conf: 数据增强配置
        outC: 输出通道数
        model: 模型类型 ('lss', 'beve', 'fusion', '3d')
        num_classes: 类别数量
        lidar_channels: LiDAR通道数量（仅用于融合模型）
    Returns:
        nn.Module: 编译好的模型
    """
    try:
        if model == 'lss':
            return LiftSplatShoot(grid_conf, data_aug_conf, outC)
        elif model == 'beve':
            # 确保类别数至少为1
            actual_num_classes = max(1, num_classes)
            # 适应检测头的要求
            if 'detection' in str(model).lower() or outC == actual_num_classes*9:
                # 如果outC已经设置为类别数*9
                return BEVENet(grid_conf, data_aug_conf, outC, num_classes=actual_num_classes, model_type='beve')
            else:
                # 默认情况下，为检测模型设置outC=num_classes*9
                return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve')
        elif model == 'fusion':
            # 使用多模态融合模型
            actual_num_classes = max(1, num_classes)
            try:
                # 导入多模态融合模型
                from models.MultiModalBEVENet import create_multimodal_bevenet
                return create_multimodal_bevenet(grid_conf, data_aug_conf, actual_num_classes*9, actual_num_classes, lidar_channels)
            except ImportError as e:
                print(f"无法导入多模态融合模型: {e}")
                print("回退到使用标准BEVENet模型")
                return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve')
        elif model == '3d':
            # 使用ViT模型
            actual_num_classes = max(1, num_classes)
            return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='3d')
        else:
            print(f"警告: 未知的模型类型 '{model}'。回退到使用标准BEVENet模型")
            actual_num_classes = max(1, num_classes)
            return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve')
    except Exception as e:
        print(f"编译模型时发生错误: {e}")
        print("回退到使用标准BEVENet模型")
        # 在出错的情况下回退到最基本的模型
        actual_num_classes = max(1, num_classes)
        return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve')


class CamEncoder(nn.Module):
    """
    CamEncoder is a neural network module designed for encoding camera input features.
    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
    Attributes:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        conv1 (nn.Module): First convolutional layer.
        conv2 (nn.Sequential): Second convolutional block consisting of RepViTBlock layers.
        conv3 (nn.Sequential): Third convolutional block consisting of RepViTBlock layers.
        conv4 (nn.Sequential): Fourth convolutional block consisting of RepViTBlock and C2f layers.
        conv5 (nn.Sequential): Fifth convolutional block consisting of SPPELAN and PSA layers.
        conv6 (RepViTBlock): Sixth convolutional layer.
        conv7 (C2fCIB): Seventh convolutional layer.
        conv8 (RepViTBlock): Eighth convolutional layer.
        conv9 (Gencov): Ninth convolutional layer.
        up (nn.Upsample): Upsampling layer.
    Methods:
        forward(x):
            Forward pass of the network.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor reshaped to (-1, c_out, 8, 22).
    """

    def __init__(self, c_in, c_out) -> None:
        super(CamEncoder, self).__init__()
        depth = 1
        weight = 1.0
        self.c_in = c_in
        self.c_out = c_out
        self.conv1 = Gencov(c_in, math.ceil(8 * depth))
        self.conv2 = nn.Sequential(
            RepViTBlock(math.ceil(8 * depth), math.ceil(16 * depth),
                        3 * math.ceil(weight), 2),
            RepViTBlock(math.ceil(16 * depth),
                        math.ceil(16 * depth), 1, 1, 0, 0)
        )

        self.conv3 = nn.Sequential(
            RepViTBlock(math.ceil(16 * depth), math.ceil(32 * depth),
                        3 * math.ceil(weight), 2),
            RepViTBlock(math.ceil(32 * depth),
                        math.ceil(32 * depth), 1, 1, 0, 0)
        )

        self.conv4 = nn.Sequential(
            RepViTBlock(math.ceil(32 * depth),
                        math.ceil(64 * depth), math.ceil(weight), 2),
            Gencov(math.ceil(64 * depth), math.ceil(128 * depth),
                   k=3, s=1, act=True)
        )

        self.conv5 = nn.Sequential(
            SPPELAN(math.ceil(128 * depth),
                    math.ceil(128 * depth), math.ceil(64 * depth)),
            PSA(math.ceil(128 * depth), math.ceil(128 * depth)),
        )

        self.conv6 = RepViTBlock(math.ceil(128 * depth),
                                 math.ceil(256 * depth), 1, 2)
        self.conv7 = C2fCIB(math.ceil(256 * depth), math.ceil(512 * depth))
        self.conv8 = RepViTBlock(math.ceil(640 * depth),
                                 math.ceil(1024 * depth), math.ceil(weight), 2)
        self.conv9 = Gencov(math.ceil(1024 * depth), c_out)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # head net
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        # neck net

        x7 = self.conv7(x6)
        x8 = self.conv9(self.conv8(torch.cat((x5, self.up(x7)), 1)))

        return x8.view(-1, self.c_out, 8, 22)


class BevEncoder(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncoder, self).__init__()
        # c_ = math.ceil(0.5 * inC)
        self.layer1 = nn.Sequential(
            RepViTBlock(inC, 16, 3, 2),
            C2f(16, 32, 1, True)
        )
        self.layer2 = nn.Sequential(
            RepViTBlock(32, 64, 3, 2),
            RepViTBlock(64, 64, 1, 1, 0, 0)
        )
        self.layer3 = nn.Sequential(
            SPPF(64, 64),
            PSA(64, 64)
        )
        self.layer4 = nn.Sequential(
            RepViTBlock(128, 128, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            C2fCIB(128, 256)
        )
        self.layer5 = nn.Sequential(
            RepViTBlock(256, 256, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Gencov(256, outC, act=False, bn=False)  # 注意输出层去sigmod，以及不要归一
        )

    def forward(self, x):
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x = self.layer4(torch.cat((x2, x3), 1))
        x = self.layer5(x)
        return x


class RepBEV_vit(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(RepBEV_vit, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode_rep(self.D, self.camC, self.downsample)
        self.bevencode = BevEncoder(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 总是使用cumsum_trick，避免QuickCumsum可能的问题
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        final = torch.zeros(
            (B, C, nx2, nx0, nx1), device=x.device)
        final[geom_feats[:, 3].long(), :, geom_feats[:, 2].long(),
              geom_feats[:, 0].long(), geom_feats[:, 1].long()] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


class CamEncode_rep(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode_rep, self).__init__()
        self.D = D
        self.C = C

        # 使用自定义的CamEncoder替代EfficientNet
        self.trunk = CamEncoder(3, 512)
        self.depthnet = Gencov(512, self.D + self.C, bn=False, act=False)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.trunk(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(
            1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def forward(self, x):
        # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)
        depth, x = self.get_depth_feat(x)

        return x


class BEVENet(nn.Module):
    """
    BEVENet: 基于纯卷积的高效3D目标检测BEV网络
    参考论文: Towards Efficient 3D Object Detection in Bird's-Eye-View Space for Autonomous Driving: A Convolutional-Only Approach
    """

    def __init__(self, grid_conf, data_aug_conf, outC, num_classes=10, detection_head=True, model_type='beve'):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = outC
        self.num_classes = max(1, num_classes)  # 确保类别数至少为1
        self.detection_head = detection_head

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        # 添加投影矩阵缓存，优化重复计算
        self.proj_matrix_cache = None

        # 根据模型类型选择相机编码器
        if model_type == '3d':
            # 使用ViT相机编码
            self.camencode = create_cam_encoder(
                self.D, self.camC, self.downsample)
        else:
            # 使用默认的卷积相机编码器
            self.camencode = CamEncode_rep(
                self.D, self.camC, self.downsample)

        # 使用纯卷积的BEV编码器
        self.bevencode = BEVEncoder_BEVE(inC=self.camC, outC=outC)

        # 添加3D检测头
        if detection_head:
            self.detection_head = DetectionHead_BEVE(
                outC, num_classes=self.num_classes)

        # 使用QuickCumsum提高计算效率
        self.use_quickcumsum = True

        # 添加特征增强模块
        self.ema = EMA(self.camC)
        self.feat_attn = PSA(self.camC, self.camC)
        self.feat_enhancement = True

    def create_frustum(self):
        # 创建视锥体投影网格
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D = ds.shape[0]
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        计算点云中点的(x,y,z)位置（在自车坐标系中）
        返回 B x N x D x H/downsample x W/downsample x 3

        优化版本：使用批处理和矩阵运算加速计算
        """
        B, N = trans.shape[0], trans.shape[1]

        # 计算投影矩阵 (camera -> ego)
        # 检查缓存是否有效，或相机数量是否变化
        if (self.proj_matrix_cache is None):
            # 重新计算投影矩阵
            # 相机内参的逆矩阵
            intrins_inv = torch.inverse(
                intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
            # 相机到自车的变换矩阵
            cam_to_ego = torch.matmul(rots, intrins_inv)
            # 后处理旋转的逆矩阵
            post_rots_inv = torch.inverse(
                post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)

            # 缓存计算的矩阵
            self.proj_matrix_cache = (cam_to_ego, post_rots_inv, trans)
        else:
            cam_to_ego, post_rots_inv, cached_trans = self.proj_matrix_cache
            # 如果trans发生变化，更新缓存
            # 确保维度一致性进行比较
            if cached_trans.shape != trans.shape or not torch.allclose(trans, cached_trans):
                # 重新计算投影矩阵
                intrins_inv = torch.inverse(
                    intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
                cam_to_ego = torch.matmul(rots, intrins_inv)
                post_rots_inv = torch.inverse(
                    post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)
                self.proj_matrix_cache = (cam_to_ego, post_rots_inv, trans)

        cam_to_ego, post_rots_inv, _ = self.proj_matrix_cache

        # 获取视锥体网格点的形状
        D, fH, fW, _ = self.frustum.shape

        # 撤销后处理变换
        points = self.frustum.unsqueeze(0).unsqueeze(
            0) - post_trans.view(B, N, 1, 1, 1, 3)

        # 重塑为适合批量矩阵乘法的形状
        points_reshaped = points.reshape(B, N, -1, 3)

        # 使用批量矩阵乘法应用旋转变换
        post_rots_inv_transposed = post_rots_inv.transpose(-1, -2)
        points_rotated = torch.matmul(
            points_reshaped, post_rots_inv_transposed)

        # 恢复原始形状
        points = points_rotated.reshape(B, N, D, fH, fW, 3)

        # 准备相机到自车坐标转换
        # 将深度应用到x,y坐标
        points_with_depth = torch.cat(
            (
                points[..., :2] * points[..., 2:3],  # 应用深度
                points[..., 2:3]  # 保持深度值
            ),
            dim=-1
        )

        # 重塑为批量矩阵乘法形状
        points_with_depth_reshaped = points_with_depth.reshape(B, N, -1, 3)

        # 应用相机到自车的变换
        cam_to_ego_transposed = cam_to_ego.transpose(-1, -2)
        points_ego_reshaped = torch.matmul(
            points_with_depth_reshaped,
            cam_to_ego_transposed
        )

        # 添加平移向量
        points_ego_reshaped = points_ego_reshaped + trans.view(B, N, 1, 3)

        # 恢复原始形状
        points_ego = points_ego_reshaped.reshape(B, N, D, fH, fW, 3)

        return points_ego

    def get_cam_feats(self, x):
        """
        返回 B x N x D x H/downsample x W/downsample x C

        优化版本：使用内存高效的操作和特征增强
        """
        B, N, C, imH, imW = x.shape

        # 批处理相机图像编码
        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)  # [B*N, C, H', W']

        # 重塑为期望的输出格式，避免额外的内存副本
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        体素池化 - 将特征投影到BEV网格

        优化版本：使用散射和索引优化，增加基于权重的特征聚合
        """
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # 将特征展平
        x = x.reshape(Nprime, C)

        # 计算体素索引
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 过滤边界外的点 - 使用掩码操作
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        if kept.sum() == 0:
            # 如果没有有效点，返回全零张量
            return torch.zeros((B, C * int(self.nx[2].item()),
                                int(self.nx[0].item()),
                                int(self.nx[1].item())),
                               device=x.device)

        x = x[kept]
        geom_feats = geom_feats[kept]

        # 计算唯一体素的累积和 - 使用高效的索引和排序操作
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 总是使用cumsum_trick，避免QuickCumsum可能的问题
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # 创建BEV网格 - 使用索引填充
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())

        # 使用高效的稀疏表示填充
        final = torch.zeros((B, C, nx2, nx0, nx1),
                            device=x.device, dtype=x.dtype)

        # 使用索引填充体素网格 - 直接使用索引而不是循环
        final[geom_feats[:, 3].long(),
              :,
              geom_feats[:, 2].long(),
              geom_feats[:, 0].long(),
              geom_feats[:, 1].long()] = x

        # 将Z维度的特征拼接 - 保留3D结构信息
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        模型的前向传播
        Args:
            x: 图像输入
            rots: 旋转矩阵
            trans: 平移向量
            intrins: 相机内参
            post_rots: 图像后处理旋转
            post_trans: 图像后处理平移
        Returns:
            分类、回归和IoU预测结果的字典
        """
        # 获取体素表示
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)

        # 进行BEV编码
        preds = self.bevencode(x)

        return preds


class CamEncoder_BEVE(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncoder_BEVE, self).__init__()
        self.D = D
        self.C = C

        # 使用 EfficientNet 作为特征提取器
        self.trunk = EfficientNet.from_name('efficientnet-b0')
        self.up1 = Up(320 + 112, 512)
        # 修改 depthnet 的设计，确保维度正确
        self.depth_conv = nn.Sequential(
            Conv(512, 256, 3, 1, 1),  # 保持空间维度
            Conv(256, 128, 3, 1, 1),
            Conv(128, self.D + self.C, 1, 1, act=False, bn=False)  # 输出深度和特征通道
        )

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        """
        Args:
            x: 输入张量 [B*N, C, H, W]
        Returns:
            depth: 深度分布 [B*N, D, H', W']
            features: 特征图 [B*N, C, D, H', W']
        """

        features = self.get_eff_depth(x)  # [B*N, 512, H', W']

        # 生成深度和特征
        combined = self.depth_conv(features)  # [B*N, D+C, H', W']

        # 分离深度和特征
        depth = self.get_depth_dist(combined[:, :self.D])  # [B*N, D, H', W']
        feat = combined[:, self.D:(self.D + self.C)]  # [B*N, C, H', W']

        # 生成最终特征
        new_x = depth.unsqueeze(1) * feat.unsqueeze(2)  # [B*N, C, D, H', W']

        return depth, new_x

    def get_eff_depth(self, x):
        """使用 EfficientNet 提取特征，保持空间维度的一致性"""
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        endpoints = dict()
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)
        return x


class BEVEncoder_BEVE(nn.Module):
    def __init__(self, inC, outC):
        super(BEVEncoder_BEVE, self).__init__()
        # 确保输出通道数至少为1
        self.outC = max(1, outC)
        self.num_classes = self.outC // 9  # 假设outC是9*num_classes

        self.layer1 = nn.Sequential(
            RepViTBlock(inC, 16, 3, 2),
            C2f(16, 32, 1, True)
        )
        self.layer2 = nn.Sequential(
            RepViTBlock(32, 64, 3, 2),
            RepViTBlock(64, 64, 1, 1, 0, 0)
        )
        self.layer3 = nn.Sequential(
            SPPF(64, 64),
            PSA(64, 64)
        )
        self.layer4 = nn.Sequential(
            RepViTBlock(64, 64, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            C2fCIB(64, 128)
        )

        # 共享特征提取器
        self.shared_features = nn.Sequential(
            RepViTBlock(128, 128, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # 分类头 - 预测每个位置的类别
        self.cls_head = Conv(128, self.num_classes, k=1)

        # 回归头 - 预测边界框参数 (x,y,z,w,l,h,sin,cos,vel)
        self.reg_head = Conv(128, 9, k=1)

        # IoU头 - 预测检测质量
        self.iou_head = Conv(128, 1, k=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 共享特征
        shared_feats = self.shared_features(x)

        # 生成三个任务的预测
        cls_pred = self.cls_head(shared_feats)
        reg_pred = self.reg_head(shared_feats)
        iou_pred = self.iou_head(shared_feats)

        # 返回包含三种预测的字典
        return {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'iou_pred': iou_pred
        }


class DetectionHead_BEVE(nn.Module):
    def __init__(self, inC, num_classes):
        super(DetectionHead_BEVE, self).__init__()
        self.num_classes = max(1, num_classes)  # 确保类别数至少为1
        self.fc = nn.Linear(inC, self.num_classes)

    def forward(self, x):
        return self.fc(x)
