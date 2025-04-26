from .tools import gen_dx_bx, cumsum_trick
import math

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet152, ResNet152_Weights
from torchvision.ops import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

from utils.RepVit import RepViTBlock
from utils.common import EMA, PSA, SPPELAN, Conv, Gencov, C2fCIB, C2f, SPPF
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
            # drop_connect_rate = self.trunk._global_params.drop_connect_rate # 原代码
            # 修复：检查 _global_params 是否存在以及 drop_connect_rate 属性
            drop_connect_rate = 0.0
            if hasattr(self.trunk, '_global_params') and self.trunk._global_params is not None and hasattr(self.trunk._global_params, 'drop_connect_rate'):
                drop_connect_rate = self.trunk._global_params.drop_connect_rate

            if drop_connect_rate and drop_connect_rate > 0:  # 添加检查 > 0
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

        trunk = resnet50(pretrained=False, zero_init_residual=True)
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
        self.frustum = self.create_frustum(
            fH=self.data_aug_conf['final_dim'][0], fW=self.data_aug_conf['final_dim'][1])
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self, fH, fW):
        """创建视锥体，使用动态确定的特征图尺寸"""
        # 获取图像原始尺寸 (可能不需要了，但保留以防万一)
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # fH, fW = ogfH // self.downsample, ogfW // self.downsample # No longer use self.downsample

        # 创建深度范围 (使用 self.D)
        d_coords = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        # D = d_coords.shape[0] # D is now self.D

        # 创建像素坐标 (使用传入的 fH, fW)
        # Note: linspace should use the original dimensions ogfW/ogfH for coordinate generation
        # Then these coordinates are sampled at the feature map resolution fW/fH
        x_coords = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(self.D, fH, fW)
        y_coords = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(self.D, fH, fW)

        # 添加齐次坐标
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        # Return the tensor, not nn.Parameter, as it's dynamically created
        return frustum

    def get_geometry(self, frustum, rots, trans, intrins, post_rots, post_trans):
        """
        计算几何变换，从相机坐标系转换到BEV坐标系
        参考BEVDepth实现

        Args:
            frustum: [D, H, W, 4] - 动态创建的视锥体
            rots, trans, intrins, post_rots, post_trans: 相机参数
        返回:
            [B, N, D, H, W, 3] - 几何坐标
        """
        B, N, _, _ = intrins.shape
        D, H, W, _ = frustum.shape  # Get dimensions from passed frustum

        # 创建传感器到自车坐标系的变换矩阵
        sensor2ego_mats = torch.eye(4, device=rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        sensor2ego_mats[:, :, :3, :3] = rots
        sensor2ego_mats[:, :, :3, 3] = trans

        # 创建相机内参矩阵
        intrin_mats = torch.eye(4, device=intrins.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        intrin_mats[:, :, :3, :3] = intrins

        # 创建图像数据增强变换矩阵
        ida_mats = torch.eye(4, device=post_rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        ida_mats[:, :, :3, :3] = post_rots
        ida_mats[:, :, :3, 3] = post_trans

        # 使用frustum进行投影 (Use the passed frustum)
        points = frustum.to(rots.device)
        points = points.view(1, 1, D, H, W, 4, 1)

        # 应用图像数据增强的逆变换
        ida_mats = ida_mats.view(B, N, 1, 1, 1, 4, 4)
        points = ida_mats.inverse().matmul(points)

        # 将点从像素坐标转换为相机坐标
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)

        # 应用相机到自车的变换
        combine = sensor2ego_mats.matmul(torch.inverse(intrin_mats))
        points = combine.view(B, N, 1, 1, 1, 4, 4).matmul(points)

        # 提取3D坐标
        return points[..., :3, 0]

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
        """
        体素池化 - 将特征投影到BEV网格，参考BEVDepth实现

        Args:
            geom_feats: [B, N, D, H, W, 3] - 几何坐标
            x: [B, N, C, D, H, W] - 已加权的特征
        返回:
            [B, C, nx0, nx1] - BEV特征
        """
        # B, N, D, H, W, C = x.shape  # <<<< WRONG ORDER
        B, N, C, D, H, W = x.shape  # <<<< CORRECT ORDER

        # 计算网格索引，确保使用正确的tensor
        voxel_size = torch.tensor([
            (self.grid_conf['xbound'][1] -
             self.grid_conf['xbound'][0]) / self.nx[0].item(),
            (self.grid_conf['ybound'][1] -
             self.grid_conf['ybound'][0]) / self.nx[1].item(),
            (self.grid_conf['zbound'][1] -
             self.grid_conf['zbound'][0]) / self.nx[2].item()
        ], device=geom_feats.device)

        voxel_coord = torch.tensor([
            self.grid_conf['xbound'][0],
            self.grid_conf['ybound'][0],
            self.grid_conf['zbound'][0]
        ], device=geom_feats.device)

        # 计算体素索引
        geom_indices = ((geom_feats - (voxel_coord - voxel_size / 2.0).view(1, 1, 1, 1, 1, 3)) /
                        voxel_size.view(1, 1, 1, 1, 1, 3)).long()

        # 确保索引在有效范围内
        # Use the correct spatial dimensions (H, W) and depth dimension (D) from unpacking
        valid_mask = ((geom_indices[..., 0] >= 0) & (geom_indices[..., 0] < self.nx[0]) &
                      (geom_indices[..., 1] >= 0) & (geom_indices[..., 1] < self.nx[1]) &
                      (geom_indices[..., 2] >= 0) & (geom_indices[..., 2] < self.nx[2]))
        # Expected shape: [B, N, D, H, W] = [8, 5, 41, 8, 22]

        # 展平所有特征和索引
        # [B, N, C, D, H, W] -> [B, N, D, H, W, C]
        x = x.permute(0, 1, 3, 4, 5, 2)
        geom_indices = geom_indices.view(-1, 3)
        # Expected shape: [B*N*D*H*W] = [288640]
        valid_mask = valid_mask.view(-1)

        # 创建批次索引
        batch_idx = torch.arange(
            B, device=geom_feats.device).view(B, 1, 1, 1, 1, 1)
        # Use the correct dimensions from unpacking for expand
        # Expected shape: [8, 5, 41, 8, 22, 1]
        batch_idx = batch_idx.expand(B, N, D, H, W, 1)
        # Indexed tensor shape: [288640], mask shape: [288640]
        batch_idx = batch_idx.reshape(-1)[valid_mask]

        # 提取有效的索引和特征
        geom_indices = geom_indices[valid_mask]

        # 将特征展平为 [ValidPoints, C]
        x = x.reshape(-1, C)
        x = x[valid_mask]

        # 创建4D索引 (batch_idx, z, x, y)
        indices = torch.cat([
            batch_idx.unsqueeze(-1),
            geom_indices[:, 2].unsqueeze(-1),  # z
            geom_indices[:, 0].unsqueeze(-1),  # x
            geom_indices[:, 1].unsqueeze(-1),  # y
        ], dim=-1)

        # 初始化输出特征体素网格
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        out = torch.zeros((B, nx2, nx0, nx1, C),
                          device=x.device, dtype=x.dtype)

        # 使用scatter_add_填充体素
        out.index_put_(
            (indices[:, 0].long(), indices[:, 1].long(),
             indices[:, 2].long(), indices[:, 3].long()),
            x,
            accumulate=True
        )

        # 沿Z轴应用最大池化
        out = out.permute(0, 4, 1, 2, 3)  # [B, C, Z, X, Y]
        out = torch.max(out, dim=2)[0]  # Max pooling along Z

        return out

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        获取BEV体素特征
        Args:
            x: [B, N, 3, H, W] - 图像输入
            rots, trans, intrins, post_rots, post_trans: 相机参数
        返回:
            [B, C, X, Y] - BEV特征图
        """
        B, N, _, imH, imW = x.shape

        # 1. 获取相机特征，确定实际 H, W
        # get_cam_feats 内部会调用 camencode 处理深度和特征
        cam_feats_output = self.get_cam_feats(x)
        # 获取 cam_feats_output 的形状以确定 H_actual, W_actual
        # Shape: [B, N, C, D, H_actual, W_actual]
        _B, _N, _C, _D, H_actual, W_actual = cam_feats_output.shape

        # 2. 动态创建 Frustum (使用实际的 H, W)
        frustum = self.create_frustum(H_actual, W_actual)  # Pass actual H, W

        # 3. 计算几何变换 (Pass the dynamically created frustum)
        geom = self.get_geometry(
            frustum, rots, trans, intrins, post_rots, post_trans)
        # geom 形状: [B, N, D, H_actual, W_actual, 3]

        bev_feat = self.voxel_pooling(geom, x)

        return bev_feat

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC, model='vit', num_classes=10, lidar_channels=64, backbone_type='resnet18'):
    """
    根据配置编译不同类型的模型
    Args:
        grid_conf: 网格配置
        data_aug_conf: 数据增强配置
        outC: 输出通道数
        model: 模型类型 ('lss', 'beve', 'fusion', '3d', 'xfeat')
        num_classes: 类别数量
        lidar_channels: LiDAR通道数量（仅用于融合模型）
        backbone_type: BEVENet使用的骨干网络 ('resnet18', 'resnet50', 'resnet152')
    Returns:
        nn.Module: 编译好的模型
    """
    try:
        if model == 'lss':
            return LiftSplatShoot(grid_conf, data_aug_conf, outC)
        elif model == 'beve':
            # --- 修改：直接使用传入的 num_classes 和 outC ---
            print(
                f"Compiling BEVENet with num_classes={num_classes}, received outC={outC}")
            actual_num_classes = max(1, num_classes)  # 确保至少为1
            # 直接将接收到的 num_classes 和 outC 传递给 BEVENet
            # BEVENet 内部会处理 num_classes，outC 主要用于兼容性或未来扩展
            # Pass backbone_type to BEVENet
            return BEVENet(grid_conf, data_aug_conf, outC, num_classes=actual_num_classes, model_type='beve', backbone_type=backbone_type)
            # --- 修改结束 ---
        elif model == 'fusion':
            # 使用多模态融合模型
            actual_num_classes = max(1, num_classes)
            # Pass backbone_type to BEVENet
            return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve', backbone_type=backbone_type)
        else:
            print(f"警告: 未知的模型类型 '{model}'。回退到使用标准BEVENet模型")
            actual_num_classes = max(1, num_classes)
            # Pass backbone_type to BEVENet
            return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve', backbone_type=backbone_type)
    except Exception as e:
        print(f"编译模型时发生错误: {e}")
        print("回退到使用标准BEVENet模型")
        # 在出错的情况下回退到最基本的模型
        actual_num_classes = max(1, num_classes)
        # Pass backbone_type to BEVENet
        return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve', backbone_type=backbone_type)


class CamEncoder(nn.Module):
    """
    CamEncoder 是一个用于编码相机输入特征的神经网络模块。
    Args:
        c_in (int): 输入通道数。
        c_out (int): 输出通道数。
    Attributes:
        c_in (int): 输入通道数。
        c_out (int): 输出通道数。
        conv1 (nn.Module): 第一个卷积层。
        conv2 (nn.Sequential): 第二个卷积块，由 RepViTBlock 层组成。
        conv3 (nn.Sequential): 第三个卷积块，由 RepViTBlock 层组成。
        conv4 (nn.Sequential): 第四个卷积块，由 RepViTBlock 和 C2f 层组成。
        conv5 (nn.Sequential): 第五个卷积块，由 SPPELAN 和 PSA 层组成。
        conv6 (RepViTBlock): 第六个卷积层。
        conv7 (C2fCIB): 第七个卷积层。
        conv8 (RepViTBlock): 第八个卷积层。
        conv9 (Gencov): 第九个卷积层。
        up (nn.Upsample): 上采样层。
    Methods:
        forward(x):
            网络的前向传播。
            Args:
                x (torch.Tensor): 输入张量。
            Returns:
                torch.Tensor: 输出张量，重塑为 (-1, c_out, 8, 22)。

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
        self.camencode = CamEncode_rep(self.D, self.camC)
        self.bevencode = BevEncoder(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self, fH=None, fW=None):
        """创建视锥体，使用动态确定的特征图尺寸 (或默认值)

        Args:
            fH (int | None): Actual feature map height. If None, calculated from config.
            fW (int | None): Actual feature map width. If None, calculated from config.
        """
        ogfH, ogfW = self.data_aug_conf['final_dim']

        # If fH, fW are not provided (e.g., compatibility with old calls), calculate defaults
        # Assume a default downsampling if needed for calculation, though ideally fH/fW are passed.
        if fH is None or fW is None:
            print(
                "Warning: create_frustum called without fH/fW. Using default calculation based on final_dim.")
            # Use final_dim directly if no downsampling context is available
            # This might be inaccurate if the caller expected downsampling
            fH = ogfH
            fW = ogfW
            # If a downsample factor *was* intended by the caller, this needs revisiting.
            # For BEVENet.get_voxels, fH/fW are passed correctly.

        # 创建深度范围 (使用 self.D)
        d_coords = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)

        # 创建像素坐标 (使用传入的 fH, fW)
        # Note: linspace should use the original dimensions ogfW/ogfH for coordinate generation
        # Then these coordinates are sampled at the feature map resolution fW/fH
        x_coords = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(self.D, fH, fW)
        y_coords = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(self.D, fH, fW)

        # 添加齐次坐标
        paddings = torch.ones_like(d_coords)

        # D x H x W x 4
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, frustum, rots, trans, intrins, post_rots, post_trans):
        """
        计算几何变换，从相机坐标系转换到BEV坐标系
        参考BEVDepth实现

        Args:
            frustum: [D, H, W, 4] - 动态创建的视锥体
            rots, trans, intrins, post_rots, post_trans: 相机参数
        返回:
            [B, N, D, H, W, 3] - 几何坐标
        """
        B, N, _, _ = intrins.shape
        D, H, W, _ = frustum.shape  # Get dimensions from passed frustum

        # 创建传感器到自车坐标系的变换矩阵
        sensor2ego_mats = torch.eye(4, device=rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        sensor2ego_mats[:, :, :3, :3] = rots
        sensor2ego_mats[:, :, :3, 3] = trans

        # 创建相机内参矩阵
        intrin_mats = torch.eye(4, device=intrins.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        intrin_mats[:, :, :3, :3] = intrins

        # 创建图像数据增强变换矩阵
        ida_mats = torch.eye(4, device=post_rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        ida_mats[:, :, :3, :3] = post_rots
        ida_mats[:, :, :3, 3] = post_trans

        # 使用frustum进行投影 (Use the passed frustum)
        points = frustum.to(rots.device)
        points = points.view(1, 1, D, H, W, 4, 1)

        # 应用图像数据增强的逆变换
        ida_mats = ida_mats.view(B, N, 1, 1, 1, 4, 4)
        points = ida_mats.inverse().matmul(points)

        # 将点从像素坐标转换为相机坐标
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:]), 5)

        # 应用相机到自车的变换
        combine = sensor2ego_mats.matmul(torch.inverse(intrin_mats))
        points = combine.view(B, N, 1, 1, 1, 4, 4).matmul(points)

        # 提取3D坐标
        return points[..., :3, 0]

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
        final = torch.max(final, dim=2)[0]

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        获取BEV体素特征
        Args:
            x: [B, N, 3, H, W] - 图像输入
            rots, trans, intrins, post_rots, post_trans: 相机参数
        返回:
            [B, C, X, Y] - BEV特征图
        """
        B, N, _, imH, imW = x.shape

        # 1. 获取相机特征，确定实际 H, W
        # get_cam_feats 内部会调用 camencode 处理深度和特征
        cam_feats_output = self.get_cam_feats(x)
        # 获取 cam_feats_output 的形状以确定 H_actual, W_actual
        # Shape: [B, N, C, D, H_actual, W_actual]
        _B, _N, _C, _D, H_actual, W_actual = cam_feats_output.shape

        # 2. 动态创建 Frustum (使用实际的 H, W)
        frustum = self.create_frustum(H_actual, W_actual)  # Pass actual H, W

        # 3. 计算几何变换 (Pass the dynamically created frustum)
        geom = self.get_geometry(
            frustum, rots, trans, intrins, post_rots, post_trans)
        # geom 形状: [B, N, D, H_actual, W_actual, 3]

        # 4. 准备相机矩阵字典 (移到这里，因为它被 DepthNet 使用)
        # (将原来的步骤4-9移到这里，因为它们依赖于 B, N, H_actual, W_actual)
        mats_dict = {
            'sensor2ego_mats': torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1),
            'intrin_mats': torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1),
            'ida_mats': torch.eye(4, device=post_rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1),
        }
        mats_dict['sensor2ego_mats'][:, :, :3, :3] = rots
        mats_dict['sensor2ego_mats'][:, :, :3, 3] = trans
        mats_dict['intrin_mats'][:, :, :3, :3] = intrins
        mats_dict['ida_mats'][:, :, :3, :3] = post_rots
        mats_dict['ida_mats'][:, :, :3, 3] = post_trans

        # 5. 提取2D特征用于深度预测 (需要从 cam_feats_output 转换)
        # cam_feats_output: [B, N, C, D, H_actual, W_actual]
        # 需要 [B*N, C, H_actual, W_actual]
        cam_feats_2d = cam_feats_output.permute(0, 1, 2, 4, 5, 3).reshape(
            B * N, self.camC, H_actual, W_actual, _D).mean(dim=-1)  # 取 D 维度平均

        # 6. 使用深度网络获取深度和上下文特征
        depth_feature = self._forward_depth_net(
            cam_feats_2d, mats_dict)
        # Expected Shape: [B*N, D+C, H_actual, W_actual]

        # 7. 分离深度概率和上下文特征
        depth = depth_feature[:, :self.D].softmax(dim=1)
        context = depth_feature[:, self.D:]
        # Expected Shape: [B*N, D, H_actual, W_actual]
        # Expected Shape: [B*N, C, H_actual, W_actual]

        # 8. 使用深度概率对上下文特征加权
        # context 需要 unsqueeze(2) -> [B*N, C, 1, H, W]
        # depth 需要 unsqueeze(1) -> [B*N, 1, D, H, W]
        weighted_x = depth.unsqueeze(1) * context.unsqueeze(2)
        # Expected Shape: [B*N, C, D, H_actual, W_actual]

        # 9. 将加权特征重塑为 [B, N, C, D, H_actual, W_actual]
        weighted_x = weighted_x.view(
            B, N, self.camC, self.D, H_actual, W_actual)

        # 10. 应用体素池化 (使用动态计算的 geom)
        # voxel_pooling 输入 geom: [B, N, D, H, W, 3], x: [B, N, C, D, H, W]
        bev_feat = self.voxel_pooling(geom, weighted_x)

        return bev_feat

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


class BEVEncoder_BEVE(nn.Module):
    # --- 修改：接收 num_classes 参数，并移除对 outC 的错误依赖 ---
    def __init__(self, inC, outC, num_classes):
        super(BEVEncoder_BEVE, self).__init__()
        self.num_classes = max(1, num_classes)  # 直接使用传递的 num_classes
    # --- 修改结束 ---

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

        # --- Refactored Detection Head ---
        head_inter_channels = 128  # Intermediate channel size for head layers

        # Classification Head
        self.cls_head = nn.Sequential(
            # 3x3 Conv + BN + ReLU (assuming Conv includes this)
            Conv(128, head_inter_channels, k=3, p=1),
            # Final 1x1 Conv: Output logits, no activation/BN
            Conv(head_inter_channels, self.num_classes, k=1)
        )

        # Regression Head
        self.reg_head = nn.Sequential(
            Conv(128, head_inter_channels, k=3, p=1),
            # Final 1x1 Conv: Output 9 regression values, no activation/BN
            Conv(head_inter_channels, 9, k=1)
        )

        # IoU Head
        self.iou_head = nn.Sequential(
            Conv(128, head_inter_channels, k=3, p=1),
            # Final 1x1 Conv: Output 1 IoU logit, no activation/BN
            Conv(head_inter_channels, 1, k=1)
        )
        # --- End Refactored Head ---

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


class CamEncode_rep(nn.Module):
    # This class might become obsolete or serve a different purpose if needed elsewhere
    # For now, just remove trunk initialization and forward logic related to it.
    def __init__(self, D_unused, C_unused):  # D, C might no longer be needed here
        super(CamEncode_rep, self).__init__()
        # Remove trunk initialization
        # self.trunk = CamEncoder(3, 512)
        print("Warning: CamEncode_rep is potentially obsolete after refactoring.")
        pass

    # Remove old depth methods
    # def get_depth_dist(...): ...
    # def get_depth_feat(...): ...

    def forward(self, x):
        # This forward is now likely unused in the BEVENet Method B path
        raise NotImplementedError(
            "CamEncode_rep.forward should not be called directly in BEVENet Method B path")


class DepthNet(nn.Module):
    """深度预测网络 (改进版)
    - 使用 FiLM 层进行相机参数条件化
    - 输出深度 Bin 分类 logits 和 Bin 内残差回归值
    """

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.depth_channels = depth_channels  # D
        self.context_channels = context_channels  # C

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),  # Bias=False if using BN
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # MLP for FiLM parameters (output gamma and beta for mid_channels)
        # Input dim 27 assumes flattened sensor2ego (16) + intrin (9 rounded up?) - Check this
        # Let's make it flexible or confirm the source of 27.
        # Assuming 27 is correct for now.
        cam_param_dim = 27
        self.film_mlp = nn.Sequential(
            nn.Linear(cam_param_dim, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels * 2)  # Output gamma and beta
        )

        # Depth Head - Outputs Bin Logits (D channels) and Bin Residuals (D channels)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.depth_channels * 2,
                      kernel_size=1, padding=0)  # Output 2*D channels
        )

        # Context Head - Outputs Context Features (C channels)
        self.context_conv = nn.Sequential(  # Renamed from context_conv2 for clarity
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, self.context_channels,
                      kernel_size=1, padding=0)  # Output C channels
        )

    def forward(self, x, mats_dict):
        """前向传播
        Args:
            x: 图像特征 [B*N, C_in, H, W]
            mats_dict: 包含相机内外参的字典
        Returns:
            Dict: {
                'bin_logits': 深度 bin 分类 logits [B*N, D, H, W],
                'bin_residuals': 深度 bin 残差 [B*N, D, H, W],
                'context': 上下文特征 [B*N, C_out, H, W]
            }
        """
        BN, C_in, H, W = x.shape

        # 1. Reduce input features
        x_reduced = self.reduce_conv(x)  # Shape: [BN, mid_channels, H, W]
        mid_channels = x_reduced.shape[1]

        # 2. Prepare camera parameters
        sensor2ego_mat = mats_dict['sensor2ego_mats']  # [B, N, 4, 4]
        intrin_mat = mats_dict['intrin_mats']       # [B, N, 4, 4]
        B, N = sensor2ego_mat.shape[:2]

        # Flatten and select parameters (ensure correct dimension, e.g., 27)
        # Taking first 16 from ego, first 11 from intrin? Needs verification.
        # Example: Flatten 4x4=16 ego, 3x3=9 intrin, pad/select to 27.
        # Let's assume the concatenation and slicing logic is correct for dim 27.
        cam_params_flat = torch.cat([
            sensor2ego_mat.view(B*N, -1)[:, :16],  # Example: take first 16
            intrin_mat.view(B*N, -1)[:, :9],     # Example: take first 9 (3x3)
            torch.zeros(B*N, 2, device=x.device)  # Example: padding to 27
        ], dim=1).contiguous()

        # Ensure correct dimension if changed
        if cam_params_flat.shape[1] != self.film_mlp[0].in_features:
            raise ValueError(
                f"Camera parameter dimension mismatch. Expected {self.film_mlp[0].in_features}, got {cam_params_flat.shape[1]}")

        # 3. Generate FiLM parameters
        # Shape: [BN, mid_channels * 2]
        film_params = self.film_mlp(cam_params_flat)
        # Split into gamma and beta
        # Each [BN, mid_channels]
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        # Reshape for broadcasting: [BN, mid_channels, 1, 1]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # 4. Apply FiLM conditioning
        # Shape: [BN, mid_channels, H, W]
        x_conditioned = gamma * x_reduced + beta

        # 5. Predict Depth (Bin Logits + Residuals) and Context
        depth_head_output = self.depth_conv(
            x_conditioned)  # Shape: [BN, 2*D, H, W]
        context = self.context_conv(x_conditioned)       # Shape: [BN, C, H, W]

        # 6. Split depth head output
        # Shape: [BN, D, H, W]
        bin_logits = depth_head_output[:, :self.depth_channels, :, :]
        # Shape: [BN, D, H, W]
        bin_residuals = depth_head_output[:, self.depth_channels:, :, :]

        return {
            'bin_logits': bin_logits,
            'bin_residuals': bin_residuals,
            'context': context
        }


class BEVENet(nn.Module):
    """
    BEVENet: 基于纯卷积的高效3D目标检测BEV网络 (已改进: 使用ResNet-50 + FPN)
    参考论文: Towards Efficient 3D Object Detection in Bird's-Eye-View Space for Autonomous Driving: A Convolutional-Only Approach
    """

    # --- 修改：添加 backbone_type 参数 ---
    def __init__(self, grid_conf, data_aug_conf, outC, num_classes=10, detection_head=True, model_type='beve', backbone_type='resnet18'):
        # --- 修改结束 ---
        """
        初始化 BEVENet (已改进: 可选 ResNet + FPN)

        Args:
            grid_conf : 网格配置
            data_aug_conf : 数据增强配置
            outC : 输出通道数 (传递给 BEVEncoder)
            num_classes : 类别数
            detection_head : 是否使用检测头 (当前 BEVEncoder 总是使用)
            model_type : 模型类型 (当前未使用)
            backbone_type (str): 使用的 ResNet 骨干类型 ('resnet18', 'resnet50', 'resnet152')
        """
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = outC  # Pass to BEVEncoder
        self.num_classes = max(1, num_classes)
        # Currently unused, BEVEncoder handles heads
        self.detection_head = detection_head
        self.backbone_type = backbone_type  # Store backbone type

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # D determined dynamically based on grid_conf['dbound']
        d_start, d_end, d_step = self.grid_conf['dbound']
        self.D = int((d_end - d_start) / d_step)
        # Create depth bin centers for residual calculation
        # Shape: [D]
        self.d_centers = nn.Parameter(torch.arange(
            d_start + d_step / 2, d_end, d_step), requires_grad=False)
        self.d_step = d_step  # Store step size

        # --- Backbone and FPN Initialization (Configurable ResNet) ---
        # 选择 ResNet 骨干网络
        # 推荐：
        # - 'resnet18': 速度快，内存占用小，精度相对较低。
        # - 'resnet50': 精度和速度/内存的良好平衡 (常用)。
        # - 'resnet152': 精度最高，但速度慢，内存占用大。
        if self.backbone_type == 'resnet18':
            weights = ResNet18_Weights.DEFAULT
            base_backbone = resnet18(weights=weights)
            # ResNet-18/34 输出通道: layer1=64, layer2=128, layer3=256, layer4=512
            in_channels_list = [64, 128, 256, 512]
        elif self.backbone_type == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            base_backbone = resnet50(weights=weights)
            # ResNet-50/101/152 输出通道: layer1=256, layer2=512, layer3=1024, layer4=2048
            in_channels_list = [256, 512, 1024, 2048]
        elif self.backbone_type == 'resnet152':
            weights = ResNet152_Weights.DEFAULT
            base_backbone = resnet152(weights=weights)
            # ResNet-50/101/152 输出通道: layer1=256, layer2=512, layer3=1024, layer4=2048
            in_channels_list = [256, 512, 1024, 2048]
        else:
            raise ValueError(
                f"不支持的 backbone_type: {self.backbone_type}. 请选择 'resnet18', 'resnet50', 或 'resnet152'.")

        print(f"使用骨干网络: {self.backbone_type}")

        # Define layers to extract features from (ResNet stages)
        # Using standard ResNet layer names
        return_nodes = {
            # 'relu': 'feat0', # Output before stage 1 (optional, /4 stride)
            'layer1': 'feat1',  # /4 stride output
            'layer2': 'feat2',  # /8 stride output
            'layer3': 'feat3',  # /16 stride output
            'layer4': 'feat4',  # /32 stride output
        }
        self.backbone = create_feature_extractor(
            base_backbone, return_nodes=return_nodes)

        fpn_out_channels = 32  # Common FPN output channel size

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=fpn_out_channels,
            # Use ExtraFPNBlock for P6 if needed, default adds maxpool for P5->P6
            # extra_blocks=LastLevelMaxPool() # Or other FPN block type if desired
        )
        # --- End Backbone/FPN Init ---

        # --- DepthNet and BEVEncoder Initialization ---
        # DepthNet input now comes from FPN (using highest resolution FPN output, typically P2)
        depthnet_in_channels = fpn_out_channels
        depthnet_mid_channels = 128  # Keep same intermediate size for now
        # Context channels from DepthNet (output of DepthNet, input to Voxel Pooling)
        self.camC = 64
        bev_input_channels = self.camC  # Channels from voxel pooling into BEV Encoder

        # Instantiate the improved DepthNet
        self.depth_net = DepthNet(
            in_channels=depthnet_in_channels,  # Input from FPN
            mid_channels=depthnet_mid_channels,
            context_channels=self.camC,
            depth_channels=self.D
        )

        # --- 修改：使用新的 BEVEncoderCenterPointHead ---
        # BEV Encoder + CenterPoint Head
        # outC 参数对于 CenterPoint head 不再直接使用，因为输出由内部任务决定
        print(
            f"初始化 BEVEncoderCenterPointHead (输入通道: {bev_input_channels}, 类别数: {self.num_classes})")
        self.bevencode = BEVEncoderCenterPointHead(
            inC=bev_input_channels,
            num_classes=self.num_classes
        )
        # --- 修改结束 ---

        # Feature enhancement modules (applied to DepthNet input)
        self.ema = EMA(depthnet_in_channels)  # Channels match FPN output
        # Channels match FPN output
        self.feat_attn = PSA(depthnet_in_channels, depthnet_in_channels)
        self.feat_enhancement = True  # Toggle enhancement

    def get_geometry_at_depth(self, pixel_coords, depth_values, rots, trans, intrins, post_rots, post_trans):
        """
        Projects pixel coordinates with associated depth values to ego frame.

        Args:
            pixel_coords (torch.Tensor): Pixel coordinates (x, y) [H, W, 2].
            depth_values (torch.Tensor): Refined depth for each pixel [B*N, H, W].
            rots, trans, intrins, post_rots, post_trans: Camera parameters.

        Returns:
            torch.Tensor: 3D points in ego frame [B, N, H, W, 3].
        """
        B, N, _, _ = intrins.shape
        BN, H, W = depth_values.shape
        # Ensure B*N matches BN, handle potential mismatch if needed
        if B * N != BN:
            # This might happen if e.g., batch size was 1 during inference but > 1 during training
            # Or if some cameras were dropped. Let's try to reshape BN based on B, N
            # This assumes the order is consistent. Add a warning.
            print(
                f"Warning: Mismatch B*N ({B*N}) vs BN ({BN}) in get_geometry_at_depth. Reshaping BN.")
            depth_values = depth_values.view(B, N, H, W)
        else:
            # Reshape even if matching for consistency
            depth_values = depth_values.view(B, N, H, W)

        # Ensure pixel_coords is on the correct device and shape [1, 1, H, W, 2]
        if pixel_coords.shape[0] != H or pixel_coords.shape[1] != W:
            raise ValueError(
                f"Pixel coord shape mismatch. Expected H={H}, W={W}, got {pixel_coords.shape}")
        pixel_coords = pixel_coords.to(depth_values.device).view(1, 1, H, W, 2)

        # Reshape depth_values for broadcasting: [B, N, H, W, 1]
        depth_values = depth_values.unsqueeze(-1)

        # Create points in camera pixel coordinates (homogeneous): [B, N, H, W, 4]
        # Use x, y from pixel_coords, d from depth_values
        points_pixel_d = torch.cat([
            pixel_coords[..., 0:1].expand(B, N, H, W, 1) * depth_values,  # x*d
            pixel_coords[..., 1:2].expand(B, N, H, W, 1) * depth_values,  # y*d
            depth_values,                                               # d
            torch.ones_like(depth_values)                                # 1
        ], dim=-1)
        # Add final dim: [B, N, H, W, 4, 1]
        points_pixel_d = points_pixel_d.unsqueeze(-1)

        # --- Transformation Logic (similar to original get_geometry but starting from pixel_d) ---
        # Mats shapes: [B, N, 4, 4]
        sensor2ego_mats = torch.eye(4, device=rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        sensor2ego_mats[:, :, :3, :3] = rots
        sensor2ego_mats[:, :, :3, 3] = trans

        intrin_mats = torch.eye(4, device=intrins.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        intrin_mats[:, :, :3, :3] = intrins

        ida_mats = torch.eye(4, device=post_rots.device).view(
            1, 1, 4, 4).repeat(B, N, 1, 1)
        ida_mats[:, :, :3, :3] = post_rots
        ida_mats[:, :, :3, 3] = post_trans

        # Apply image augmentation inverse
        # Reshape mats for broadcasting: [B, N, 1, 1, 4, 4]
        ida_mats_inv = ida_mats.inverse().view(B, N, 1, 1, 4, 4)
        # Perform matmul: [B, N, 1, 1, 4, 4] @ [B, N, H, W, 4, 1] -> [B, N, H, W, 4, 1]
        points_aug_inv = ida_mats_inv.matmul(points_pixel_d)

        # Apply camera to ego transformation (Combine sensor2ego and intrin_inv)
        # Reshape combine for broadcasting: [B, N, 1, 1, 4, 4]
        combine = sensor2ego_mats.matmul(
            torch.inverse(intrin_mats)).view(B, N, 1, 1, 4, 4)
        # Perform matmul: [B, N, 1, 1, 4, 4] @ [B, N, H, W, 4, 1] -> [B, N, H, W, 4, 1]
        points_ego = combine.matmul(points_aug_inv)

        # Return 3D coordinates: [B, N, H, W, 3]
        return points_ego[..., :3, 0]

    def voxel_pooling(self, geom_feats, x):
        """
        Args:
            geom_feats: [B, N, H, W, 3] - Refined 3D coordinates for each pixel.
            x: [B, N, H, W, C] - Context features for each pixel.
        Returns:
            [B, C, nx0, nx1] - BEV features
        """
        B, N, H, W, C = x.shape
        Nprime = B * N * H * W  # Total number of points

        # flatten x: [Nprime, C]
        x = x.reshape(Nprime, C)

        # flatten indices: [Nprime, 3]
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)

        # Create batch index
        batch_ix = torch.arange(B, device=x.device).view(
            B, 1, 1, 1).expand(B, N, H, W)
        batch_ix = batch_ix.reshape(Nprime)  # [Nprime]

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        x = x[kept]
        geom_feats = geom_feats[kept]
        batch_ix = batch_ix[kept]

        # Combine geom feats and batch index for ranking
        geom_feats_batched = torch.cat(
            (geom_feats, batch_ix.unsqueeze(1)), 1)  # [Points, 4]

        # get tensors from the same voxel next to each other
        # Sort by x, then y, then z, then batch
        ranks = geom_feats_batched[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats_batched[:, 1] * (self.nx[2] * B) \
            + geom_feats_batched[:, 2] * B \
            + geom_feats_batched[:, 3]
        sorts = ranks.argsort()
        x, geom_feats_batched, ranks = x[sorts], geom_feats_batched[sorts], ranks[sorts]

        # Use cumsum trick for summing features in the same voxel
        # Make sure cumsum_trick is imported or defined
        if x.shape[0] == 0:
            # Handle empty tensor case after filtering
            final = torch.zeros((B, C, int(self.nx[0]), int(
                self.nx[1])), device=x.device, dtype=x.dtype)
            return final

        x, geom_feats_batched = cumsum_trick(x, geom_feats_batched, ranks)

        # griddify (B x C x Z x X x Y)
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        final = torch.zeros((B, C, nx2, nx0, nx1),
                            device=x.device, dtype=x.dtype)

        # Use batch index from geom_feats_batched
        final[geom_feats_batched[:, 3], :, geom_feats_batched[:, 2],
              geom_feats_batched[:, 0], geom_feats_batched[:, 1]] = x

        # collapse Z (summing features before max-pooling might be better?)
        # Let's keep max pooling for now, but summing is another option.
        final = torch.max(final, dim=2)[0]  # [B, C, nx0, nx1]

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        获取BEV体素特征 (Method B: Refined Depth Projection with ResNet-50 + FPN)
        """
        B, N, _, imH, imW = x.shape

        # 1. Extract multi-scale features using ResNet + FPN
        x_2d = x.view(B * N, 3, imH, imW)
        # Pass through backbone (ResNet feature extractor)
        # Output: {'feat1': ..., 'feat2': ...}
        backbone_features = self.backbone(x_2d)

        # Pass backbone features through FPN
        # Output: OrderedDict, keys likely match input + 'pool'?
        fpn_features = self.fpn(backbone_features)
        # Or default keys '0', '1', '2', '3', ... for P2-P5
        # Let's assume keys match backbone_features input for now.

        # Select the highest resolution FPN feature map (corresponds to 'feat1' input -> P2 level)
        # Check actual keys if this fails. FPN might rename them.
        # If keys are numerical strings '0', '1', '2', '3', use '0'.
        # Assuming the key for the highest res output (P2) is the first key from FPN output
        # Or explicitly use the key corresponding to 'feat1' if FPN preserves it.
        # Let's try accessing via the original input key name first.
        fpn_key_high_res = 'feat1'  # Or potentially '0' if FPN uses default indexing
        if fpn_key_high_res not in fpn_features:
            # Fallback if FPN renames keys (e.g., to numerical indices '0', '1', ...)
            potential_keys = list(fpn_features.keys())
            print(
                f"Warning: FPN key '{fpn_key_high_res}' not found. Available keys: {potential_keys}. Attempting to use first key: '{potential_keys[0]}'")
            if not potential_keys:
                raise KeyError("FPN returned no feature maps.")
            # Use the first key (likely highest res P2)
            fpn_key_high_res = potential_keys[0]

        img_feats = fpn_features[fpn_key_high_res]
        BN, C_feat, fH, fW = img_feats.shape  # Get dimensions from FPN output

        if BN != B * N:
            print(f"Warning: BN ({BN}) != B*N ({B*N}) after FPN.")
            # Handle potential reshaping or error if needed, though unlikely here

        # 1.1 Apply optional feature enhancement to the selected FPN features
        if self.feat_enhancement:
            if hasattr(self, 'ema'):
                img_feats = img_feats * self.ema(img_feats)
            if hasattr(self, 'feat_attn'):
                img_feats = img_feats * self.feat_attn(img_feats)

        # 2. Prepare camera parameter dictionary
        mats_dict = {
            'sensor2ego_mats': torch.eye(4, device=rots.device).view(1, 1, 4, 4).repeat(B, N, 1, 1),
            'intrin_mats': torch.eye(4, device=intrins.device).view(1, 1, 4, 4).repeat(B, N, 1, 1),
            'rots': rots, 'trans': trans, 'intrins': intrins,
            'post_rots': post_rots, 'post_trans': post_trans
        }
        mats_dict['sensor2ego_mats'][:, :, :3, :3] = rots
        mats_dict['sensor2ego_mats'][:, :, :3, 3] = trans
        mats_dict['intrin_mats'][:, :, :3, :3] = intrins
        # Add post_rots/trans to ida_mats within mats_dict if DepthNet uses them
        # Current DepthNet uses flattened sensor2ego and intrinsics directly
        # Let's ensure DepthNet's film_mlp input dim calculation is robust or updated

        # 3. Predict depth bins, residuals, and context features using DepthNet
        # DepthNet now takes FPN features as input
        depth_net_output = self.depth_net(img_feats, mats_dict)
        bin_logits = depth_net_output['bin_logits']
        bin_residuals = depth_net_output['bin_residuals']
        context = depth_net_output['context']  # Shape: [BN, self.camC, fH, fW]

        # 4. Calculate refined depth per pixel
        depth_probs = bin_logits.softmax(dim=1)
        argmax_bin = depth_probs.argmax(dim=1)
        residuals_at_argmax = torch.gather(
            bin_residuals, 1, argmax_bin.unsqueeze(1)
        ).squeeze(1)
        # Sigmoid [-inf, inf] -> [0, 1]
        scaled_residuals = residuals_at_argmax.sigmoid()
        d_centers_reshaped = self.d_centers.view(1, self.D, 1, 1)
        d_center_at_argmax = torch.gather(
            # Use fH, fW from FPN feature map
            d_centers_reshaped.expand(BN, -1, fH, fW),
            1,
            argmax_bin.unsqueeze(1)
        ).squeeze(1)
        # Refined depth: Center + (scaled_residual - 0.5) * step
        d_refined = d_center_at_argmax + (scaled_residuals - 0.5) * self.d_step

        # 5. Calculate 3D geometry using refined depth
        # Create pixel coordinate grid matching the *selected FPN feature map size* (fH, fW)
        # Original image dims (ogfH, ogfW) for linspace coordinate range
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # Generate coords based on original image size, map to feature grid points (fH, fW)
        pixel_x = torch.linspace(
            0, ogfW - 1, fW, device=x.device)  # Sample at fW points
        pixel_y = torch.linspace(
            0, ogfH - 1, fH, device=x.device)  # Sample at fH points
        pixel_grid_y, pixel_grid_x = torch.meshgrid(
            pixel_y, pixel_x, indexing='ij')
        # Shape: [fH, fW, 2]
        pixel_coords = torch.stack([pixel_grid_x, pixel_grid_y], dim=-1)

        geom_feats_refined = self.get_geometry_at_depth(
            pixel_coords, d_refined, rots, trans, intrins, post_rots, post_trans
        )  # Output: [B, N, fH, fW, 3]

        # 6. Prepare context features for pooling
        # Reshape context from [BN, C, fH, fW] to [B, N, fH, fW, C]
        context_features = context.view(
            B, N, self.camC, fH, fW).permute(0, 1, 3, 4, 2)

        # 7. Voxel Pooling
        # Input geom: [B, N, fH, fW, 3], Input context: [B, N, fH, fW, C]
        bev_feat = self.voxel_pooling(geom_feats_refined, context_features)
        # Output: [B, C, nx0, nx1]

        return bev_feat

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

        # --- 修改：调用新的 BEV 编码器和头 ---
        # 进行BEV编码和预测头处理
        # self.bevencode 现在返回一个包含多个预测头输出的字典
        preds = self.bevencode(x)
        # --- 修改结束 ---

        return preds


# --- 新增：CenterPoint 风格的 BEV 编码器和检测头 ---
def make_conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
    """辅助函数创建 Conv-BN-ReLU 块"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class BEVEncoderCenterPointHead(nn.Module):
    """
    CenterPoint 风格的 BEV 特征编码器和检测头
    包含一个 FPN 结构的 BEV Backbone 和多个任务预测头
    """

    def __init__(self, inC, num_classes, backbone_out_channels=128, head_inter_channels=64):
        """
        Args:
            inC (int): 输入 BEV 特征图的通道数 (来自 voxel pooling, e.g., 64)
            num_classes (int): 检测类别数量
            backbone_out_channels (int): BEV Backbone 输出特征图的通道数
            head_inter_channels (int): 每个预测头中间层的通道数
        """
        super().__init__()
        self.num_classes = num_classes

        C1 = inC  # 64
        C2 = 128
        C3 = 256
        C_up1 = 128
        C_up2 = backbone_out_channels  # 128

        # --- BEV Backbone (FPN-like) ---
        # Downsampling path
        self.down1 = make_conv_block(inC, C1)  # Stride 1
        self.down2 = make_conv_block(C1, C2, stride=2)  # Stride 2 -> H/2, W/2
        self.down3 = make_conv_block(C2, C3, stride=2)  # Stride 2 -> H/4, W/4

        # Upsampling path
        self.up1_conv = make_conv_block(C3, C_up1)
        self.up1_upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up1_fuse = make_conv_block(
            C_up1 + C2, C_up1)  # Fuse upsampled + skip

        self.up2_conv = make_conv_block(C_up1, C_up2)
        self.up2_upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up2_fuse = make_conv_block(
            C_up2 + C1, C_up2)  # Fuse upsampled + skip
        # Final BEV feature map shape: [B, C_up2, H_bev, W_bev]
        # --- End BEV Backbone ---

        # --- Detection Heads ---
        heads = {}
        # 1. Heatmap Head
        heads['heatmap'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(head_inter_channels, num_classes,
                      kernel_size=1)  # Final conv, no BN/ReLU
            # Sigmoid activation applied during loss calculation or post-processing
        )
        # 2. Center Offset Head (x, y)
        heads['offset'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(head_inter_channels, 2, kernel_size=1)  # Output dx, dy
        )
        # 3. Height (z) Head
        heads['z_coord'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(head_inter_channels, 1, kernel_size=1)  # Output z
        )
        # 4. Dimension (w, l, h) Head
        heads['dimension'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            # Output w, l, h (often log scale)
            nn.Conv2d(head_inter_channels, 3, kernel_size=1)
        )
        # 5. Rotation (sin(yaw), cos(yaw)) Head
        heads['rotation'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            # Output sin(yaw), cos(yaw)
            nn.Conv2d(head_inter_channels, 2, kernel_size=1)
        )
        # 6. Velocity (vx, vy) Head
        heads['velocity'] = nn.Sequential(
            make_conv_block(backbone_out_channels,
                            head_inter_channels, kernel_size=3, padding=1),
            nn.Conv2d(head_inter_channels, 2, kernel_size=1)  # Output vx, vy
        )

        self.heads = nn.ModuleDict(heads)
        # --- End Detection Heads ---

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input BEV features [B, C_in, H_bev, W_bev]
        Returns:
            dict[str, torch.Tensor]: Dictionary containing predictions from each head.
        """
        # BEV Backbone Forward
        d1 = self.down1(x)    # [B, C1, H, W]
        d2 = self.down2(d1)   # [B, C2, H/2, W/2]
        d3 = self.down3(d2)   # [B, C3, H/4, W/4]

        u1_ = self.up1_conv(d3)        # [B, C_up1, H/4, W/4]
        u1 = self.up1_upsample(u1_)    # [B, C_up1, H/2, W/2]
        f1 = torch.cat([u1, d2], dim=1)  # [B, C_up1 + C2, H/2, W/2]
        f1 = self.up1_fuse(f1)         # [B, C_up1, H/2, W/2]

        u2_ = self.up2_conv(f1)        # [B, C_up2, H/2, W/2]
        u2 = self.up2_upsample(u2_)    # [B, C_up2, H, W]
        f2 = torch.cat([u2, d1], dim=1)  # [B, C_up2 + C1, H, W]
        bev_features = self.up2_fuse(f2)  # [B, C_up2, H, W]

        # Heads Forward
        outputs = {}
        for head_name, head_module in self.heads.items():
            outputs[head_name] = head_module(bev_features)

        return outputs
# --- 新增结束 ---
