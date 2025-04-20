from .tools import gen_dx_bx, cumsum_trick
import math

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

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
        model: 模型类型 ('lss', 'beve', 'fusion', '3d', 'xfeat')
        num_classes: 类别数量
        lidar_channels: LiDAR通道数量（仅用于融合模型）
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
            return BEVENet(grid_conf, data_aug_conf, outC, num_classes=actual_num_classes, model_type='beve')
            # --- 修改结束 ---
        elif model == 'fusion':
            # 使用多模态融合模型
            actual_num_classes = max(1, num_classes)
            return BEVENet(grid_conf, data_aug_conf, actual_num_classes*9, num_classes=actual_num_classes, model_type='beve')
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
        geom = self.get_geometry(
            self.frustum, rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


class CamEncode_rep(nn.Module):
    def __init__(self, D, C):
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


class DepthNet(nn.Module):
    """深度预测网络，参考BEVDepth实现"""

    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.bn = nn.BatchNorm1d(27)  # 相机内外参矩阵扁平化后的尺寸

        # 深度MLP
        self.depth_mlp = nn.Sequential(
            nn.Linear(27, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Context MLP
        self.context_mlp = nn.Sequential(
            nn.Linear(27, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
        )

        # SE层用于相机感知特征增强
        self.depth_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.context_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 深度预测头 - 输出 depth_channels (D)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels,
                      kernel_size=1, padding=0),  # 输出 D 通道
        )

        # 上下文特征头 - 输出 context_channels (C)
        self.context_conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, context_channels,
                      kernel_size=1, padding=0),  # 输出 C 通道
        )

    def forward(self, x, mats_dict):
        """前向传播
        Args:
            x: 图像特征 [B*N, C, H, W] - 这里的 C 应该是 in_channels
            mats_dict: 包含相机内外参的字典
        Returns:
            # depth_digit: 深度预测 [B*N, D, H, W]
            # context: 上下文特征 [B*N, C, H, W]
            torch.Tensor: 拼接后的深度和上下文特征 [B*N, D+C, H, W]
        """
        # 获取输入特征的形状
        BN, C_in, H, W = x.shape

        # 预处理特征，得到中间特征 x_reduced
        x_reduced = self.reduce_conv(x)  # Shape: [BN, mid_channels, H, W]
        mid_channels = x_reduced.shape[1]

        # 处理相机参数
        sensor2ego_mat = mats_dict['sensor2ego_mats']  # [B, N, 4, 4]
        intrin_mat = mats_dict['intrin_mats']  # [B, N, 4, 4]
        # ida_mat = mats_dict.get('ida_mats', torch.eye(4).to(x.device).view(
        #     1, 1, 4, 4).repeat(sensor2ego_mat.shape[0], sensor2ego_mat.shape[1], 1, 1))

        # 合并并展平相机参数
        B, N = sensor2ego_mat.shape[:2]
        # 创建相机识别向量 (需要从mats_dict提取有效部分)
        # 假设 sensor2ego_mat 和 intrin_mat 足够提供 27 个参数
        # 确保提取的参数与bn层的输入维度匹配
        param_mat = torch.cat([
            sensor2ego_mat.view(B*N, -1),
            intrin_mat.view(B*N, -1),
        ], dim=1)[:, :27].contiguous()  # Shape: [BN, 27]

        # 应用批归一化
        param_mat = self.bn(param_mat)  # Shape: [BN, 27]

        # 1. 使用MLP生成相机参数引导
        depth_guide = self.depth_mlp(param_mat).view(
            BN, mid_channels, 1, 1)  # Shape: [BN, mid, 1, 1]
        context_guide = self.context_mlp(param_mat).view(
            BN, mid_channels, 1, 1)  # Shape: [BN, mid, 1, 1]

        # 2. 使用SE层计算注意力权重 (基于引导信息和 x_reduced)
        # 注意：这里SE层输入的是 x_reduced * guide，这与标准SE不同，但遵循我们之前的逻辑
        # 如果要严格遵循BEVDepth的SELayer，需要修改SELayer类和这里的调用
        depth_attention = self.depth_se(
            x_reduced * depth_guide)  # Shape: [BN, mid, 1, 1]
        context_attention = self.context_se(
            x_reduced * context_guide)  # Shape: [BN, mid, 1, 1]

        # 3. 将注意力权重应用到 x_reduced
        x_depth_attended = x_reduced * \
            depth_attention  # Shape: [BN, mid, H, W]
        x_context_attended = x_reduced * \
            context_attention  # Shape: [BN, mid, H, W]

        # 4. 使用卷积头预测深度和上下文
        depth_digit = self.depth_conv(x_depth_attended)  # Shape: [BN, D, H, W]
        # Shape: [BN, C, H, W] - 这里的 C 是 context_channels
        context = self.context_conv2(x_context_attended)

        # 拼接结果
        # Shape: [BN, D+C, H, W]
        output = torch.cat([depth_digit, context], dim=1)
        return output


class BEVENet(nn.Module):
    """
    BEVENet: 基于纯卷积的高效3D目标检测BEV网络
    参考论文: Towards Efficient 3D Object Detection in Bird's-Eye-View Space for Autonomous Driving: A Convolutional-Only Approach
    """

    def __init__(self, grid_conf, data_aug_conf, outC, num_classes=10, detection_head=True, model_type='xfeat'):
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

        # Remove downsample and pre-calculated frustum
        # self.downsample = 16
        self.camC = 64
        # self.frustum = self.create_frustum()
        # self.D determined dynamically later
        # self.D, _, _, _ = self.frustum.shape

        # D will be determined based on grid_conf['dbound']
        d_start, d_end, d_step = self.grid_conf['dbound']
        self.D = int((d_end - d_start) / d_step)

        # 添加投影矩阵缓存，优化重复计算
        self.proj_matrix_cache = None

        # 相机编码器 - D is now determined from grid_conf
        self.camencode = CamEncode_rep(self.D, self.camC)

        # 添加深度预测网络
        self.depth_net = DepthNet(
            in_channels=self.camC,
            mid_channels=128,
            context_channels=self.camC,
            depth_channels=self.D
        )

        # 使用纯卷积的BEV编码器
        # --- 修改：传递 num_classes 给 BEVEncoder_BEVE ---
        self.bevencode = BEVEncoder_BEVE(
            inC=self.camC, outC=self.outC, num_classes=self.num_classes)
        # --- 修改结束 ---

        # 添加3D检测头
        # 添加特征增强模块
        self.ema = EMA(self.camC)
        self.feat_attn = PSA(self.camC, self.camC)
        self.feat_enhancement = True

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

        # 创建像素坐标 (使用 fH, fW and target image size ogfH, ogfW)
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
        """
        获取相机特征
        x: [B, N, 3, H, W] - 输入图像
        返回: [B, N, C, D, fH, fW] - 相机特征
        """
        B, N, C, imH, imW = x.shape

        # 将批次和相机维度合并
        x = x.view(B*N, C, imH, imW)

        # 使用相机编码器提取特征
        x = self.camencode(x)

        # 重新整形为 [B, N, C, D, fH, fW]
        _, C, D, H, W = x.shape
        x = x.view(B, N, C, D, H, W)

        return x

    def _forward_depth_net(self, feat, mats_dict):
        """
        使用深度网络前向传播
        feat: [B*N, C, H, W] - 图像特征
        mats_dict: 相机内外参字典
        返回: [B*N, D+C, H, W] - 深度预测和上下文特征
        """
        return self.depth_net(feat, mats_dict)

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
            # drop_connect_rate = self.trunk._global_params.drop_connect_rate # 原代码
            # 修复：检查 _global_params 是否存在以及 drop_connect_rate 属性
            drop_connect_rate = 0.0
            if hasattr(self.trunk, '_global_params') and self.trunk._global_params is not None and hasattr(self.trunk._global_params, 'drop_connect_rate'):
                drop_connect_rate = self.trunk._global_params.drop_connect_rate

            if drop_connect_rate and drop_connect_rate > 0:  # 添加检查 > 0
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
            Conv(head_inter_channels, self.num_classes, k=1, act=False, bn=False)
        )

        # Regression Head
        self.reg_head = nn.Sequential(
            Conv(128, head_inter_channels, k=3, p=1),
            # Final 1x1 Conv: Output 9 regression values, no activation/BN
            Conv(head_inter_channels, 9, k=1, act=False, bn=False)
        )

        # IoU Head
        self.iou_head = nn.Sequential(
            Conv(128, head_inter_channels, k=3, p=1),
            # Final 1x1 Conv: Output 1 IoU logit, no activation/BN
            Conv(head_inter_channels, 1, k=1, act=False, bn=False)
        )
        # --- End Refactored Head ---

        # # --- Original Simple Head ---
        # # 分类头 - 预测每个位置的类别
        # self.cls_head = Conv(128, self.num_classes, k=1)
        #
        # # 回归头 - 预测边界框参数 (x,y,z,w,l,h,sin,cos,vel)
        # self.reg_head = Conv(128, 9, k=1)
        #
        # # IoU头 - 预测检测质量
        # self.iou_head = Conv(128, 1, k=1)
        # # --- End Original Simple Head ---

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
