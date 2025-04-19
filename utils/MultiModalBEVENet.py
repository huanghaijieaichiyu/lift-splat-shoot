import torch
import torch.nn as nn
import torch.nn.functional as F

from .CamEncoder_3d import CamEncoder_3d
from .FusionEncoder import LidarEncoder
from .common import Conv, PSA, SPPF, C2f, C2fCIB, EMA, ChannelAttention
from .RepVit import RepViTBlock
from src.tools import gen_dx_bx, cumsum_trick, QuickCumsum


class CrossAttention(nn.Module):
    """跨模态注意力模块，实现相机特征和雷达特征之间的注意力交互"""

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


class MultiModalBEVEncoder(nn.Module):
    """融合相机和LiDAR特征的BEV编码器"""

    def __init__(self, inC, outC):
        super(MultiModalBEVEncoder, self).__init__()
        # 确保输出通道数至少为1
        self.outC = max(1, outC)
        self.num_classes = self.outC // 9  # 假设outC是9*num_classes

        # 共享特征提取器
        self.shared_conv = nn.Sequential(
            RepViTBlock(inC, 64, 3, 2),
            C2f(64, 128, 1, True)
        )

        # 下采样阶段
        self.down_sample = nn.Sequential(
            RepViTBlock(128, 256, 3, 2),
            SPPF(256, 256),
            PSA(256, 256)
        )

        # 多尺度特征融合模块
        self.multi_scale_fusion = nn.Sequential(
            RepViTBlock(256, 256, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            C2fCIB(256, 384)
        )

        # 上采样阶段
        self.up_sample = nn.Sequential(
            RepViTBlock(384, 384, 3, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(384, 384, 3, 1, 1)
        )

        # 分类头 - 预测每个位置的类别
        self.cls_head = nn.Conv2d(384, self.num_classes, kernel_size=1)

        # 回归头 - 预测边界框参数 (x,y,z,w,l,h,sin,cos,vel)
        self.reg_head = nn.Conv2d(384, 9, kernel_size=1)

        # IoU头 - 预测检测质量
        self.iou_head = nn.Conv2d(384, 1, kernel_size=1)

    def forward(self, x):
        # 特征提取
        x = self.shared_conv(x)

        # 下采样特征
        x = self.down_sample(x)

        # 多尺度特征融合
        x = self.multi_scale_fusion(x)

        # 上采样得到细粒度特征
        shared_feats = self.up_sample(x)

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


class MultiModalBEVENet(nn.Module):
    """
    多模态BEVENet：结合相机和LiDAR特征的3D目标检测网络
    """

    def __init__(self, grid_conf, data_aug_conf, outC, num_classes=10,
                 lidar_channels=62, model_type='fusion', use_enhanced_fusion=True):
        super(MultiModalBEVENet, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = outC
        self.num_classes = max(1, num_classes)
        self.use_enhanced_fusion = use_enhanced_fusion

        # 网格参数初始化
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]

        # 初始化投影矩阵缓存
        self.proj_matrix_cache = None

        # 设置use_quickcumsum标志
        self.use_quickcumsum = False  # 默认使用cumsum_trick

        # 相机特征提取
        self.cam_backbone = nn.Sequential(
            Conv(3, 32, k=7, s=2, p=3),
            Conv(32, 64, k=3, s=2, p=1),
            C2f(64, 64, n=1),
            Conv(64, 128, k=3, s=2, p=1),
            C2f(128, 128, n=2),
            Conv(128, self.camC, k=3, s=2, p=1)
        )

        # 3D特征提取
        self.depth_net = nn.Sequential(
            Conv(self.camC, self.camC, k=3, s=1, p=1),
            Conv(self.camC, self.D, k=1, s=1, p=0)
        )

        # LiDAR特征提取
        self.lidar_encoder = nn.Sequential(
            Conv(18, 32, k=3, s=1, p=1),
            Conv(32, 64, k=3, s=2, p=1),
            C2f(64, 64, n=1),
            SPPF(64, self.camC),
            PSA(self.camC, self.camC)
        )

        # 跨模态注意力
        self.cross_attn_cam2lidar = CrossAttention(self.camC)
        self.cross_attn_lidar2cam = CrossAttention(self.camC)

        # 特征融合
        fusion_channels = 2 * self.camC
        self.fusion_net = Conv(fusion_channels, fusion_channels, k=1)

        # BEV头部网络
        self.bev_head = nn.Sequential(
            Conv(fusion_channels, 128, k=1, s=1),
            C2f(128, 256, n=2),
            SPPF(256, 256),
            Conv(256, 384, k=1, s=1)
        )

        # 检测头
        self.cls_head = nn.Conv2d(384, self.num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(384, 9, kernel_size=1)
        self.iou_head = nn.Conv2d(384, 1, kernel_size=1)

    def create_frustum(self):
        """创建视锥体投影网格"""
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
        """
        B, N, _ = trans.shape

        # 计算投影矩阵 (camera -> ego)
        # 初始化或更新投影矩阵缓存
        if self.proj_matrix_cache is None:
            # 首次计算所有矩阵
            intrins_inv = torch.inverse(
                intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
            cam_to_ego = torch.matmul(rots, intrins_inv)
            post_rots_inv = torch.inverse(
                post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)
            self.proj_matrix_cache = {
                'cam_to_ego': cam_to_ego,
                'post_rots_inv': post_rots_inv,
                'trans': trans.clone(),
                'batch_size': B,
                'num_cams': N
            }
        elif (self.proj_matrix_cache['batch_size'] != B or
              self.proj_matrix_cache['num_cams'] != N or
              not torch.allclose(self.proj_matrix_cache['trans'], trans)):
            # 如果批次大小、相机数量变化或trans发生变化，重新计算
            intrins_inv = torch.inverse(
                intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
            cam_to_ego = torch.matmul(rots, intrins_inv)
            post_rots_inv = torch.inverse(
                post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)
            self.proj_matrix_cache = {
                'cam_to_ego': cam_to_ego,
                'post_rots_inv': post_rots_inv,
                'trans': trans.clone(),
                'batch_size': B,
                'num_cams': N
            }

        # 从缓存中获取矩阵
        cam_to_ego = self.proj_matrix_cache['cam_to_ego']
        post_rots_inv = self.proj_matrix_cache['post_rots_inv']

        # 获取视锥体网格点的形状
        D, fH, fW, _ = self.frustum.shape

        # 撤销后处理变换
        points = self.frustum.unsqueeze(0).unsqueeze(
            0) - post_trans.view(B, N, 1, 1, 1, 3)

        # 重塑为适合批量矩阵乘法的形状
        points = points.reshape(B, N, -1, 3)
        points = torch.matmul(points, post_rots_inv.transpose(-1, -2))
        points = points.reshape(B, N, D, fH, fW, 3)

        # 准备相机到自车坐标转换
        points_with_depth = torch.cat((
            points[..., :2] * points[..., 2:3],
            points[..., 2:3]
        ), dim=-1)

        # 应用相机到自车的变换
        points = points_with_depth.reshape(B, N, -1, 3)
        points = torch.matmul(points, cam_to_ego.transpose(-1, -2))
        points = points + trans.view(B, N, 1, 3)
        points = points.reshape(B, N, D, fH, fW, 3)

        return points

    def get_cam_feats(self, x):
        """
        提取相机特征
        Args:
            x: [B, N, 3, H, W] - 相机图像
        Returns:
            features: [B, N, D, H', W', C] - 3D特征体素
        """
        B, N, C, imH, imW = x.shape

        # 批处理相机图像编码
        x = x.view(B * N, C, imH, imW)
        x = self.cam_backbone(x)  # [B*N, camC, H', W']

        # 记录backbone输出的特征维度
        _, C, H, W = x.shape

        # 预测深度分布
        depth = self.depth_net(x)  # [B*N, D, H', W']
        depth = depth.softmax(dim=1)  # 在深度维度上归一化

        # 生成3D特征，确保维度正确
        x = x.unsqueeze(2)  # [B*N, camC, 1, H', W']
        depth = depth.unsqueeze(1)  # [B*N, 1, D, H', W']
        x = x * depth  # [B*N, camC, D, H', W']

        # 重塑为所需的输出格式
        x = x.view(B, N, self.camC, self.D, H, W)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, H', W', C]

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        体素池化 - 将特征投影到BEV网格
        Args:
            geom_feats: [B, N, D, H', W', 3] - 几何特征
            x: [B, N, D, H', W', C] - 特征
        Returns:
            final: [B, C*D, X, Y] - BEV特征
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

        # 过滤边界外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        if kept.sum() == 0:
            nx0 = int(self.nx[0].item())
            nx1 = int(self.nx[1].item())
            nx2 = int(self.nx[2].item())
            return torch.zeros((B, C * nx2, nx0, nx1), device=x.device)

        x = x[kept]
        geom_feats = geom_feats[kept]

        # 计算唯一体素的累积和
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 始终使用cumsum_trick，避免QuickCumsum可能的问题
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # 创建BEV网格
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())
        final = torch.zeros((B, C, nx2, nx0, nx1), device=x.device)
        final[geom_feats[:, 3].long(), :, geom_feats[:, 2].long(),
              geom_feats[:, 0].long(), geom_feats[:, 1].long()] = x

        # 合并Z维度
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, cam_imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev):
        """
        结合相机和LiDAR信息生成BEV特征
        Args:
            cam_imgs: [B, N, 3, H, W] - 相机图像
            rots, trans, intrins, post_rots, post_trans: 相机参数
            lidar_bev: [B, C, H, W] - LiDAR BEV特征图
        Returns:
            bev_feats: [B, 384, H, W] - BEV特征
        """
        # 相机特征处理
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        cam_feats = self.get_cam_feats(cam_imgs)
        cam_voxels = self.voxel_pooling(geom, cam_feats)  # [B, C*D, X, Y]

        # LiDAR特征处理
        lidar_feats = self.lidar_encoder(lidar_bev)  # [B, C, X, Y]

        # 确保特征尺寸匹配
        if lidar_feats.shape[2:] != cam_voxels.shape[2:]:
            lidar_feats = F.interpolate(
                lidar_feats,
                size=cam_voxels.shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # 跨模态特征融合
        cam_feats_2d = cam_voxels.view(cam_voxels.shape[0], self.camC, -1,
                                       *cam_voxels.shape[2:]).mean(dim=2)
        cam_enhanced = self.cross_attn_cam2lidar(cam_feats_2d, lidar_feats)
        lidar_enhanced = self.cross_attn_lidar2cam(lidar_feats, cam_feats_2d)

        # 特征融合
        fused = torch.cat([cam_enhanced, lidar_enhanced], dim=1)
        fused = self.fusion_net(fused)

        # BEV特征提取
        bev_feats = self.bev_head(fused)

        return bev_feats

    def forward(self, cam_imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev):
        """
        模型前向传播
        Args:
            cam_imgs: [B, N, 3, H, W] - 相机图像
            rots: [B, N, 3, 3] - 旋转矩阵
            trans: [B, N, 3] - 平移向量
            intrins: [B, N, 3, 3] - 相机内参
            post_rots: [B, N, 3, 3] - 后处理旋转
            post_trans: [B, N, 3] - 后处理平移
            lidar_bev: [B, C, H, W] - LiDAR BEV特征图
        Returns:
            preds: dict - 预测结果字典
        """
        # 获取BEV特征
        bev_feats = self.get_voxels(
            cam_imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev)

        # 生成预测结果
        cls_pred = self.cls_head(bev_feats)
        reg_pred = self.reg_head(bev_feats)
        iou_pred = self.iou_head(bev_feats)

        return {
            'cls_pred': cls_pred,
            'reg_pred': reg_pred,
            'iou_pred': iou_pred
        }


def create_multimodal_bevenet(grid_conf, data_aug_conf, outC, num_classes=10, lidar_channels=64):
    """
    创建多模态BEVENet的工厂函数
    Args:
        grid_conf: 网格配置
        data_aug_conf: 数据增强配置
        outC: 输出通道数
        num_classes: 类别数量
        lidar_channels: LiDAR输入通道数
    Returns:
        MultiModalBEVENet实例
    """
    return MultiModalBEVENet(grid_conf, data_aug_conf, outC, num_classes, lidar_channels)
