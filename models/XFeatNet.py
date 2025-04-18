import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.tools import gen_dx_bx, cumsum_trick
    from models.xfeat_cam_encoder import XFeatCamEncoder, XFeatBEVEncoder, XFeatMultiModalEncoder
except ImportError:
    print("无法导入XFeatNet模块")
    raise


class XFeatBEVNet(nn.Module):
    """
    基于XFeat的BEV网络
    整合多个组件以实现高效、稳定的3D检测
    """

    def __init__(self, grid_conf, data_aug_conf, outC, num_classes=10,
                 detection_head=True, lidar_channels=None):
        super().__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf
        self.outC = outC
        self.num_classes = max(1, num_classes)  # 确保类别数至少为1
        self.detection_head = detection_head
        self.use_lidar = lidar_channels is not None

        # 网格参数
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # 下采样率
        self.downsample = 16
        self.camC = 64

        # 创建视锥体
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape

        # 投影矩阵缓存，优化计算性能
        self.proj_matrix_cache = None

        # 相机编码器 - 使用XFeat设计的轻量高效结构
        self.camencode = XFeatCamEncoder(
            self.D, self.camC, self.downsample, enhance_features=True)

        # BEV编码器
        if self.use_lidar:
            # 如果使用LiDAR，就创建融合编码器
            self.bevencode = nn.Sequential(
                XFeatMultiModalEncoder(self.camC, lidar_channels, 128, 128),
                XFeatBEVEncoder(128, outC)
            )
        else:
            # 仅使用相机
            self.bevencode = XFeatBEVEncoder(self.camC, outC)

        # 特征增强标志
        self.feat_enhancement = True

        # 使用数值安全的EPS常数
        self.eps = 1e-6

    def create_frustum(self):
        """创建视锥体投影网格"""
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

        优化版本：使用批处理和矩阵运算加速计算，同时确保数值稳定性
        """
        B, N = trans.shape[0], trans.shape[1]

        # 计算投影矩阵 (camera -> ego)
        # 检查缓存是否有效
        if self.proj_matrix_cache is None or \
           (hasattr(self.proj_matrix_cache, "__len__") and len(self.proj_matrix_cache) >= 3 and
                (self.proj_matrix_cache[2].shape != trans.shape or not torch.allclose(self.proj_matrix_cache[2], trans))):

            # 计算内参矩阵的逆矩阵 - 添加数值稳定性检查
            # 使用try-except以防内参矩阵不可逆
            try:
                intrins_inv = torch.inverse(
                    intrins.view(B*N, 3, 3)).view(B, N, 3, 3)
            except RuntimeError:
                # 如果不可逆，添加小的对角项
                modified_intrins = intrins.clone()
                diag_mask = torch.eye(3, device=intrins.device).unsqueeze(
                    0).unsqueeze(0).expand(B, N, 3, 3)
                modified_intrins = modified_intrins + diag_mask * self.eps
                intrins_inv = torch.inverse(
                    modified_intrins.view(B*N, 3, 3)).view(B, N, 3, 3)

            # 计算相机到自车的变换矩阵
            cam_to_ego = torch.matmul(rots, intrins_inv)

            # 计算后处理旋转的逆矩阵
            try:
                post_rots_inv = torch.inverse(
                    post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)
            except RuntimeError:
                # 如果不可逆，添加小的对角项
                modified_post_rots = post_rots.clone()
                diag_mask = torch.eye(3, device=post_rots.device).unsqueeze(
                    0).unsqueeze(0).expand(B, N, 3, 3)
                modified_post_rots = modified_post_rots + diag_mask * self.eps
                post_rots_inv = torch.inverse(
                    modified_post_rots.view(B*N, 3, 3)).view(B, N, 3, 3)

            # 缓存计算的矩阵
            self.proj_matrix_cache = (cam_to_ego, post_rots_inv, trans.clone())

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
        提取相机特征
        返回 B x N x D x H/downsample x W/downsample x C

        优化版本：使用XFeatCamEncoder提高效率并确保维度正确
        """
        B, N, C, imH, imW = x.shape

        # 批处理相机图像编码
        x = x.reshape(B * N, C, imH, imW)

        # 使用相机编码器获取特征
        # 预期输出: [B*N, C, D, H', W']
        x = self.camencode(x)

        # 检查输出形状是否符合预期
        if len(x.shape) != 5:
            raise ValueError(
                f"相机编码器输出形状错误: 预期5D张量 [B*N, C, D, H', W']，但得到了 {x.shape}")

        _, C_out, D_out, H_out, W_out = x.shape

        # 验证输出维度是否符合预期
        if D_out != self.D:
            print(f"警告: 输出深度维度 {D_out} 与预期 {self.D} 不匹配，将调整")
            # 调整深度维度
            x = F.interpolate(x.transpose(1, 2), size=(C_out, H_out, W_out),
                              mode='nearest').transpose(1, 2)
            _, C_out, D_out, H_out, W_out = x.shape

        # 重塑为期望的输出格式
        x = x.reshape(B, N, C_out, D_out, H_out, W_out)
        x = x.permute(0, 1, 3, 4, 5, 2)  # [B, N, D, H', W', C]

        # 额外的数值稳定性检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("警告: 相机特征中包含NaN或Inf值，将替换为0")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x

    def voxel_pooling(self, geom_feats, x):
        """
        体素池化 - 将特征投影到BEV网格

        优化版本：处理边界情况，添加维度检查和数值稳定性
        """
        B, N, D, H, W, C = x.shape

        # 检查几何特征形状是否正确
        if geom_feats.shape[:3] != (B, N, D) or geom_feats.shape[-1] != 3:
            expected_shape = (B, N, D, H, W, 3)
            raise ValueError(
                f"几何特征形状错误: 预期 {expected_shape}，但得到了 {geom_feats.shape}")

        # 输出检查
        print(f"体素池化输入: geom_feats {geom_feats.shape}, x {x.shape}")

        # 确保输入不包含NaN或Inf
        if torch.isnan(geom_feats).any() or torch.isinf(geom_feats).any():
            print("警告: 几何特征中包含NaN或Inf值，将替换为0")
            geom_feats = torch.nan_to_num(
                geom_feats, nan=0.0, posinf=0.0, neginf=0.0)

        Nprime = B * N * D * H * W

        # 展平特征
        x = x.reshape(Nprime, C)

        # 计算体素索引 - 使用更稳定的方法
        # 确保bx和dx不包含NaN/Inf
        safe_bx = torch.nan_to_num(self.bx, nan=0.0, posinf=1e6, neginf=-1e6)
        safe_dx = torch.nan_to_num(self.dx, nan=1.0, posinf=1.0, neginf=1.0)

        # 避免除以接近零的值
        safe_dx = torch.clamp(safe_dx, min=1e-6)

        # 计算体素索引，clamp防止出现异常值
        geom_feats = ((geom_feats - (safe_bx - safe_dx / 2.)) / safe_dx)
        geom_feats = torch.clamp(geom_feats, min=-1e6, max=1e6)  # 限制范围，防止极端值
        geom_feats = geom_feats.long()
        geom_feats = geom_feats.reshape(Nprime, 3)

        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # 过滤边界外的点
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])

        # 检查是否有有效点
        if kept.sum() == 0:
            print("警告: 视锥体中没有有效点，返回全零特征。这可能表明几何计算有问题。")
            nx0 = int(self.nx[0].item())
            nx1 = int(self.nx[1].item())
            nx2 = int(self.nx[2].item())
            return torch.zeros((B, C * nx2, nx0, nx1), device=x.device)

        x = x[kept]
        geom_feats = geom_feats[kept]

        # 打印保留的点数量
        print(f"体素池化保留点数: {kept.sum().item()} / {Nprime}")

        # 计算唯一体素的累积和
        # 防止整数溢出，使用int64
        ranks = geom_feats[:, 0].to(torch.int64) * (self.nx[1].to(torch.int64) * self.nx[2].to(torch.int64) * B) \
            + geom_feats[:, 1].to(torch.int64) * (self.nx[2].to(torch.int64) * B) \
            + geom_feats[:, 2].to(torch.int64) * B \
            + geom_feats[:, 3].to(torch.int64)

        # 排序并合并相同rank的特征
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # 使用cumsum_trick，添加错误处理
        try:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        except RuntimeError as e:
            print(f"警告: cumsum_trick出错: {e}")
            # 使用备用方法处理
            try:
                # 检测到相同rank的连续段
                keep = torch.ones(
                    ranks.shape[0], device=ranks.device, dtype=torch.bool)
                keep[1:] = ranks[1:] != ranks[:-1]

                # 获取唯一rank的索引
                uniq_inv = torch.cumsum(keep, 0) - 1  # 每个元素映射到唯一rank的索引

                # 创建输出张量
                out_feats = torch.zeros_like(geom_feats)
                out_x = torch.zeros_like(x)

                # 填充最后一个具有相同rank的元素
                out_feats[keep] = geom_feats[keep]
                out_x[keep] = x[keep]

                # 返回结果
                x, geom_feats = out_x[keep], out_feats[keep]
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                # 如果所有方法都失败，返回全零特征
                nx0 = int(self.nx[0].item())
                nx1 = int(self.nx[1].item())
                nx2 = int(self.nx[2].item())
                return torch.zeros((B, C * nx2, nx0, nx1), device=x.device)

        # 创建BEV网格
        nx0 = int(self.nx[0].item())
        nx1 = int(self.nx[1].item())
        nx2 = int(self.nx[2].item())

        # 使用索引填充体素网格
        final = torch.zeros((B, C, nx2, nx0, nx1), device=x.device)

        # 安全索引，避免可能的越界访问
        b_idx = geom_feats[:, 3].long().clamp(0, B-1)
        z_idx = geom_feats[:, 2].long().clamp(0, nx2-1)
        x_idx = geom_feats[:, 0].long().clamp(0, nx0-1)
        y_idx = geom_feats[:, 1].long().clamp(0, nx1-1)

        # 设置值
        final[b_idx, :, z_idx, x_idx, y_idx] = x

        # 将Z维度的特征拼接
        final = torch.cat(final.unbind(dim=2), 1)

        # 检查特征中是否有NaN或Inf
        if torch.isnan(final).any() or torch.isinf(final).any():
            print("警告: BEV特征中包含NaN或Inf值，将替换为0")
            final = torch.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """获取体素特征"""
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans, lidar_bev=None):
        """
        模型前向传播
        Args:
            x: 图像输入
            rots: 旋转矩阵
            trans: 平移向量
            intrins: 相机内参
            post_rots: 图像后处理旋转
            post_trans: 图像后处理平移
            lidar_bev: LiDAR BEV特征 (可选)
        Returns:
            分类、回归和IoU预测结果的字典
        """
        # 获取体素表示
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)

        # 在使用LiDAR的情况下合并特征
        if lidar_bev is not None and self.use_lidar:
            # 确保LiDAR特征与相机特征尺寸一致
            if lidar_bev.shape[2:] != x.shape[2:]:
                lidar_bev = F.interpolate(
                    lidar_bev,
                    size=x.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            # BEV编码器会自动处理特征融合
            preds = self.bevencode((x, lidar_bev))
        else:
            # 仅使用相机特征
            preds = self.bevencode(x)

        # 组织输出格式
        if isinstance(preds, dict):
            # 如果已经是字典格式，直接返回
            return preds
        elif isinstance(preds, torch.Tensor):
            # 将张量分割为所需的组件
            cls_channels = self.num_classes
            reg_channels = 9  # 8个边界框参数 + 1个角度
            iou_channels = 1

            # 确保通道数匹配
            if preds.shape[1] >= cls_channels + reg_channels + iou_channels:
                return {
                    'cls_pred': preds[:, :cls_channels],
                    'reg_pred': preds[:, cls_channels:cls_channels+reg_channels],
                    'iou_pred': preds[:, cls_channels+reg_channels:cls_channels+reg_channels+iou_channels]
                }
            else:
                # 如果通道数不匹配，创建合适的输出
                return {
                    'cls_pred': preds[:, :min(cls_channels, preds.shape[1])],
                    'reg_pred': torch.zeros((preds.shape[0], reg_channels, preds.shape[2], preds.shape[3]),
                                            device=preds.device),
                    'iou_pred': torch.zeros((preds.shape[0], iou_channels, preds.shape[2], preds.shape[3]),
                                            device=preds.device)
                }
        else:
            raise ValueError(f"不支持的预测类型: {type(preds)}")

# 创建XFeatBEVNet的工厂函数


def create_xfeat_bevnet(grid_conf, data_aug_conf, outC, num_classes=10, lidar_channels=None):
    """
    创建XFeatBEVNet
    Args:
        grid_conf: 网格配置
        data_aug_conf: 数据增强配置
        outC: 输出通道数
        num_classes: 类别数量
        lidar_channels: LiDAR通道数 (可选)
    Returns:
        XFeatBEVNet: 创建的模型
    """
    return XFeatBEVNet(grid_conf, data_aug_conf, outC, num_classes, lidar_channels=lidar_channels)
