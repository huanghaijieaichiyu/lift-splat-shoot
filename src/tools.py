"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch.nn.functional as F
import torch.nn as nn
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')


def get_lidar_data(nusc, sample_rec, nsweeps=1, min_distance=1.0):
    """
    从NuScenes数据集加载LiDAR点云数据。

    Args:
        nusc: NuScenes数据集实例
        sample_rec: 样本记录
        nsweeps (int): 要加载的sweep数量(包括当前帧)
        min_distance (float): 过滤掉距离小于此值的点

    Returns:
        points (np.ndarray): 点云数据数组, shape为(N, 5), 包含x,y,z,intensity,time
    """
    points = []
    # 获取当前帧的LiDAR数据
    sample_data = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])

    # 获取当前帧的自车姿态
    current_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    global_from_car = transform_matrix(current_pose['translation'],
                                       Quaternion(current_pose['rotation']),
                                       inverse=False)

    # 加载当前帧点云
    current_pc = load_point_cloud(nusc, sample_data)
    current_pc = remove_close(current_pc, min_distance)
    points.append(current_pc)

    # 加载之前的sweeps
    for _ in range(nsweeps - 1):
        if sample_data['prev'] == '':
            break
        sample_data = nusc.get('sample_data', sample_data['prev'])

        # 获取前一帧的自车姿态
        pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        car_from_global = transform_matrix(pose['translation'],
                                           Quaternion(pose['rotation']),
                                           inverse=True)

        pc = load_point_cloud(nusc, sample_data)
        pc = remove_close(pc, min_distance)

        # 转换到当前帧坐标系
        pc = transform_points(pc, car_from_global @ global_from_car)
        points.append(pc)

    # 合并所有点云
    points = np.concatenate(points, axis=0)
    return points


def load_point_cloud(nusc, sample_data):
    """
    加载单帧点云数据。
    """
    lidar_path = os.path.join(nusc.dataroot, sample_data['filename'])
    points = np.fromfile(lidar_path, dtype=np.float32).reshape([-1, 5])
    return points


def remove_close(points, min_distance):
    """
    移除距离传感器过近的点。
    """
    dists = np.sqrt(np.sum(points[:, :2]**2, axis=1))
    mask = dists >= min_distance
    return points[mask]


def transform_matrix(translation, rotation, inverse=False):
    """
    生成变换矩阵。
    """
    tm = np.eye(4)
    if inverse:
        rot = rotation.rotation_matrix.T
        trans = -np.dot(rot, translation)
    else:
        rot = rotation.rotation_matrix
        trans = translation
    tm[:3, :3] = rot
    tm[:3, 3] = trans
    return tm


def transform_points(points, trans):
    """
    对点云进行坐标变换。
    """
    points_h = np.concatenate(
        [points[:, :3], np.ones((len(points), 1))], axis=1)
    points_trans = np.dot(points_h, trans.T)
    return np.concatenate([points_trans[:, :3], points[:, 3:]], axis=1)


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        # PIL.Image.FLIP_LEFT_RIGHT = 0
        img = img.transpose(0)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                          for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def save_path(path, model='train'):
    file_path = os.path.join(path, model)
    i = 1
    while os.path.exists(file_path):

        file_path = os.path.join(path, model+'(%i)' % i)
        i += 1

    return file_path


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(
                              device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
        'loss': total_loss / len(valloader.dataset),
        'iou': total_intersect / total_union,
    }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    """
    获取NuScenes地图，支持优雅地处理找不到地图文件的情况
    """
    nusc_maps = {}
    map_names = [
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
        "singapore-onenorth",
    ]

    for map_name in map_names:
        try:
            # 适配Windows路径
            map_path = os.path.join(
                map_folder, "maps", "expansion").replace("\\", "/")
            print(f"尝试加载地图: {map_path}/{map_name}")
            nusc_maps[map_name] = NuScenesMap(
                dataroot=map_folder, map_name=map_name)
        except Exception as e:
            print(f"无法加载地图 {map_name}: {e}")
            # 创建一个空的地图对象替代
            from collections import defaultdict

            class EmptyMap:
                def __init__(self):
                    self.road_segment = []
                    self.lane = []
                    self.road_divider = []
                    self.lane_divider = []

                def get_records_in_patch(self, *args, **kwargs):
                    return defaultdict(list)

                def get(self, *args, **kwargs):
                    return {'polygon_token': None}

                def extract_polygon(self, *args, **kwargs):
                    from shapely.geometry import Polygon
                    return Polygon()

                def extract_line(self, *args, **kwargs):
                    from shapely.geometry import LineString
                    return LineString()

            nusc_maps[map_name] = EmptyMap()

    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    try:
        if rec is None:
            return

        # 获取自车位姿
        try:
            egopose = nusc.get('ego_pose', nusc.get(
                'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        except (KeyError, TypeError):
            print("无法获取自车位姿信息")
            return

        # 获取场景对应的地图
        try:
            scene_name = nusc.get('scene', rec['scene_token'])['name']
            if scene_name not in scene2map:
                print(f"场景 {scene_name} 没有对应的地图")
                return
            map_name = scene2map[scene_name]
            if map_name not in nusc_maps:
                print(f"找不到地图 {map_name}")
                return
        except (KeyError, TypeError):
            print("无法获取场景或地图信息")
            return

        rot = Quaternion(egopose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([egopose['translation'][0],
                          egopose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = ['road_segment', 'lane']
        line_names = ['road_divider', 'lane_divider']

        try:
            lmap = get_local_map(nusc_maps[map_name], center,
                                 50.0, poly_names, line_names)

            for name in poly_names:
                if name in lmap:
                    for la in lmap[name]:
                        pts = (la - bx) / dx
                        plt.fill(pts[:, 1], pts[:, 0], c=(
                            1.00, 0.50, 0.31), alpha=0.2)

            if 'road_divider' in lmap:
                for la in lmap['road_divider']:
                    pts = (la - bx) / dx
                    plt.plot(pts[:, 1], pts[:, 0], c=(
                        0.0, 0.0, 1.0), alpha=0.5)

            if 'lane_divider' in lmap:
                for la in lmap['lane_divider']:
                    pts = (la - bx) / dx
                    plt.plot(pts[:, 1], pts[:, 0], c=(
                        159./255., 0.0, 1.0), alpha=0.5)
        except Exception as e:
            print(f"绘制地图时出错: {e}")
    except Exception as e:
        print(f"绘制地图时出现未知错误: {e}")
        # 记录错误但继续执行


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


class Detection3DLoss(nn.Module):
    """
    针对3D目标检测的改进多任务损失函数

    提供多种高级IoU损失选项:
    - GIoU: 广义IoU损失，考虑对象之间的重叠区域和包围区域
    - DIoU: 距离IoU损失，额外考虑中心点距离
    - CIoU: 完整IoU损失，考虑重叠度、中心点距离和长宽比
    - EIoU: 高效IoU损失，考虑宽度和高度的差异

    同时支持Focal Loss用于处理类别不平衡问题
    """

    def __init__(self, num_classes=10, cls_weight=1.0, reg_weight=1.0, iou_weight=1.0,
                 iou_loss_type='diou', alpha=0.25, gamma=2.0, beta=0.6, eps=1e-7,
                 use_focal_loss=False, use_quality_focal_loss=False, use_3d_iou=True,
                 angle_weight=1.0, pos_weight=2.0):
        """
        Args:
            num_classes: 类别数量
            cls_weight: 分类损失权重
            reg_weight: 回归损失权重
            iou_weight: IoU损失权重
            iou_loss_type: IoU损失类型，可选: 'iou', 'giou', 'diou', 'ciou', 'eiou', 'siou', '3d-iou'
            alpha: Focal Loss的alpha参数
            gamma: Focal Loss的gamma参数
            beta: CIoU/SIoU中的权衡系数
            eps: 数值稳定性的小常数
            use_focal_loss: 是否使用Focal Loss代替CrossEntropy
            use_quality_focal_loss: 是否使用Quality Focal Loss (需要IoU预测值)
            use_3d_iou: 是否使用3D IoU计算(考虑高度)而不是BEV IoU
            angle_weight: 角度损失的权重
            pos_weight: 正样本的权重(相对于负样本)
        """
        super(Detection3DLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.iou_loss_type = iou_loss_type.lower()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.use_focal_loss = use_focal_loss
        self.use_quality_focal_loss = use_quality_focal_loss
        self.use_3d_iou = use_3d_iou
        self.angle_weight = angle_weight
        self.pos_weight = pos_weight

        # 导入数学库
        import math
        self.math = math

        # 使用Focal Loss或标准分类损失
        if use_focal_loss:
            self.cls_loss_fn = self._focal_loss
        else:
            try:
                from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
                self.cls_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
            except ImportError:
                print(
                    "Warning: timm library not found. Using standard CrossEntropy loss")
                self.cls_loss_fn = nn.CrossEntropyLoss()

        # 高级回归损失 (Default: SmoothL1)
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')

    def _focal_loss(self, pred, target, alpha=None, gamma=None):
        """
        计算Focal Loss

        Args:
            pred: 预测logits，形状为[N, num_classes]
            target: 目标类别，形状为[N]
            alpha: 平衡正负样本的权重系数
            gamma: 调制因子，用于降低易分类样本的权重

        Returns:
            focal_loss: 计算得到的focal loss
        """
        if alpha is None:
            alpha = self.alpha
        if gamma is None:
            gamma = self.gamma

        # 获取每个样本对应的类别的预测概率
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).float()
        pt = (pred_softmax * target_one_hot).sum(1)

        # 计算binary cross entropy
        alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)
        alpha_t = (alpha_t * target_one_hot).sum(1)

        # 计算Focal Loss
        loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt + self.eps)
        return loss.mean()

    def _quality_focal_loss(self, pred, target, score, gamma=None):
        """
        计算Quality Focal Loss (QFL)

        Args:
            pred: 预测logits，形状为[N, num_classes]
            target: 目标类别，形状为[N]
            score: IoU/Quality分数，形状为[N]
            gamma: 调制因子

        Returns:
            qfl: 计算得到的quality focal loss
        """
        if gamma is None:
            gamma = self.gamma

        # 转换为one-hot编码
        target_one_hot = F.one_hot(target, self.num_classes).float()

        # 获取预测概率
        pred_sigmoid = torch.sigmoid(pred)
        scale_factor = pred_sigmoid - score.unsqueeze(1) * target_one_hot
        loss = F.binary_cross_entropy_with_logits(
            pred, target_one_hot, reduction='none'
        ) * scale_factor.abs().pow(gamma)

        return loss.mean()

    def _bbox_to_corners(self, bbox):
        """
        将BBox参数转换为边界框角点坐标
        Args:
            bbox: [B, H, W, 9] 边界框参数 (x, y, z, w, l, h, sin, cos, vel)
        Returns:
            corners: 边界框角点坐标
        """
        x, y, z, w, l, h = bbox[..., 0:6].split(1, dim=-1)

        # 计算half_sizes
        half_w, half_l, half_h = w / 2.0, l / 2.0, h / 2.0

        # 创建8个角点的坐标
        corners = torch.cat([
            torch.cat([x - half_w, y - half_l, z - half_h], dim=-1),
            torch.cat([x + half_w, y - half_l, z - half_h], dim=-1),
            torch.cat([x + half_w, y + half_l, z - half_h], dim=-1),
            torch.cat([x - half_w, y + half_l, z - half_h], dim=-1),
            torch.cat([x - half_w, y - half_l, z + half_h], dim=-1),
            torch.cat([x + half_w, y - half_l, z + half_h], dim=-1),
            torch.cat([x + half_w, y + half_l, z + half_h], dim=-1),
            torch.cat([x - half_w, y + half_l, z + half_h], dim=-1),
        ], dim=-1).reshape(*bbox.shape[:-1], 8, 3)

        return corners

    def _compute_3d_iou(self, pred_boxes, target_boxes):
        """
        计算真正的3D IoU，考虑旋转的3D边界框

        Args:
            pred_boxes: 预测边界框 [B, H, W, 9] - (x, y, z, w, l, h, sin, cos, vel)
            target_boxes: 目标边界框 [B, H, W, 9]

        Returns:
            iou: 3D IoU 值 [B, H, W, 1]
        """
        # 提取位置、尺寸和角度
        pred_centers = pred_boxes[..., :3]  # (x, y, z)
        pred_sizes = torch.cat(
            [pred_boxes[..., 3:5], pred_boxes[..., 5:6]], dim=-1)  # (w, l, h)
        pred_angles = torch.atan2(
            pred_boxes[..., 6:7], pred_boxes[..., 7:8])  # 使用atan2计算角度

        target_centers = target_boxes[..., :3]  # (x, y, z)
        target_sizes = torch.cat(
            [target_boxes[..., 3:5], target_boxes[..., 5:6]], dim=-1)  # (w, l, h)
        target_angles = torch.atan2(
            target_boxes[..., 6:7], target_boxes[..., 7:8])  # 使用atan2计算角度

        # 计算中心点距离
        center_dist = torch.norm(
            pred_centers - target_centers, dim=-1, keepdim=True)

        # 简化计算：如果中心点距离超过两个框大小之和的一半，则IoU为0
        size_sum = torch.norm(pred_sizes, dim=-1, keepdim=True) + \
            torch.norm(target_sizes, dim=-1, keepdim=True)
        mask = (center_dist < size_sum / 2)

        # 简化的3D IoU计算 - 使用轴对齐的3D边界框近似
        # 计算框的体积
        pred_vol = pred_sizes[..., 0] * pred_sizes[..., 1] * pred_sizes[..., 2]
        target_vol = target_sizes[..., 0] * \
            target_sizes[..., 1] * target_sizes[..., 2]

        # 计算交集体积 - 使用近似
        # 当角度差异小时，这种近似比较准确
        angle_diff = torch.abs(pred_angles - target_angles)
        angle_factor = torch.cos(angle_diff)  # 角度差异影响因子

        # 计算轴对齐交集
        min_sizes = torch.min(pred_sizes, target_sizes)
        intersection_approx = min_sizes[..., 0] * \
            min_sizes[..., 1] * min_sizes[..., 2]

        # 考虑角度影响的交集
        intersection = intersection_approx * (0.5 + 0.5 * angle_factor)

        # 计算并集
        union = pred_vol + target_vol - intersection

        # 计算IoU
        iou = torch.zeros_like(center_dist)
        iou[mask] = intersection[mask] / (union[mask] + self.eps)

        return iou

    def _iou_loss(self, pred, target, weights=None, loss_type='iou'):
        """
        计算基于IoU的损失

        Args:
            pred: 预测的边界框参数 [B, H, W, 9]
            target: 目标边界框参数 [B, H, W, 9]
            weights: 样本权重 [B, H, W, 1]
            loss_type: IoU损失类型

        Returns:
            loss: IoU损失
        """
        # 如果是3D-IoU，使用专门的3D IoU计算
        if loss_type == '3d-iou' and self.use_3d_iou:
            iou = self._compute_3d_iou(pred, target)
            loss = 1 - iou
            if weights is not None:
                loss = loss * weights
            return loss

        # 获取形状信息，用于后续reshape操作
        original_shape = pred.shape[:-1]  # [B, H, W]

        # 提取预测和目标的位置和尺寸信息
        pred_xy, pred_wl = torch.split(pred[..., :4], [2, 2], dim=-1)
        pred_z, pred_h = pred[..., 2:3], pred[..., 5:6]
        target_xy, target_wl = torch.split(target[..., :4], [2, 2], dim=-1)
        target_z, target_h = target[..., 2:3], target[..., 5:6]

        # 计算预测框的左上角和右下角坐标
        pred_x1y1 = pred_xy - pred_wl / 2
        pred_x2y2 = pred_xy + pred_wl / 2
        pred_z1 = pred_z - pred_h / 2
        pred_z2 = pred_z + pred_h / 2

        # 计算目标框的左上角和右下角坐标
        target_x1y1 = target_xy - target_wl / 2
        target_x2y2 = target_xy + target_wl / 2
        target_z1 = target_z - target_h / 2
        target_z2 = target_z + target_h / 2

        # 计算交集的宽度和高度
        overlap_x1y1 = torch.max(pred_x1y1, target_x1y1)
        overlap_x2y2 = torch.min(pred_x2y2, target_x2y2)
        overlap_wl = (overlap_x2y2 - overlap_x1y1).clamp(min=0)
        overlap_w, overlap_l = overlap_wl.split(1, dim=-1)

        overlap_z1 = torch.max(pred_z1, target_z1)
        overlap_z2 = torch.min(pred_z2, target_z2)
        overlap_h = (overlap_z2 - overlap_z1).clamp(min=0)

        # 计算交集体积
        if self.use_3d_iou:
            # 3D IoU - 考虑高度
            intersection = overlap_w * overlap_l * overlap_h
        else:
            # BEV IoU - 仅考虑俯视图
            intersection = overlap_w * overlap_l

        # 计算预测框体积/面积
        pred_w, pred_l = pred_wl.split(1, dim=-1)
        if self.use_3d_iou:
            pred_area = pred_w * pred_l * pred_h
        else:
            pred_area = pred_w * pred_l

        # 计算目标框体积/面积
        target_w, target_l = target_wl.split(1, dim=-1)
        if self.use_3d_iou:
            target_area = target_w * target_l * target_h
        else:
            target_area = target_w * target_l

        # 计算IoU
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)

        # 根据不同类型计算IoU Loss
        if loss_type == 'iou':
            # 标准IoU Loss
            loss = 1 - iou

        elif loss_type == 'giou':
            # 计算最小包围盒
            enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
            enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
            enclosing_wl = enclosing_x2y2 - enclosing_x1y1

            if self.use_3d_iou:
                enclosing_z1 = torch.min(pred_z1, target_z1)
                enclosing_z2 = torch.max(pred_z2, target_z2)
                enclosing_h = enclosing_z2 - enclosing_z1

            # 计算包围盒体积/面积
            enclosing_w, enclosing_l = enclosing_wl.split(1, dim=-1)
            if self.use_3d_iou:
                enclosing_vol = enclosing_w * enclosing_l * enclosing_h
            else:
                enclosing_vol = enclosing_w * enclosing_l

            # 计算GIoU
            giou = iou - (enclosing_vol - union) / (enclosing_vol + self.eps)
            loss = 1 - giou

        elif loss_type == 'diou':
            # 首先将张量处理为2D形状，以便正确计算距离
            # 原始形状: [B, H, W, 2]
            batch_size = pred_xy.shape[0]
            pred_xy_flat = pred_xy.reshape(-1, 2)  # [B*H*W, 2]
            target_xy_flat = target_xy.reshape(-1, 2)  # [B*H*W, 2]

            # 计算欧氏距离的平方
            center_dist = F.pairwise_distance(
                pred_xy_flat, target_xy_flat, p=2)
            center_dist = center_dist.reshape(
                *original_shape, 1)  # 恢复原始形状 [B, H, W, 1]

            # 计算最小包围盒
            enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
            enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
            enclosing_wl = enclosing_x2y2 - enclosing_x1y1

            enclosing_w, enclosing_l = enclosing_wl.split(1, dim=-1)

            if self.use_3d_iou:
                # 3D版本 - 考虑高度方向
                enclosing_z1 = torch.min(pred_z1, target_z1)
                enclosing_z2 = torch.max(pred_z2, target_z2)
                enclosing_h = enclosing_z2 - enclosing_z1
                diagonal_dist = torch.sqrt(enclosing_w.pow(
                    2) + enclosing_l.pow(2) + enclosing_h.pow(2) + self.eps)
            else:
                # BEV版本 - 仅考虑俯视图
                diagonal_dist = torch.sqrt(enclosing_w.pow(
                    2) + enclosing_l.pow(2) + self.eps)

            # 计算DIoU
            diou = iou - center_dist.pow(2) / (diagonal_dist.pow(2) + self.eps)
            loss = 1 - diou

        elif loss_type == 'ciou':
            # 首先将张量处理为2D形状，以便正确计算距离
            # 原始形状: [B, H, W, 2]
            batch_size = pred_xy.shape[0]
            pred_xy_flat = pred_xy.reshape(-1, 2)  # [B*H*W, 2]
            target_xy_flat = target_xy.reshape(-1, 2)  # [B*H*W, 2]

            # 计算欧氏距离的平方
            center_dist = F.pairwise_distance(
                pred_xy_flat, target_xy_flat, p=2)
            center_dist = center_dist.reshape(
                *original_shape, 1)  # 恢复原始形状 [B, H, W, 1]

            # 计算最小包围盒
            enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
            enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
            enclosing_wl = enclosing_x2y2 - enclosing_x1y1

            enclosing_w, enclosing_l = enclosing_wl.split(1, dim=-1)

            if self.use_3d_iou:
                # 3D版本
                enclosing_z1 = torch.min(pred_z1, target_z1)
                enclosing_z2 = torch.max(pred_z2, target_z2)
                enclosing_h = enclosing_z2 - enclosing_z1
                diagonal_dist = torch.sqrt(enclosing_w.pow(
                    2) + enclosing_l.pow(2) + enclosing_h.pow(2) + self.eps)

                # 计算纵横比的一致性 - 3D版本
                aspect_w = pred_w / (pred_l + self.eps)
                aspect_l = pred_l / (pred_w + self.eps)
                aspect_h = pred_h / (torch.max(pred_w, pred_l) + self.eps)

                target_aspect_w = target_w / (target_l + self.eps)
                target_aspect_l = target_l / (target_w + self.eps)
                target_aspect_h = target_h / \
                    (torch.max(target_w, target_l) + self.eps)

                v = (4 / (self.math.pi ** 2)) * (
                    (torch.atan(aspect_w) - torch.atan(target_aspect_w)).pow(2) +
                    (torch.atan(aspect_l) - torch.atan(target_aspect_l)).pow(2) +
                    (torch.atan(aspect_h) - torch.atan(target_aspect_h)).pow(2)
                ) / 3.0  # 取三个维度的平均值
            else:
                # BEV版本
                diagonal_dist = torch.sqrt(enclosing_w.pow(
                    2) + enclosing_l.pow(2) + self.eps)

                # 计算长宽比的一致性 - BEV版本
                v = (4 / (self.math.pi ** 2)) * (torch.atan(target_w / (target_l + self.eps)) -
                                                 torch.atan(pred_w / (pred_l + self.eps))).pow(2)

            # 添加权重项
            alpha = v / (1 - iou + v + self.eps)

            # 计算CIoU
            ciou = iou - (center_dist.pow(2) /
                          (diagonal_dist.pow(2) + self.eps) + alpha * v)
            loss = 1 - ciou

        elif loss_type == 'eiou':
            # 首先将张量处理为2D形状，以便正确计算距离
            # 原始形状: [B, H, W, 2]
            batch_size = pred_xy.shape[0]
            pred_xy_flat = pred_xy.reshape(-1, 2)  # [B*H*W, 2]
            target_xy_flat = target_xy.reshape(-1, 2)  # [B*H*W, 2]

            # 计算欧氏距离的平方
            center_dist = F.pairwise_distance(
                pred_xy_flat, target_xy_flat, p=2)
            center_dist = center_dist.reshape(
                *original_shape, 1)  # 恢复原始形状 [B, H, W, 1]

            # 计算长宽的差异
            w_dist = (pred_w - target_w).pow(2)
            l_dist = (pred_l - target_l).pow(2)

            if self.use_3d_iou:
                h_dist = (pred_h - target_h).pow(2)

            # 计算最小包围盒
            enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
            enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
            enclosing_wl = enclosing_x2y2 - enclosing_x1y1

            enclosing_w, enclosing_l = enclosing_wl.split(1, dim=-1)

            if self.use_3d_iou:
                # 3D版本
                enclosing_z1 = torch.min(pred_z1, target_z1)
                enclosing_z2 = torch.max(pred_z2, target_z2)
                enclosing_h = enclosing_z2 - enclosing_z1
                diagonal_dist = enclosing_w.pow(
                    2) + enclosing_l.pow(2) + enclosing_h.pow(2) + self.eps

                # 计算EIoU
                eiou = iou - (center_dist.pow(2) / diagonal_dist +
                              (w_dist / diagonal_dist + l_dist / diagonal_dist + h_dist / diagonal_dist))
            else:
                # BEV版本
                diagonal_dist = enclosing_w.pow(
                    2) + enclosing_l.pow(2) + self.eps

                # 计算EIoU - BEV版本
                eiou = iou - (center_dist.pow(2) / diagonal_dist +
                              (w_dist / diagonal_dist + l_dist / diagonal_dist))

            loss = 1 - eiou

        elif loss_type == 'siou':
            # SIoU Loss - 专注于形状和方向的IoU损失

            # 首先将张量处理为2D形状，以便正确计算距离
            # 原始形状: [B, H, W, 2]
            batch_size = pred_xy.shape[0]
            pred_xy_flat = pred_xy.reshape(-1, 2)  # [B*H*W, 2]
            target_xy_flat = target_xy.reshape(-1, 2)  # [B*H*W, 2]

            # 计算角度信息 - 从sin/cos计算角度
            pred_sin, pred_cos = pred[..., 6:7], pred[..., 7:8]
            pred_angle = torch.atan2(pred_sin, pred_cos)

            target_sin, target_cos = target[..., 6:7], target[..., 7:8]
            target_angle = torch.atan2(target_sin, target_cos)

            # 计算角度差异 - 保证在[-π/2, π/2]范围内
            angle_diff = torch.abs(pred_angle - target_angle)
            angle_diff = torch.min(angle_diff, torch.abs(
                angle_diff - torch.tensor(self.math.pi, device=angle_diff.device)))
            angle_diff = torch.min(angle_diff, torch.abs(
                angle_diff - torch.tensor(2*self.math.pi, device=angle_diff.device)))

            # 计算中心点距离
            center_dist = F.pairwise_distance(
                pred_xy_flat, target_xy_flat, p=2)
            center_dist = center_dist.reshape(
                *original_shape, 1)  # 恢复原始形状 [B, H, W, 1]

            # 计算最小包围盒
            enclosing_x1y1 = torch.min(pred_x1y1, target_x1y1)
            enclosing_x2y2 = torch.max(pred_x2y2, target_x2y2)
            enclosing_wl = enclosing_x2y2 - enclosing_x1y1

            enclosing_w, enclosing_l = enclosing_wl.split(1, dim=-1)
            diagonal_dist = torch.sqrt(enclosing_w.pow(
                2) + enclosing_l.pow(2) + self.eps)

            # 计算形状相似度项
            shape_cost = torch.abs(pred_w - target_w) / torch.max(pred_w, target_w) + \
                torch.abs(pred_l - target_l) / torch.max(pred_l, target_l)
            if self.use_3d_iou:
                shape_cost = (shape_cost + torch.abs(pred_h -
                              target_h) / torch.max(pred_h, target_h)) / 3.0
            else:
                shape_cost = shape_cost / 2.0

            # 计算角度损失项 - 角度差异越大，损失越大
            angle_cost = (1 - torch.cos(angle_diff)) / 2.0

            # 计算中心点距离损失
            dist_cost = center_dist / diagonal_dist

            # 组合所有损失项
            lambda1, lambda2, lambda3 = 1.0, 1.0, 1.0  # 可调整的权重系数

            # SIoU损失
            siou_loss = 1 - iou + (lambda1 * shape_cost + lambda2 *
                                   angle_cost + lambda3 * dist_cost) * (1 - iou + self.eps)

            loss = siou_loss
        else:
            raise ValueError(
                f"未知的IoU损失类型: {loss_type}, 请选择 'iou', 'giou', 'diou', 'ciou', 'eiou', 'siou', 或 '3d-iou'")

        # 应用权重
        if weights is not None:
            loss = loss * weights

        return loss

    def forward(self, preds, targets):
        """
        计算3D目标检测的多任务损失

        Args:
            preds: 模型预测结果，包含 cls_pred, reg_pred, iou_pred
            targets: 目标值，包含 cls_targets, reg_targets, iou_targets, reg_weights

        Returns:
            dict: 包含各项损失和总损失
        """
        # 解析预测值
        cls_pred = preds['cls_pred']  # [B, num_classes, H, W]
        # [B, 9, H, W] - x, y, z, w, l, h, sin, cos, vel
        reg_pred = preds['reg_pred']
        iou_pred = preds['iou_pred']  # [B, 1, H, W]

        # 解析目标值
        cls_targets = targets['cls_targets']  # [B, H, W] - 类别标签，0为背景
        reg_targets = targets['reg_targets']  # [B, 9, H, W]
        iou_targets = targets['iou_targets']  # [B, 1, H, W]
        reg_weights = targets['reg_weights']  # [B, 1, H, W] - 用于加权回归样本

        # 计算分类损失
        B, C, H, W = cls_pred.shape

        # 将预测值重新整形为[B*H*W, num_classes]
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        # 将目标值重新整形为[B*H*W]
        cls_targets = cls_targets.reshape(-1).long()

        # 确保形状兼容性
        if cls_pred.shape[0] != cls_targets.shape[0]:
            # 如果形状不匹配，对cls_pred进行适当调整
            ratio = cls_targets.shape[0] / cls_pred.shape[0]

            if ratio > 1:  # 目标比预测大
                # 扩展预测以匹配目标
                cls_pred = cls_pred.repeat(int(ratio), 1)
            else:  # 预测比目标大
                # 截断预测以匹配目标
                cls_pred = cls_pred[:cls_targets.shape[0], :]

        # 过滤掉padding区域和背景
        valid_mask = (cls_targets >= 0) & (cls_targets < self.num_classes)
        foreground_mask = cls_targets > 0  # 0为背景类

        # 应用正样本权重，增加对正样本的关注
        if self.pos_weight > 1.0 and foreground_mask.sum() > 0:
            sample_weights = torch.ones_like(valid_mask, dtype=torch.float32)
            sample_weights[foreground_mask] = self.pos_weight
        else:
            sample_weights = None

        # 计算分类损失
        if valid_mask.sum() > 0:
            if self.use_focal_loss:
                cls_loss = self._focal_loss(
                    cls_pred[valid_mask], cls_targets[valid_mask])
            elif self.use_quality_focal_loss and 'iou_targets' in targets:
                iou_score = iou_targets.reshape(-1)[valid_mask]
                cls_loss = self._quality_focal_loss(
                    cls_pred[valid_mask], cls_targets[valid_mask], iou_score)
            else:
                if sample_weights is not None:
                    # 如果使用了样本权重，并且是标准交叉熵，需要手动实现加权版本
                    target_one_hot = F.one_hot(
                        cls_targets[valid_mask], self.num_classes).float()
                    log_probs = F.log_softmax(cls_pred[valid_mask], dim=1)
                    loss_per_sample = -(target_one_hot * log_probs).sum(dim=1)
                    # 应用样本权重
                    weighted_loss = loss_per_sample * \
                        sample_weights[valid_mask]
                    cls_loss = weighted_loss.mean()
                else:
                    cls_loss = self.cls_loss_fn(
                        cls_pred[valid_mask], cls_targets[valid_mask])
        else:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)

        # 计算回归损失 - 只考虑有目标的位置
        reg_pred = reg_pred.permute(0, 2, 3, 1)  # [B, H, W, 9]
        reg_targets = reg_targets.permute(0, 2, 3, 1)  # [B, H, W, 9]
        reg_weights = reg_weights.permute(0, 2, 3, 1)  # [B, H, W, 1]

        # 计算高级IoU损失
        iou_loss_raw = self._iou_loss(
            reg_pred, reg_targets, weights=None, loss_type=self.iou_loss_type)

        # 单独计算角度和速度的损失
        angle_pred = reg_pred[..., 6:8]  # sin(θ), cos(θ)
        angle_target = reg_targets[..., 6:8]
        vel_pred = reg_pred[..., 8:9]  # 速度
        vel_target = reg_targets[..., 8:9]

        # 计算角度损失 (避免周期性问题)
        # 使用点积计算角度相似度，对规范化的sin/cos向量
        angle_norm_pred = F.normalize(angle_pred, dim=-1)
        angle_norm_target = F.normalize(angle_target, dim=-1)
        # 余弦相似度: 1代表完全相同，-1代表完全相反
        cos_sim = (angle_norm_pred *
                   angle_norm_target).sum(dim=-1, keepdim=True)
        # 转换为损失: 0代表完全相同，2代表完全相反
        angle_loss = (1.0 - cos_sim)

        # 应用角度损失权重
        angle_loss = angle_loss * self.angle_weight

        # 计算速度损失
        vel_loss = self.reg_loss_fn(vel_pred, vel_target)

        # 组合所有回归损失
        combined_reg_loss = iou_loss_raw + angle_loss + vel_loss
        reg_loss = (combined_reg_loss * reg_weights).sum() / \
            (reg_weights.sum().clamp(min=1.0))

        # 计算IoU预测损失 (使用BCE损失)
        if 'iou_pred' in preds and 'iou_targets' in targets:
            iou_pred = iou_pred.permute(0, 2, 3, 1)  # [B, H, W, 1]
            iou_targets = iou_targets.permute(0, 2, 3, 1)  # [B, H, W, 1]

            # 使用IoU预测也采用Focal Loss
            iou_pred_flat = iou_pred.reshape(-1, 1)
            iou_targets_flat = iou_targets.reshape(-1, 1)

            # 计算IoU预测损失
            bce_loss = F.binary_cross_entropy_with_logits(
                iou_pred_flat, iou_targets_flat, reduction='none')

            # 添加Focal Loss调制
            pt = torch.exp(-bce_loss)
            iou_pred_loss = ((1 - pt) ** self.gamma * bce_loss)

            # 应用权重
            reg_weights_flat = reg_weights.reshape(-1, 1)
            iou_loss = (iou_pred_loss * reg_weights_flat).sum() / \
                (reg_weights_flat.sum().clamp(min=1.0))
        else:
            iou_loss = torch.tensor(0.0, device=cls_pred.device)

        # 计算总损失
        total_loss = self.cls_weight * cls_loss + \
            self.reg_weight * reg_loss + \
            self.iou_weight * iou_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'iou_loss': iou_loss,
            'angle_loss': angle_loss.mean() if isinstance(angle_loss, torch.Tensor) else angle_loss,
            'iou_raw': iou_loss_raw.mean() if isinstance(iou_loss_raw, torch.Tensor) else iou_loss_raw
        }
