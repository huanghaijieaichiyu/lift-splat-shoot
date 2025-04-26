"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import math
from typing import Dict, Optional
from torch.utils.tensorboard.writer import SummaryWriter
from torch import nn
import time
from contextlib import nullcontext
from torch.cuda.amp import autocast
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
# Import Shapely for accurate rotated IoU calculation
try:
    import shapely.geometry as sg
    import shapely.ops as so
    SHAPELY_AVAILABLE = True
except ImportError:
    print("Warning: Shapely library not found. Rotated IoU calculation will use a placeholder.")
    print("Install with: pip install shapely")
    SHAPELY_AVAILABLE = False


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


# Helper function to convert box parameters to rotated corners
def corners_from_center_dims_yaw(centers: torch.Tensor, dims: torch.Tensor, yaws: torch.Tensor) -> torch.Tensor:
    """
    Converts BEV box parameters (center, dimensions, yaw) to corner coordinates.

    Args:
        centers: Box centers [N, 2] (x, y).
        dims: Box dimensions [N, 2] (length, width). Assumes l corresponds to yaw direction.
        yaws: Box yaw angles [N] (in radians).

    Returns:
        corners: Box corners [N, 4, 2] in counter-clockwise order.
                 (Front-Left, Rear-Left, Rear-Right, Front-Right)
    """
    batch_size = centers.shape[0]
    device = centers.device
    len_half, width_half = dims[:, 0:1] / 2, dims[:, 1:2] / 2

    # Base corners relative to center (0, 0)
    base_corners = torch.tensor([
        [1,  1],  # Front-Right
        [-1,  1],  # Rear-Right
        [-1, -1],  # Rear-Left
        [1, -1]  # Front-Left
        # [N, 4, 2]
    ], dtype=centers.dtype, device=device) * torch.cat([len_half, width_half], dim=1).unsqueeze(1)

    # Rotation matrix
    cos_yaw = torch.cos(yaws)  # [N]
    sin_yaw = torch.sin(yaws)  # [N]
    # Note: This is rotation for the coordinate system,
    # to rotate points, we use the transpose.
    # Rotation matrix for points: [[cos, -sin], [sin, cos]]
    rot_matrix = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=-1),
        torch.stack([sin_yaw, cos_yaw], dim=-1)
    ], dim=1)  # [N, 2, 2]

    # Rotate corners and add center offset
    # [N, 4, 2] @ [N, 2, 2] -> [N, 4, 2] (batch-wise matmul requires compatible shapes or broadcasting)
    # Using einsum for clarity on batch matrix multiplication:
    rotated_corners = torch.einsum(
        'nij,nlj->nil', base_corners, rot_matrix)  # Output: [N, 4, 2]

    corners = rotated_corners + centers.unsqueeze(1)  # [N, 4, 2]

    # Reorder corners to expected sequence (e.g., front-left, rear-left, rear-right, front-right)
    # Current order from base_corners * transform is likely FR, RR, RL, FL (Check calculation)
    # Let's explicitly define order: FL(idx 3), RL(idx 2), RR(idx 1), FR(idx 0)
    corners_reordered = corners[:, [3, 2, 1, 0], :]

    return corners_reordered  # [N, 4, 2]


# Helper function for rotated box IoU (requires shapely or custom implementation)
# Using a placeholder for now, as a robust implementation is complex.
# A real implementation would use polygon intersection algorithms.
def rotated_iou_bev(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """
    Calculates Batched IoU of rotated 2D boxes using Shapely.
    A robust implementation requires polygon intersection (e.g., using shapely
    or a custom CUDA kernel). This version is a very rough approximation/placeholder.
    """
    if not SHAPELY_AVAILABLE:
        # Fallback to placeholder if shapely is not installed
        print(
            "Warning: Shapely not found, using placeholder IoU! Results will be inaccurate.")
        box1_min, _ = torch.min(corners1, dim=1)
        box1_max, _ = torch.max(corners1, dim=1)
        box2_min, _ = torch.min(corners2, dim=1)
        box2_max, _ = torch.max(corners2, dim=1)
        inter_min = torch.max(box1_min, box2_min)
        inter_max = torch.min(box1_max, box2_max)
        inter_dims = (inter_max - inter_min).clamp(min=0)
        inter_area = inter_dims[:, 0] * inter_dims[:, 1]
        area1 = (box1_max - box1_min)[:, 0] * (box1_max - box1_min)[:, 1]
        area2 = (box2_max - box2_min)[:, 0] * (box2_max - box2_min)[:, 1]
        union = area1 + area2 - inter_area
        iou = inter_area / union.clamp(min=1e-7)
        iou[union <= 0] = 0
        return iou

    # Shapely operates on CPU, convert tensors
    corners1_np = corners1.detach().cpu().numpy()
    corners2_np = corners2.detach().cpu().numpy()
    num_boxes = corners1.shape[0]
    ious = torch.zeros(num_boxes, dtype=corners1.dtype,
                       device='cpu')  # Store results on CPU first

    for i in range(num_boxes):
        try:
            poly1 = sg.Polygon(corners1_np[i])
            poly2 = sg.Polygon(corners2_np[i])

            if not poly1.is_valid or not poly2.is_valid:
                # print(f"Warning: Invalid polygon created for index {i}. Skipping IoU.")
                continue  # Leave IoU as 0

            intersection_area = poly1.intersection(poly2).area
            area1 = poly1.area
            area2 = poly2.area
            union_area = area1 + area2 - intersection_area

            if union_area > 1e-7:  # Avoid division by zero
                ious[i] = intersection_area / union_area

        except Exception as e:
            # Handle potential errors during polygon creation or intersection
            print(f"Shapely error at index {i}: {e}")
            # print(f" Corners1[{i}]: {corners1_np[i]}")
            # print(f" Corners2[{i}]: {corners2_np[i]}")
            continue  # Leave IoU as 0

    # Move results back to the original device
    return ious.to(corners1.device)


# --- 新增: 通用的基于 IoU 的损失函数 ---
def rotated_iou_based_loss(
    corners_pred: torch.Tensor, corners_target: torch.Tensor,
    centers_pred: torch.Tensor, centers_target: torch.Tensor,
    dims_pred: torch.Tensor, dims_target: torch.Tensor,  # CIoU 需要尺寸信息 (l, w)
    mode: str = 'diou',  # 'diou' 或 'ciou'
    eps: float = 1e-7
) -> torch.Tensor:
    """
    计算旋转框的 IoU-based 损失 (DIoU 或 CIoU)。
    使用精确的旋转 IoU (如果 Shapely 可用) 和轴对齐的包围框对角线近似。

    Args:
        corners_pred: 预测框角点 [N, 4, 2]。
        corners_target: 目标框角点 [N, 4, 2]。
        centers_pred: 预测框中心 [N, 2]。
        centers_target: 目标框中心 [N, 2]。
        dims_pred: 预测框尺寸 [N, 2] (l, w)。
        dims_target: 目标框尺寸 [N, 2] (l, w)。
        mode: 损失类型, 'diou' 或 'ciou'。
        eps: 用于数值稳定性的小值。

    Returns:
        loss: 每个框对的损失值 [N]。
    """
    # 使用改进后的函数计算 IoU
    iou = rotated_iou_bev(corners_pred, corners_target)

    # --- 计算 DIoU 惩罚项 (中心点距离) ---
    d2 = ((centers_pred - centers_target) ** 2).sum(dim=-1)  # [N]

    # --- 计算最小包围框的对角线平方 c^2 (使用轴对齐近似) ---
    all_corners = torch.cat([corners_pred, corners_target], dim=1)  # [N, 8, 2]
    min_coords, _ = torch.min(all_corners, dim=1)  # [N, 2]
    max_coords, _ = torch.max(all_corners, dim=1)  # [N, 2]
    c2 = ((max_coords - min_coords) ** 2).sum(dim=-
                                              # [N], clamp 防止除零
                                              1).clamp(min=eps)

    # 计算基础的 DIoU 损失部分: 1 - IoU + d^2/c^2
    diou_term = d2 / c2
    loss = 1.0 - iou + diou_term

    # --- 如果是 CIoU 模式, 添加长宽比惩罚项 ---
    if mode.lower() == 'ciou':
        # 计算长宽比惩罚项 v
        # dims are (l, w), so aspect ratio is w/l
        arctan_pred = torch.atan(
            dims_pred[:, 1] / dims_pred[:, 0].clamp(min=eps))
        arctan_target = torch.atan(
            dims_target[:, 1] / dims_target[:, 0].clamp(min=eps))
        v = (4 / (np.pi ** 2)) * \
            torch.pow(arctan_pred - arctan_target, 2)  # [N]

        # 计算权重因子 alpha (常见实现方式)
        # 使用 detach 防止 iou 影响 v 的梯度
        # 当 IoU 较高时，降低长宽比惩罚的权重 (alpha -> 0)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)  # [N]
            # alpha[iou <= 0.5] = 0 # 另一种常见做法：只在 IoU > 0.5 时应用 alpha*v

        # 添加 CIoU 的长宽比惩罚项
        ciou_term = alpha * v
        loss = loss + ciou_term

    elif mode.lower() != 'diou':
        raise ValueError(f"不支持的 IoU 损失模式: {mode}. 请选择 'diou' 或 'ciou'")

    return loss


class DetectionBEVLoss(nn.Module):
    """
    BEV 检测损失，计算各分量的原始损失值。
    权重将在训练循环中通过 DWA 动态计算并应用。

    计算以下原始损失:
    - 分类 (Focal Loss)
    - BEV 框回归 (Rotated DIoU/CIoU Loss)
    - Z 坐标回归 (Smooth L1)
    - 高度回归 (Smooth L1)
    - 速度回归 (Smooth L1)
    - IoU 预测头 (BCE)。

    返回各分量原始损失的字典。
    """

    def __init__(self, num_classes: int,
                 bev_loss_type: str = 'ciou',  # 'diou' 或 'ciou'
                 alpha: float = 0.25, gamma: float = 2.0, beta: float = 1.0,  # beta for SmoothL1
                 eps: float = 1e-7):
        """
        Args:
            num_classes: 前景目标类别数。
            bev_loss_type: BEV 回归使用的损失类型 ('diou' 或 'ciou')。
            alpha: Focal Loss 的 alpha 参数。
            gamma: Focal Loss 的 gamma 参数。
            beta: Smooth L1 Loss 的 beta 参数 (delta)。
            eps: 用于数值稳定性的小 epsilon 值。
        """
        super().__init__()
        if bev_loss_type.lower() not in ['diou', 'ciou']:
            raise ValueError(
                f"不支持的 bev_loss_type: {bev_loss_type}. 请选择 'diou' 或 'ciou'")
        self.bev_loss_type = bev_loss_type.lower()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

        self.smooth_l1_loss = nn.SmoothL1Loss(beta=self.beta, reduction='none')
        self.iou_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    # _focal_loss 方法保持不变
    def _focal_loss(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        target_labels = target_labels.long()
        pred_softmax = F.softmax(pred_logits, dim=1)
        pt = pred_softmax.gather(1, target_labels.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, self.eps, 1.0 - self.eps)
        alpha_t = torch.where(target_labels > 0, self.alpha,
                              1 - self.alpha).to(pred_logits.device)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return loss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算各分量的原始（未加权）检测损失。

        Args:
            preds: 模型预测字典...
            targets: 真值目标字典...

        Returns:
            包含各分量原始损失的字典:
            cls_loss', 'bev_loss', 'z_loss', 'h_loss',
            'vel_loss', 'iou_loss'。
        """
        # --- Input parsing and reshaping (remains the same) ---
        cls_pred = preds['cls_pred']
        reg_pred = preds['reg_pred']
        iou_pred = preds.get('iou_pred', None)
        cls_targets = targets['cls_targets']
        reg_targets = targets['reg_targets']
        reg_weights = targets['reg_weights']
        iou_targets = targets.get('iou_targets', None)

        B, _, H, W = cls_pred.shape
        expected_reg_dim = 9
        actual_reg_dim = reg_pred.shape[1]
        if actual_reg_dim < expected_reg_dim:
            print(
                f"Warning: Regression prediction dim ({actual_reg_dim}) < expected ({expected_reg_dim}). Padding with zeros.")
            padding = torch.zeros(B, expected_reg_dim -
                                  actual_reg_dim, H, W, device=reg_pred.device)
            reg_pred = torch.cat([reg_pred, padding], dim=1)
            target_reg_dim = reg_targets.shape[1]
            if target_reg_dim < expected_reg_dim:
                print(
                    f"Warning: Regression target dim ({target_reg_dim}) < expected ({expected_reg_dim}). Padding with zeros.")
                padding_target = torch.zeros(
                    B, expected_reg_dim - target_reg_dim, H, W, device=reg_targets.device)
                reg_targets = torch.cat([reg_targets, padding_target], dim=1)

        num_predictions = B * H * W
        cls_pred_flat = cls_pred.permute(0, 2, 3, 1).reshape(
            num_predictions, self.num_classes)
        reg_pred_flat = reg_pred.permute(
            0, 2, 3, 1).reshape(num_predictions, expected_reg_dim)

        if iou_pred is not None:
            iou_pred_flat = iou_pred.permute(
                0, 2, 3, 1).reshape(num_predictions, 1)
        else:
            iou_pred_flat = None

        cls_targets_flat = cls_targets.reshape(num_predictions)
        reg_targets_flat = reg_targets.permute(
            0, 2, 3, 1).reshape(num_predictions, expected_reg_dim)
        reg_weights_flat = reg_weights.reshape(num_predictions)

        if iou_targets is not None:
            iou_targets_flat = iou_targets.reshape(num_predictions, 1)
        else:
            iou_targets_flat = None

        # --- Masks (remains the same) ---
        pos_mask = reg_weights_flat > 0
        num_pos = pos_mask.sum().clamp(min=1.0)

        # --- Raw Classification Loss (Focal Loss) ---
        valid_cls_mask = cls_targets_flat >= 0
        if valid_cls_mask.sum() == 0:
            raw_cls_loss = torch.tensor(0.0, device=cls_pred.device)
        else:
            loss_cls_all = self._focal_loss(
                cls_pred_flat[valid_cls_mask], cls_targets_flat[valid_cls_mask])
            raw_cls_loss = loss_cls_all.sum() / num_pos

        # --- Raw Regression Losses (calculated only on positive samples) ---
        raw_bev_loss = torch.tensor(0.0, device=reg_pred.device)
        raw_z_loss = torch.tensor(0.0, device=reg_pred.device)
        raw_h_loss = torch.tensor(0.0, device=reg_pred.device)
        raw_vel_loss = torch.tensor(0.0, device=reg_pred.device)
        raw_iou_loss = torch.tensor(0.0, device=reg_pred.device)

        if pos_mask.sum() > 0:
            reg_pred_pos = reg_pred_flat[pos_mask]
            reg_targets_pos = reg_targets_flat[pos_mask]

            # --- Raw BEV Loss (DIoU or CIoU) ---
            centers_pred = reg_pred_pos[:, :2]
            dims_pred_wl = reg_pred_pos[:, 3:5] if reg_pred_pos.shape[1] >= 5 else torch.zeros_like(
                centers_pred)
            dims_pred = dims_pred_wl[:, [1, 0]]
            sin_cos_pred = reg_pred_pos[:, 6:8] if reg_pred_pos.shape[1] >= 8 else torch.zeros_like(
                centers_pred)
            yaw_pred = torch.atan2(sin_cos_pred[:, 0], sin_cos_pred[:, 1])
            centers_target = reg_targets_pos[:, :2]
            dims_target_wl = reg_targets_pos[:, 3:5] if reg_targets_pos.shape[1] >= 5 else torch.zeros_like(
                centers_target)
            dims_target = dims_target_wl[:, [1, 0]]
            sin_cos_target = reg_targets_pos[:, 6:8] if reg_targets_pos.shape[1] >= 8 else torch.zeros_like(
                centers_target)
            yaw_target = torch.atan2(
                sin_cos_target[:, 0], sin_cos_target[:, 1])
            corners_pred = corners_from_center_dims_yaw(
                centers_pred, dims_pred, yaw_pred)
            corners_target = corners_from_center_dims_yaw(
                centers_target, dims_target, yaw_target)
            loss_bev_all = rotated_iou_based_loss(
                corners_pred, corners_target, centers_pred, centers_target, dims_pred, dims_target, mode=self.bev_loss_type, eps=self.eps)
            raw_bev_loss = loss_bev_all.sum() / num_pos

            # --- Raw Z Loss (Smooth L1) ---
            z_pred = reg_pred_pos[:, 2:3] if reg_pred_pos.shape[1] >= 3 else torch.zeros_like(
                reg_pred_pos[:, :1])
            z_target = reg_targets_pos[:, 2:3] if reg_targets_pos.shape[1] >= 3 else torch.zeros_like(
                reg_targets_pos[:, :1])
            loss_z_all = self.smooth_l1_loss(z_pred, z_target)
            raw_z_loss = loss_z_all.sum() / num_pos

            # --- Raw Height Loss (Smooth L1) ---
            h_pred = reg_pred_pos[:, 5:6] if reg_pred_pos.shape[1] >= 6 else torch.zeros_like(
                reg_pred_pos[:, :1])
            h_target = reg_targets_pos[:, 5:6] if reg_targets_pos.shape[1] >= 6 else torch.zeros_like(
                reg_targets_pos[:, :1])
            loss_h_all = self.smooth_l1_loss(h_pred, h_target)
            raw_h_loss = loss_h_all.sum() / num_pos

            # --- Raw Velocity Loss (Smooth L1) ---
            vel_dim = 2
            vel_pred = reg_pred_pos[:, 8:8+vel_dim] if reg_pred_pos.shape[1] >= 8 + vel_dim \
                else torch.zeros((reg_pred_pos.shape[0], vel_dim), device=reg_pred_pos.device)
            vel_target = reg_targets_pos[:, 8:8+vel_dim] if reg_targets_pos.shape[1] >= 8+vel_dim else torch.zeros(
                (reg_targets_pos.shape[0], vel_dim), device=reg_targets_pos.device)
            if vel_pred.shape[1] == vel_target.shape[1] and vel_pred.shape[1] > 0:
                loss_vel_all = self.smooth_l1_loss(vel_pred, vel_target)
                raw_vel_loss = loss_vel_all.sum() / num_pos

            # --- Raw IoU Prediction Loss (BCE) ---
            if iou_pred_flat is not None and iou_targets_flat is not None:
                iou_pred_pos = iou_pred_flat[pos_mask]
                iou_targets_pos = iou_targets_flat[pos_mask]
                loss_iou_pred_all = self.iou_loss_fn(
                    iou_pred_pos, iou_targets_pos)
                raw_iou_loss = loss_iou_pred_all.sum() / num_pos

        # --- 返回只包含原始损失的字典 ---
        raw_losses = {
            'raw_cls_loss': raw_cls_loss,
            'raw_bev_loss': raw_bev_loss,
            'raw_z_loss': raw_z_loss,
            'raw_h_loss': raw_h_loss,
            'raw_vel_loss': raw_vel_loss,
            'raw_iou_loss': raw_iou_loss,
        }

        # 如果没有iou头，iou损失为0，但仍然包含在字典中
        # raw_losses['raw_iou_loss'] = raw_iou_loss

        return raw_losses  # 只返回原始损失


# === MODIFICATION START: Replace _neg_loss and FocalLoss with a standard implementation ===
# Original _neg_loss function (commented out or removed)
# def _neg_loss(pred, gt): ...

# Standard Focal Loss computation function (works with logits)
# Returns unnormalized, per-pixel loss
def compute_focal_loss_with_logits(pred_logits, gt_heatmap, alpha=0.25, gamma=2.0, epsilon=1e-12):
    """
    Computes focal loss from logits and a ground truth heatmap (e.g., Gaussian).
    Args:
        pred_logits: Model predictions (logits) [B, C, H, W]
        gt_heatmap: Ground truth heatmap [B, C, H, W] (values usually 0-1)
        alpha: Alpha parameter for focal loss.
        gamma: Gamma parameter for focal loss.
        epsilon: Small value for numerical stability.
    Returns:
        torch.Tensor: Per-pixel focal loss, unnormalized [B, C, H, W].
    """
    # Use BCEWithLogitsLoss for numerical stability with logits
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_logits, gt_heatmap, reduction='none')

    # Calculate pt (probability of predicting the ground truth class)
    # For Gaussian heatmap, pt is sigmoid(logits) where gt is near 1, and 1-sigmoid(logits) where gt is near 0.
    pred_prob = torch.sigmoid(pred_logits)
    # Use gt_heatmap directly as the target probability for pt calculation where gt is positive
    p_t = pred_prob * gt_heatmap + (1 - pred_prob) * (1 - gt_heatmap)
    # Clamp p_t for stability with log
    p_t = torch.clamp(p_t, epsilon, 1.0 - epsilon)

    # Calculate the focal loss modulating factor (1-pt)^gamma
    modulating_factor = torch.pow(1.0 - p_t, gamma)

    # Calculate the alpha factor
    # Use 0.5 threshold for alpha application
    alpha_factor = torch.where(gt_heatmap >= 0.5, alpha, 1.0 - alpha)

    # Compute the final focal loss
    focal_loss = alpha_factor * modulating_factor * bce_loss

    return focal_loss  # Return unnormalized loss per pixel


class FocalLoss(nn.Module):
    '''nn.Module wrapper for focal loss'''

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # Store compute function internally
        self._compute = compute_focal_loss_with_logits

    def forward(self, pred_logits, gt_heatmap):
        """ Returns the sum of focal loss over all pixels. Normalization should be done outside. """
        # Calculate per-pixel loss using the compute function
        loss_per_pixel = self._compute(
            pred_logits, gt_heatmap, self.alpha, self.gamma)
        # Return the sum of losses, normalization happens in CenterPointLoss
        return loss_per_pixel.sum()
# === MODIFICATION END ===


def _gather_feat(feat, ind, mask=None):
    """Gather feature based on index - used for regression losses"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLoss(nn.Module):
    """Regression loss for offsets, depth, dimension, orientation.
    Args:
        loss_type (str): One of ['L1', 'SmoothL1']. Default 'L1'.
    """

    def __init__(self, loss_type='L1'):
        super(RegLoss, self).__init__()
        if loss_type == 'L1':
            self.crit = nn.L1Loss(reduction='none')
        elif loss_type == 'SmoothL1':
            self.crit = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(
                f"Unsupported regression loss type: {loss_type}")

    def forward(self, output, target, ind, mask):
        """ Forward pass.
        Args:
            output (torch.Tensor): Prediction tensor [B, C, H, W].
            target (torch.Tensor): Target tensor [B, max_objs, C].
            ind (torch.Tensor): Target indices [B, max_objs].
            mask (torch.Tensor): Target mask [B, max_objs], indicating valid objects.
        Returns:
            torch.Tensor: Regression loss scalar.
            torch.Tensor: Masked loss tensor [B, max_objs, C] (for debugging/analysis).
        """
        pred = _transpose_and_gather_feat(output, ind)  # [B, max_objs, C]
        # Ensure target and pred have compatible shapes
        if pred.shape != target.shape:
            # This could happen if C differs or padding logic is wrong
            # Attempt to reshape target if necessary and possible
            # Example: target might be [B, max_objs] needs [B, max_objs, 1]
            if pred.dim() == 3 and target.dim() == 2 and pred.shape[2] == 1:
                target = target.unsqueeze(-1)
            else:
                raise ValueError(
                    f"Shape mismatch in RegLoss: pred {pred.shape}, target {target.shape}")

        loss = self.crit(pred, target)  # [B, max_objs, C]

        # --- MODIFICATION START: Use mask for normalization ---
        # mask has shape [B, max_objs]
        # Expand mask to match loss shape [B, max_objs, C]
        mask_expanded = mask.unsqueeze(2).expand_as(loss).float()

        # Calculate loss only on valid items, average over valid items
        masked_loss = loss * mask_expanded
        num_valid = mask_expanded.sum()

        if num_valid > 0:
            total_loss = masked_loss.sum() / num_valid
        else:
            total_loss = masked_loss.sum()  # Avoid division by zero, loss is 0 anyway
        # --- MODIFICATION END ---

        # Return total scalar loss and the masked loss tensor per element
        return total_loss, masked_loss


class CenterPointLoss(nn.Module):
    """ Loss for CenterPoint model.
    Includes Focal Loss for heatmap and L1 loss for regression targets.
    """

    def __init__(self, num_classes=10, loss_weights=None, focal_loss_alpha=0.25, focal_loss_gamma=2.0):
        super(CenterPointLoss, self).__init__()
        self.num_classes = num_classes
        # Default weights if none provided
        default_weights = {
            'heatmap': 1.0,
            'offset': 1.0,
            'z_coord': 1.0,
            'dimension': 1.0,
            'rotation': 1.0,
            'velocity': 1.0
        }
        self.loss_weights = loss_weights if loss_weights is not None else default_weights

        # === MODIFICATION START: Use revised FocalLoss ===
        self.crit_cls = FocalLoss(
            alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        # === MODIFICATION END ===
        self.crit_reg = RegLoss(loss_type='L1')  # L1 loss for regression

    def forward(self, outputs, targets):
        """ Forward pass.
        Args:
            outputs (dict): Dictionary of prediction tensors from each head.
                Keys: 'heatmap', 'offset', 'z_coord', 'dimension', 'rotation', 'velocity'.
            targets (dict): Dictionary of target tensors.
                Keys: 'target_heatmap', 'target_mask' (dense heatmap mask),
                      'target_indices', 'reg_mask' (sparse regression mask),
                      'target_offset', 'target_z_coord', 'target_dimension',
                      'target_rotation', 'target_velocity', 'num_objs'.
        Returns:
            dict: Dictionary of raw loss values for each task head.
        """
        losses = {}

        # 1. Classification loss (Focal Loss)
        pred_heatmap = outputs['heatmap']       # Logits [B, num_classes, H, W]
        target_heatmap = targets['target_heatmap'].to(
            pred_heatmap.device)  # Ensure device match

        # === MODIFICATION START: Calculate Focal Loss with logits and normalize correctly ===
        # Calculate the sum of focal loss across all pixels using logits
        focal_loss_sum = self.crit_cls(pred_heatmap, target_heatmap)

        # Normalize by the number of positive objects in the batch
        # Use reg_mask (sparse mask) to count positive objects
        if 'reg_mask' in targets:
            # Sum over all objects in the batch
            num_pos = targets['reg_mask'].sum().float().clamp(min=1.0)
        elif 'num_objs' in targets:
            # Fallback: use num_objs if reg_mask is not available
            # Handle list case from DataLoader collation
            if isinstance(targets['num_objs'], list):
                num_pos = torch.tensor(sum(
                    targets['num_objs']), dtype=torch.float, device=pred_heatmap.device).clamp(min=1.0)
            elif isinstance(targets['num_objs'], torch.Tensor):
                num_pos = targets['num_objs'].sum().float().clamp(min=1.0)
            else:  # Handle unexpected type
                print(
                    f"Warning: Unexpected type for targets['num_objs']: {type(targets['num_objs'])}. Using 1 for normalization.")
                num_pos = torch.tensor(1.0, device=pred_heatmap.device)

            # Original check for num_pos == 0 remains relevant
            if num_pos == 0:
                # Avoid division by zero
                num_pos = torch.tensor(1.0, device=pred_heatmap.device)
            print("Warning: 'reg_mask' not found in targets for CenterPointLoss normalization. Using 'num_objs' sum instead.")
        else:
            # Fallback if neither is available (less ideal)
            # Normalize by a fixed value or total elements? Let's use 1 to avoid division by zero
            # but this might skew loss if batches truly have zero objects.
            num_pos = torch.tensor(1.0, device=pred_heatmap.device)
            print("Warning: Neither 'reg_mask' nor 'num_objs' found in targets for CenterPointLoss normalization. Using 1.")

        raw_heatmap_loss = focal_loss_sum / num_pos
        losses['raw_heatmap_loss'] = raw_heatmap_loss
        # === MODIFICATION END ===

        # Prepare inputs for regression losses
        # The collate_fn should handle padding and batching correctly now
        ind = targets['target_indices'].to(pred_heatmap.device)
        reg_mask = targets['reg_mask'].to(
            pred_heatmap.device)  # Ensure device match

        # Helper function for regression loss calculation
        def calculate_reg_loss(head_name, target_key):
            if head_name in outputs and target_key in targets:
                pred = outputs[head_name]
                # Ensure target is on the same device
                target = targets[target_key].to(pred.device)
                loss, _ = self.crit_reg(
                    pred, target, ind, reg_mask)  # Pass reg_mask
                # Raw loss is returned, weighting happens outside
                return loss
            return torch.tensor(0.0, device=pred_heatmap.device)

        # 2. Regression Losses (using crit_reg with reg_mask)
        losses['raw_offset_loss'] = calculate_reg_loss(
            'offset', 'target_offset')
        losses['raw_z_coord_loss'] = calculate_reg_loss(
            'z_coord', 'target_z_coord')
        losses['raw_dimension_loss'] = calculate_reg_loss(
            'dimension', 'target_dimension')
        losses['raw_rotation_loss'] = calculate_reg_loss(
            'rotation', 'target_rotation')
        losses['raw_velocity_loss'] = calculate_reg_loss(
            'velocity', 'target_velocity')

        return losses
