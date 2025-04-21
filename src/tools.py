"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import math
from typing import Dict
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
    ], dtype=centers.dtype, device=device) * torch.cat([len_half, width_half], dim=1).unsqueeze(1)  # [N, 4, 2]

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


# Helper function for Rotated DIoU Loss calculation
def rotated_diou_loss(corners_pred: torch.Tensor, corners_target: torch.Tensor,
                      centers_pred: torch.Tensor, centers_target: torch.Tensor,
                      eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates Rotated DIoU loss.

    Args:
        corners_pred: Predicted box corners [N, 4, 2].
        corners_target: Target box corners [N, 4, 2].
        centers_pred: Predicted box centers [N, 2].
        centers_target: Target box centers [N, 2].
        eps: Small value for numerical stability.

    Returns:
        diou_loss: DIoU loss value for each pair [N].
    """
    # Calculate IoU using the (placeholder) rotated IoU function
    iou = rotated_iou_bev(corners_pred, corners_target)

    # Calculate the smallest enclosing box (axis-aligned for simplicity here)
    # A true rotated DIoU would use the smallest enclosing rotated box.
    all_corners = torch.cat([corners_pred, corners_target], dim=1)  # [N, 8, 2]
    min_coords, _ = torch.min(all_corners, dim=1)  # [N, 2]
    max_coords, _ = torch.max(all_corners, dim=1)  # [N, 2]

    # Diagonal squared of the enclosing box
    c2 = ((max_coords - min_coords) ** 2).sum(dim=-1)  # [N]

    # Center distance squared
    d2 = ((centers_pred - centers_target) ** 2).sum(dim=-1)  # [N]

    # DIoU = IoU - d2 / c2
    diou = iou - (d2 / c2.clamp(min=eps))

    # DIoU Loss = 1 - DIoU
    loss = 1.0 - diou
    return loss


class DetectionBEVLoss(nn.Module):
    """
    Refined BEV Detection Loss using Rotated DIoU for BEV components.

    Computes losses for:
    - Classification (Focal Loss)
    - BEV Box Regression (Rotated DIoU Loss for x, y, w, l, yaw)
    - Z coordinate Regression (Smooth L1)
    - Height Regression (Smooth L1)
    - Velocity Regression (Smooth L1 for vx, vy)
    - IoU prediction head (BCE).
    """

    def __init__(self, num_classes: int,
                 cls_weight: float = 3.0,
                 bev_loss_weight: float = 2.0,  # Weight for BEV DIoU loss
                 z_loss_weight: float = 1.0,
                 h_loss_weight: float = 1.0,
                 vel_loss_weight: float = 2.0,
                 iou_weight: float = 2.0,
                 alpha: float = 0.25, gamma: float = 2.0, beta: float = 1.0,  # beta for SmoothL1
                 eps: float = 1e-7):
        """
        Args:
            num_classes: Number of foreground object classes.
            cls_weight: Weight for the classification loss.
            bev_loss_weight: Weight for the BEV Rotated DIoU loss (x,y,w,l,yaw).
            z_loss_weight: Weight for the Z-coordinate Smooth L1 loss.
            h_loss_weight: Weight for the height Smooth L1 loss.
            vel_loss_weight: Weight for the velocity Smooth L1 loss.
            iou_weight: Weight for the IoU prediction head loss.
            alpha: Alpha parameter for Focal Loss.
            gamma: Gamma parameter for Focal Loss.
            beta: Beta parameter for Smooth L1 Loss (delta).
            eps: Small epsilon value for numerical stability.
        """
        super().__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        # Store individual regression weights
        self.bev_loss_weight = bev_loss_weight
        self.z_loss_weight = z_loss_weight
        self.h_loss_weight = h_loss_weight
        self.vel_loss_weight = vel_loss_weight
        self.iou_weight = iou_weight

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.eps = eps

        # Regression Loss for non-BEV components
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=self.beta)

        # IoU Prediction Loss (applied element-wise)
        self.iou_loss_fn = nn.BCEWithLogitsLoss()

    def _focal_loss(self, pred_logits: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """
        Computes Focal Loss.

        Args:
            pred_logits: Predicted logits [N, num_classes].
            target_labels: Target class labels [N].

        Returns:
            Per-element focal loss [N].
        """
        pred_softmax = F.softmax(pred_logits, dim=1)
        pt = pred_softmax.gather(1, target_labels.unsqueeze(1)).squeeze(1)
        pt = torch.clamp(pt, self.eps, 1.0 - self.eps)
        alpha_t = torch.full_like(
            target_labels, 1 - self.alpha, dtype=torch.float)
        foreground_mask = target_labels > 0
        alpha_t[foreground_mask] = self.alpha
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return loss

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Computes the combined detection loss with Rotated DIoU for BEV components.

        Args:
            preds: Dictionary of predictions from the model:
                'cls_pred': [B, num_classes, H, W] Classification logits.
                'reg_pred': [B, 9, H, W] Regression predictions (x,y,z, w,l,h, sin,cos, vx,vy).
                'iou_pred': [B, 1, H, W] IoU/centerness logits.
            targets: Dictionary of ground truth targets:
                'cls_targets': [B, H, W] Target class indices (0 for bg).
                'reg_targets': [B, 9, H, W] Target regression values.
                'reg_weights': [B, H, W] Mask indicating positive locations (1 for pos, 0 for neg).
                'iou_targets': [B, H, W] Target values for IoU head (e.g., 0.0-1.0 score).

        Returns:
            Dictionary containing computed losses: 'total_loss', 'cls_loss',
            'bev_diou_loss', 'z_loss', 'h_loss', 'vel_loss', 'iou_loss'.
        """
        # --- Input Parsing and Reshaping ---
        cls_pred = preds['cls_pred']
        reg_pred = preds['reg_pred']
        iou_pred = preds.get('iou_pred')

        cls_targets = targets['cls_targets'].long()
        reg_targets = targets['reg_targets']
        reg_weights = targets['reg_weights']  # Positive location mask
        iou_targets = targets.get('iou_targets')

        B, _, H, W = cls_pred.shape
        num_predictions = B * H * W

        # Reshape for loss calculation
        cls_pred_flat = cls_pred.permute(0, 2, 3, 1).reshape(
            num_predictions, self.num_classes)
        reg_pred_flat = reg_pred.permute(
            0, 2, 3, 1).reshape(num_predictions, 9)
        if iou_pred is not None:
            iou_pred_flat = iou_pred.permute(
                0, 2, 3, 1).reshape(num_predictions, 1)
        else:
            iou_pred_flat = None

        cls_targets_flat = cls_targets.reshape(num_predictions)
        reg_targets_flat = reg_targets.permute(
            0, 2, 3, 1).reshape(num_predictions, 9)
        reg_weights_flat = reg_weights.reshape(num_predictions)
        if iou_targets is not None:
            iou_targets_flat = iou_targets.reshape(num_predictions, 1)
        else:
            iou_targets_flat = None

        # --- Masks ---
        valid_cls_mask = cls_targets_flat >= 0
        pos_mask = reg_weights_flat > 0
        num_pos = pos_mask.sum().clamp(min=1.0)

        # --- Classification Loss (Focal Loss) ---
        if valid_cls_mask.sum() == 0:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)
        else:
            loss_cls_all = self._focal_loss(
                cls_pred_flat[valid_cls_mask], cls_targets_flat[valid_cls_mask])
            # Average over all valid elements
            cls_loss = loss_cls_all.sum() / valid_cls_mask.sum().clamp(min=1.0)

        # --- Regression Losses (Split into BEV-DIoU and L1 for others) ---
        bev_diou_loss_w = torch.tensor(0.0, device=reg_pred.device)
        z_loss_w = torch.tensor(0.0, device=reg_pred.device)
        h_loss_w = torch.tensor(0.0, device=reg_pred.device)
        vel_loss_w = torch.tensor(0.0, device=reg_pred.device)

        if num_pos > 0:
            reg_pred_pos = reg_pred_flat[pos_mask]  # [NumPos, 9]
            reg_targets_pos = reg_targets_flat[pos_mask]  # [NumPos, 9]

            # --- BEV DIoU Loss (x, y, w, l, yaw) ---
            # Extract BEV components
            centers_pred = reg_pred_pos[:, :2]    # x, y
            dims_pred_wl = reg_pred_pos[:, 3:5]  # w, l
            # ** IMPORTANT Assumption **: Model predicts direct w,l. Swap to l,w for helper.
            dims_pred = dims_pred_wl[:, [1, 0]]
            sin_cos_pred = reg_pred_pos[:, 6:8]   # sin, cos
            yaw_pred = torch.atan2(sin_cos_pred[:, 0], sin_cos_pred[:, 1])

            centers_target = reg_targets_pos[:, :2]
            dims_target_wl = reg_targets_pos[:, 3:5]  # w, l
            # ** IMPORTANT Assumption **: Targets have direct w,l. Swap to l,w for helper.
            dims_target = dims_target_wl[:, [1, 0]]
            sin_cos_target = reg_targets_pos[:, 6:8]
            yaw_target = torch.atan2(
                sin_cos_target[:, 0], sin_cos_target[:, 1])

            # Convert to corners
            corners_pred = corners_from_center_dims_yaw(
                centers_pred, dims_pred, yaw_pred)
            corners_target = corners_from_center_dims_yaw(
                centers_target, dims_target, yaw_target)

            # Calculate Rotated DIoU Loss
            bev_diou_loss_all = rotated_diou_loss(corners_pred, corners_target,
                                                  centers_pred, centers_target, self.eps)
            bev_diou_loss_w = bev_diou_loss_all.sum() / num_pos  # Average over positive samples

            # --- Z Loss (Smooth L1) ---
            z_pred = reg_pred_pos[:, 2:3]
            z_target = reg_targets_pos[:, 2:3]
            z_loss_all = self.smooth_l1_loss(z_pred, z_target)
            z_loss_w = z_loss_all.sum() / num_pos  # Average over positive samples

            # --- Height Loss (Smooth L1) ---
            # Assuming model predicts h at index 5
            h_pred = reg_pred_pos[:, 5:6]
            h_target = reg_targets_pos[:, 5:6]
            h_loss_all = self.smooth_l1_loss(h_pred, h_target)
            h_loss_w = h_loss_all.sum() / num_pos  # Average over positive samples

            # --- Velocity Loss (Smooth L1 for vx, vy) ---
            # Assuming model predicts vx, vy at indices 8, 9 (or just 8)
            vel_pred = reg_pred_pos[:, 8:]  # Take last components
            vel_target = reg_targets_pos[:, 8:]

            # Handle case where only one velocity component might be present
            if vel_pred.shape[1] == 1 and vel_target.shape[1] == 1:
                vel_loss_all = self.smooth_l1_loss(vel_pred, vel_target)
                vel_loss_w = vel_loss_all.sum() / num_pos
            elif vel_pred.shape[1] >= 2 and vel_target.shape[1] >= 2:
                # Ensure both have 2 components if more are predicted/targeted
                vel_loss_all = self.smooth_l1_loss(
                    vel_pred[:, :2], vel_target[:, :2])  # [NumPos, 2]
                vel_loss_w = vel_loss_all.sum() / num_pos  # Sum over vx,vy and average
        else:
            # Shape mismatch or missing velocity - loss is 0
            vel_loss_w = torch.tensor(0.0, device=reg_pred.device)

        # --- IoU Prediction Loss (BCE) ---
        if num_pos == 0 or iou_pred_flat is None or iou_targets_flat is None:
            iou_loss_w = torch.tensor(0.0, device=cls_pred.device)
        else:
            loss_iou_all = self.iou_loss_fn(
                iou_pred_flat[pos_mask], iou_targets_flat[pos_mask])
            iou_loss_w = loss_iou_all.sum() / num_pos

        # --- Apply Weights and Combine ---
        # Note: cls_loss was already averaged over valid samples
        final_cls_loss = cls_loss * self.cls_weight
        final_bev_diou_loss = bev_diou_loss_w * self.bev_loss_weight
        final_z_loss = z_loss_w * self.z_loss_weight
        final_h_loss = h_loss_w * self.h_loss_weight
        final_vel_loss = vel_loss_w * self.vel_loss_weight
        final_iou_loss = iou_loss_w * self.iou_weight

        total_loss = (final_cls_loss + final_bev_diou_loss + final_z_loss +
                      final_h_loss + final_vel_loss + final_iou_loss)

        return {
            'total_loss': total_loss,
            'cls_loss': final_cls_loss,           # Weighted losses
            'bev_diou_loss': final_bev_diou_loss,
            'z_loss': final_z_loss,
            'h_loss': final_h_loss,
            'vel_loss': final_vel_loss,
            'iou_loss': final_iou_loss,
        }
