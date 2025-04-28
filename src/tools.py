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
    """
    Generates grid configuration parameters.

    Args:
        xbound (list): [xmin, xmax, xstep]
        ybound (list): [ymin, ymax, ystep]
        zbound (list): [zmin, zmax, zstep]

    Returns:
        dx (torch.Tensor): Voxel dimensions [dx, dy, dz].
        bx (torch.Tensor): Bottom-left corner coordinates [bx, by, bz].
        nx (torch.Tensor): Grid dimensions [nx, ny, nz].
    """
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                          for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    """
    Efficiently sums features belonging to the same voxel using cumsum.

    Args:
        x (torch.Tensor): Flattened features (N_points, C).
        geom_feats (torch.Tensor): Voxel indices for each point (N_points, 4), cols are (x, y, z, batch_idx).
        ranks (torch.Tensor): Unique rank for each voxel per batch item (N_points,).

    Returns:
        x (torch.Tensor): Summed features per voxel (N_voxels, C).
        geom_feats (torch.Tensor): Corresponding voxel indices (N_voxels, 4).
    """
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x = x[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    # rank = ranks[kept]
    geom_feats = geom_feats[kept]
    # print(x.shape)

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    """
    Custom autograd Function for cumsum trick with backward pass.
    (Note: Backward pass might be simplified or require verification)
    """
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x = x[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # rank = ranks[kept]
        # geom_feats = geom_feats[kept]
        ctx.save_for_backward(kept)
        # print(x.shape)
        return x, geom_feats[kept]

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
    """
    Calculates intersection and union for a batch.
    Args:
        preds: [B, H, W] or [B, 1, H, W] - Model output (logits or probabilities)
        binimgs: [B, H, W] or [B, 1, H, W] - Ground truth binary masks
    Returns:
        intersect: torch.Tensor (scalar)
        union: torch.Tensor (scalar)
        iou: torch.Tensor (scalar)
    """
    # Remove channel dimension if present
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if binimgs.dim() == 4:
        binimgs = binimgs.squeeze(1)

    # Ensure preds and binimgs are boolean tensors
    # Apply sigmoid and threshold to preds if they are logits
    if preds.dtype != torch.bool:
        preds = (torch.sigmoid(preds) > 0.5)
    if binimgs.dtype != torch.bool:
        binimgs = (binimgs > 0.5)

    intersect = (preds & binimgs).sum().float()
    union = (preds | binimgs).sum().float()
    iou = intersect / (union + 1e-6)  # Add epsilon to avoid division by zero
    return intersect, union, iou


def get_val_info(model, valloader, loss_fn, device):
    """
    Calculate validation loss and IoU for the original (non-fusion) model.
    """
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0.0
    print('running eval...')
    with torch.no_grad():
        # Original validation loop (for non-fusion models)
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(tqdm(valloader)):
            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * \
                imgs.size(0)  # Weighted by batch size

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()  # Set back to train mode
    num_samples = len(valloader.dataset)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_iou = total_intersect / (total_union + 1e-6)
    return {'loss': avg_loss, 'iou': avg_iou}


def get_val_info_fusion(model, valloader, loss_fn, device):
    """
    Calculate validation loss and IoU for the FusionNet model.
    Handles batches containing LiDAR BEV data.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0.0
    print('Running validation for Fusion model...')

    with torch.no_grad():  # Disable gradient calculations
        # Iterate through the validation dataloader
        val_pbar = tqdm(enumerate(valloader), total=len(
            valloader), desc="Fusion Validation")
        for batchi, batch_data in val_pbar:
            # Unpack batch data including lidar_bev
            if len(batch_data) != 8:
                print(
                    f"Warning: Validation dataloader expected 8 items, got {len(batch_data)}. Skipping batch.")
                continue
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev = batch_data
            except ValueError as e:
                print(
                    f"Error unpacking validation batch data: {e}. Expected 8 items. Skipping batch.")
                continue

            # Move data to the target device
            imgs = imgs.to(device)
            rots = rots.to(device)
            trans = trans.to(device)
            intrins = intrins.to(device)
            post_rots = post_rots.to(device)
            post_trans = post_trans.to(device)
            binimgs = binimgs.to(device)
            lidar_bev = lidar_bev.to(device)

            # Forward pass
            preds = model(imgs, rots, trans, intrins,
                          post_rots, post_trans, lidar_bev)

            # Calculate loss
            batch_loss = loss_fn(preds, binimgs)
            # Accumulate loss weighted by actual batch size (important for last batch)
            total_loss += batch_loss.item() * imgs.size(0)

            # Calculate IoU
            intersect, union, batch_iou = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

            val_pbar.set_postfix(
                loss=f"{batch_loss.item():.4f}", iou=f"{batch_iou:.4f}")

    # Calculate average loss and IoU over the entire validation set
    num_samples = len(valloader.dataset)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    # Add epsilon for stability
    avg_iou = total_intersect / (total_union + 1e-6)

    print(
        f"Validation Complete. Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")
    model.train()  # Set model back to training mode

    return {'loss': avg_loss, 'iou': avg_iou}


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
