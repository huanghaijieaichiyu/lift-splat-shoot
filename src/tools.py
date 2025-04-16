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
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
        "singapore-onenorth",
    ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get(
        'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0],
                      egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


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
    针对3D目标检测的多任务损失函数
    """

    def __init__(self, num_classes=10, cls_weight=1.0, reg_weight=1.0, iou_weight=1.0):
        super(Detection3DLoss, self).__init__()
        self.num_classes = num_classes
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight

        # 使用timm库中的分类损失函数
        try:
            from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
            self.cls_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        except ImportError:
            print("Warning: timm library not found. Using standard CrossEntropy loss")
            self.cls_loss_fn = nn.CrossEntropyLoss()

        # 回归损失 - Smooth L1 Loss
        self.reg_loss_fn = nn.SmoothL1Loss(reduction='none')

        # IoU损失 - BCE Loss
        self.iou_loss_fn = nn.BCEWithLogitsLoss(reduction='none')

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
        # [B, 9, H, W] - x, y, z, w, l, h, sin(θ), cos(θ), 速度
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

        # 过滤掉padding区域
        valid_mask = (cls_targets >= 0) & (cls_targets < self.num_classes)

        if valid_mask.sum() > 0:
            cls_loss = self.cls_loss_fn(
                cls_pred[valid_mask], cls_targets[valid_mask])
        else:
            cls_loss = torch.tensor(0.0, device=cls_pred.device)

        # 计算回归损失 - 只考虑有目标的位置
        reg_pred = reg_pred.permute(0, 2, 3, 1)  # [B, H, W, 9]
        reg_targets = reg_targets.permute(0, 2, 3, 1)  # [B, H, W, 9]
        reg_weights = reg_weights.permute(0, 2, 3, 1)  # [B, H, W, 1]

        # 计算平滑L1损失
        reg_loss_raw = self.reg_loss_fn(reg_pred, reg_targets)  # [B, H, W, 9]
        reg_loss = (reg_loss_raw * reg_weights).sum() / \
            (reg_weights.sum().clamp(min=1.0))

        # 计算IoU损失
        iou_pred = iou_pred.permute(0, 2, 3, 1)  # [B, H, W, 1]
        iou_targets = iou_targets.permute(0, 2, 3, 1)  # [B, H, W, 1]
        iou_loss = (self.iou_loss_fn(iou_pred, iou_targets) *
                    reg_weights).sum() / (reg_weights.sum().clamp(min=1.0))

        # 计算总损失
        total_loss = self.cls_weight * cls_loss + \
            self.reg_weight * reg_loss + self.iou_weight * iou_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
            'iou_loss': iou_loss
        }
