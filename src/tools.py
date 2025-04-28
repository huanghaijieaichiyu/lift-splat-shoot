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
from nuscenes import NuScenes  # Added import
# Import necessary classes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer, BitMap
# Import geometry utils
from nuscenes.utils.geometry_utils import transform_matrix, view_points
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
    Calculates intersection, union, and components for precision/recall for a batch.
    Args:
        preds: [B, H, W] or [B, 1, H, W] - Model output (logits or probabilities)
        binimgs: [B, H, W] or [B, 1, H, W] - Ground truth binary masks
    Returns:
        intersect: torch.Tensor (scalar) - True Positives (TP)
        union: torch.Tensor (scalar)
        iou: torch.Tensor (scalar)
        fp: torch.Tensor (scalar) - False Positives
        fn: torch.Tensor (scalar) - False Negatives
    """
    # Remove channel dimension if present
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if binimgs.dim() == 4:
        binimgs = binimgs.squeeze(1)

    # Ensure preds and binimgs are boolean tensors
    # Apply sigmoid and threshold to preds if they are logits
    if preds.dtype != torch.bool:
        preds_bool = (torch.sigmoid(preds) > 0.5)
    else:
        preds_bool = preds
    if binimgs.dtype != torch.bool:
        binimgs_bool = (binimgs > 0.5)
    else:
        binimgs_bool = binimgs

    # True Positives (TP)
    intersect = (preds_bool & binimgs_bool).sum().float()
    union = (preds_bool | binimgs_bool).sum().float()
    iou = intersect / (union + 1e-6)  # Add epsilon to avoid division by zero

    # Calculate False Positives (FP) and False Negatives (FN)
    tp = intersect
    # Predicted positive, actually negative
    fp = (preds_bool & ~binimgs_bool).sum().float()
    # Predicted negative, actually positive
    fn = (~preds_bool & binimgs_bool).sum().float()

    # Alternative calculation (sometimes helps with understanding):
    # total_pred_positive = preds_bool.sum().float()
    # total_gt_positive = binimgs_bool.sum().float()
    # fp = total_pred_positive - tp
    # fn = total_gt_positive - tp

    return intersect, union, iou, fp, fn


def get_val_info(model, valloader, loss_fn, device):
    """
    Calculate validation loss, IoU, Precision, Recall, and F1-Score
    for the original (non-fusion) model.
    """
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0.0
    total_fp = 0.0
    total_fn = 0.0
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

            # iou and precision/recall components
            intersect, union, _, fp, fn = get_batch_iou(preds, binimgs)
            total_intersect += intersect  # TP
            total_union += union
            total_fp += fp
            total_fn += fn

    model.train()  # Set back to train mode
    num_samples = len(valloader.dataset)
    avg_loss = total_loss / num_samples if num_samples > 0 else 0

    # Calculate metrics
    tp = total_intersect
    precision = tp / (tp + total_fp + 1e-6)
    recall = tp / (tp + total_fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    # IoU calculated from total TP and Union
    avg_iou = tp / (total_union + 1e-6)

    return {
        'loss': avg_loss,
        'iou': avg_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Helper function to convert NuScenes map mask to a PyTorch tensor grid
def map_mask_to_tensor(map_mask, grid_conf, patch_center, patch_angle):
    """
    Converts a NuScenes BitMap mask to a PyTorch tensor aligned with the grid_conf.
    Assumes grid_conf defines the ego-centric BEV grid.
    patch_center and patch_angle define the *global* pose used to get the map_mask.
    """
    # Define the target ego-centric grid
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                          for row in [grid_conf['xbound'], grid_conf['ybound']]])
    dx = torch.Tensor([row[2]
                      for row in [grid_conf['xbound'], grid_conf['ybound']]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [grid_conf['xbound'], grid_conf['ybound']]])

    # Create the grid points in ego BEV space (origin at ego vehicle)
    xs = torch.linspace(bx[0], grid_conf['xbound'][1] - dx[0]/2.0, int(nx[0]))
    ys = torch.linspace(bx[1], grid_conf['ybound'][1] - dx[1]/2.0, int(nx[1]))
    # Ensure indexing='ij' for compatibility with typical image coordinates (H, W)
    # If model output is W, H, use 'xy'
    # Use 'ij' for H, W output grid
    xx, yy = torch.meshgrid(xs, ys, indexing='ij')
    # grid_points_ego_bev = torch.stack([xx.flatten(), yy.flatten()], dim=0) # Shape (2, N) - This might be (x, y) need (y, x) for ij
    # Use (y, x) order for ij indexing
    grid_points_ego_bev = torch.stack([yy.flatten(), xx.flatten()], dim=0)

    # Transform these ego BEV grid points to the global frame
    # Rotation matrix from global to ego: R_ego_global = R_global_ego.T
    # Rotation matrix from ego to global: R_global_ego
    # Note: patch_angle is yaw (rotation around Z). We need 2D rotation.
    cos_a, sin_a = np.cos(patch_angle), np.sin(patch_angle)
    rot_mat_global_ego = torch.tensor([[cos_a, -sin_a],
                                      [sin_a, cos_a]], dtype=torch.float32)  # Transforms points from ego to global

    trans_vec_global = torch.tensor(
        patch_center[:2], dtype=torch.float32).unsqueeze(1)  # Global position of ego

    # Apply transform: ego_pts -> global_pts = R_global_ego @ ego_pts + T_global_ego
    grid_points_global = torch.matmul(
        rot_mat_global_ego, grid_points_ego_bev) + trans_vec_global  # (2, N)

    # Query the NuScenes map mask (which is in global frame)
    # map_mask.transform_matrix transforms from global coords to mask pixel coords (u, v) ~ (col, row)
    map_tf = torch.tensor(map_mask.transform_matrix, dtype=torch.float32)
    # Add homogeneous coordinate (z=1) for affine transformation
    grid_points_global_h = torch.cat([grid_points_global, torch.ones(
        (1, grid_points_global.shape[1]))], dim=0)  # (3, N)
    grid_points_map_coords_h = torch.matmul(
        map_tf, grid_points_global_h)  # (3, N)

    # Normalize homogeneous coordinates to get pixel coordinates (u, v)
    # Handle potential division by zero if z-coordinate is zero, though unlikely for map transforms
    grid_points_map_coords_z = grid_points_map_coords_h[2:]
    # grid_points_map_coords_z[grid_points_map_coords_z == 0] = 1e-6 # Avoid division by zero
    # (2, N) = (u, v) = (col, row)
    grid_points_map_coords = grid_points_map_coords_h[:2] / (
        grid_points_map_coords_z + 1e-8)

    # Sample the mask - need integer coordinates
    # NuScenes Bitmap uses (col_idx, row_idx) which corresponds to (u, v)
    sample_u = grid_points_map_coords[0].round().long()  # Column index
    sample_v = grid_points_map_coords[1].round().long()  # Row index

    # Get mask dimensions (height, width) corresponding to (rows, cols)
    mask_h, mask_w = map_mask.mask.shape
    # Check validity of sampled coordinates
    valid_mask = (sample_u >= 0) & (sample_u < mask_w) & (
        sample_v >= 0) & (sample_v < mask_h)

    # Initialize flattened raster tensor
    # Flattened raster
    raster = torch.zeros(int(nx[0]*nx[1]), dtype=torch.bool)

    # Efficiently query the mask where coordinates are valid
    valid_indices = torch.where(valid_mask)[0]
    if len(valid_indices) > 0:
        # Note: map_mask.mask is H x W (row, col), so index with (v, u)
        raster[valid_indices] = torch.from_numpy(
            map_mask.mask[sample_v[valid_indices], sample_u[valid_indices]])

    # Reshape to grid format (H_bev, W_bev) matching meshgrid('ij') -> (ny, nx)
    # Reshape to (num_y_steps, num_x_steps)
    raster = raster.view(int(nx[1]), int(nx[0]))

    return raster.float()  # Return as float tensor (0.0 or 1.0)


def rasterize_nusc_map(nusc_map_explorer, nusc_map_api, sample_token, layer_names, grid_conf):
    """
    Rasterizes specified NuScenes map layers for a given sample token onto a BEV grid.

    Args:
        nusc_map_explorer: Initialized NuScenesMapExplorer object.
        nusc_map_api: NuScenesMap API object for the specific map location.
        sample_token: The NuScenes sample token.
        layer_names: List of map layer names to rasterize (e.g., ['drivable_area']).
        grid_conf: Dictionary defining the BEV grid (xbound, ybound, etc.).

    Returns:
        torch.Tensor: Rasterized map (H_bev, W_bev), or None if error.
    """
    try:
        sample_record = nusc_map_explorer.nusc.get('sample', sample_token)
        # Use lidar or radar, assuming lidar is primary
        sd_record = nusc_map_explorer.nusc.get('sample_data', sample_record['data'].get(
            'LIDAR_TOP', sample_record['data'].get('RADAR_FRONT')))
        if sd_record is None:
            print(
                f"Warning: No LIDAR_TOP or RADAR_FRONT found for sample {sample_token}")
            # Attempt to get any sample data to find ego pose
            any_sd_token = list(sample_record['data'].values())[0]
            sd_record = nusc_map_explorer.nusc.get('sample_data', any_sd_token)

        pose_record = nusc_map_explorer.nusc.get(
            'ego_pose', sd_record['ego_pose_token'])
        # Global coords center (x, y, z)
        ego_translation_global = pose_record['translation']
        # Ego yaw in global frame
        ego_yaw_global = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]

        # Define the query patch in GLOBAL coordinates based on ego pose and BEV grid size
        # Grid size in meters
        grid_x_size = abs(grid_conf['xbound'][1] - grid_conf['xbound'][0])
        grid_y_size = abs(grid_conf['ybound'][1] - grid_conf['ybound'][0])
        # Use a slightly larger box for safety margin when querying map mask
        # Calculate max extent needed based on rotation
        extent_needed = np.sqrt((grid_x_size/2)**2 +
                                (grid_y_size/2)**2) + 5.0  # 5m margin

        query_box_global = (
            ego_translation_global[0] - extent_needed,  # x_min_global
            ego_translation_global[1] - extent_needed,  # y_min_global
            ego_translation_global[0] + extent_needed,  # x_max_global
            ego_translation_global[1] + extent_needed  # y_max_global
        )

        # Get map mask using the GLOBAL query box and the GLOBAL ego yaw
        map_mask = nusc_map_api.get_map_mask(
            query_box_global, ego_yaw_global, layer_names, canvas_size=None)  # Let API decide canvas size

        # Check if map mask was retrieved successfully
        if map_mask is None or map_mask.mask is None:
            print(
                f"Warning: Could not retrieve map mask for layers {layer_names} at token {sample_token} in map {nusc_map_api.map_name}")
            # Return an empty tensor matching the expected output shape
            nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                                  for row in [grid_conf['xbound'], grid_conf['ybound']]])
            return torch.zeros((int(nx[1]), int(nx[0])), dtype=torch.float32)

        # Convert the NuScenes BitMap mask to our ego-centric BEV tensor grid
        # Pass the global pose (translation, yaw) used to get the mask
        raster_tensor = map_mask_to_tensor(
            map_mask, grid_conf, ego_translation_global, ego_yaw_global)

        return raster_tensor

    except KeyError as e:
        print(
            f"Error accessing NuScenes data for token {sample_token}: Missing key {e}")
    except Exception as e:
        print(
            f"Error rasterizing map for token {sample_token}: {type(e).__name__} - {e}")

    # Return an empty tensor on error, maintaining shape consistency
    nx = torch.LongTensor([(row[1] - row[0]) / row[2]
                          for row in [grid_conf['xbound'], grid_conf['ybound']]])
    return torch.zeros((int(nx[1]), int(nx[0])), dtype=torch.float32)


# Modify get_batch_iou to handle boolean inputs correctly for devkit GT
def get_batch_iou_flexible(preds, gt_masks):
    """
    Calculates intersection, union, and components for precision/recall for a batch.
    Handles both logit preds and boolean/float gt_masks.

    Args:
        preds: [B, H, W] or [B, 1, H, W] - Model output (logits)
        gt_masks: [B, H, W] or [B, 1, H, W] - Ground truth masks (float or bool)

    Returns:
        intersect: torch.Tensor (scalar) - True Positives (TP)
        union: torch.Tensor (scalar)
        iou: torch.Tensor (scalar)
        fp: torch.Tensor (scalar) - False Positives
        fn: torch.Tensor (scalar) - False Negatives
    """
    # Remove channel dimension if present
    if preds.dim() == 4:
        preds = preds.squeeze(1)
    if gt_masks.dim() == 4:
        gt_masks = gt_masks.squeeze(1)

    # Apply sigmoid and threshold to preds (always assume preds are logits)
    preds_bool = (torch.sigmoid(preds) > 0.5)

    # Ensure gt_masks are boolean
    if gt_masks.dtype != torch.bool:
        gt_masks_bool = (gt_masks > 0.5)  # Threshold if float/int
    else:
        gt_masks_bool = gt_masks

    # Ensure both are on the same device and have same shape
    if gt_masks_bool.shape != preds_bool.shape:
        print(
            f"Warning: GT mask shape {gt_masks_bool.shape} differs from Preds shape {preds_bool.shape}. Resizing GT.")
        # Resize GT mask to match prediction shape using nearest interpolation
        gt_masks_bool = F.interpolate(gt_masks_bool.unsqueeze(1).float(
        ), size=preds_bool.shape[-2:], mode='nearest').squeeze(1).bool()

    gt_masks_bool = gt_masks_bool.to(preds_bool.device)

    # True Positives (TP)
    intersect = (preds_bool & gt_masks_bool).sum().float()
    union = (preds_bool | gt_masks_bool).sum().float()
    iou = intersect / (union + 1e-6)  # Add epsilon to avoid division by zero

    tp = intersect
    # Predicted positive, actually negative
    fp = (preds_bool & ~gt_masks_bool).sum().float()
    # Predicted negative, actually positive
    fn = (~preds_bool & gt_masks_bool).sum().float()

    return intersect, union, iou, fp, fn


# Update get_val_info_fusion
def get_val_info_fusion(model, valloader, loss_fn, device, nusc, grid_conf,
                        writer, global_step,
                        map_layers=['drivable_area'],
                        final_dim_vis=None,
                        D_depth=None):
    """
    Calculate validation metrics AND performs visualization for the first batch.
    Includes:
    1. Simple metrics (loss, IoU, P, R, F1) against dataloader's 'binimgs'.
    2. Devkit-based metrics (IoU, P, R, F1) against rasterized NuScenes map layers.
    3. Tensorboard visualization for the first valid batch.
    """
    model.eval()
    total_loss_simple = 0.0
    total_tp_simple, total_fp_simple, total_fn_simple, total_union_simple = 0.0, 0.0, 0.0, 0.0
    total_tp_devkit, total_fp_devkit, total_fn_devkit, total_union_devkit = 0.0, 0.0, 0.0, 0.0
    num_samples = 0
    num_devkit_samples_ok = 0
    visualized_batch = False

    # Initialize NuScenesMapExplorer and cache MapAPIs
    nusc_map_explorer = None
    map_apis = {}
    if nusc is not None:
        try:
            nusc_map_explorer = NuScenesMapExplorer(nusc)
        except Exception as e:
            print(
                f"Warning: Failed to initialize NuScenesMapExplorer: {e}. Devkit eval might fail.")
    else:
        print("NuScenes API object is None. Skipping devkit evaluation.")

    print('Running validation for Fusion model (Simple + Devkit + Vis)...')
    with torch.no_grad():
        val_pbar = tqdm(enumerate(valloader), total=len(
            valloader), desc="Fusion Validation")
        for batchi, batch_data in val_pbar:
            # Expect 9 items: imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev, sample_tokens
            if len(batch_data) != 9:
                print(
                    f"Warning: Val dataloader expected 9 items, got {len(batch_data)}. Skipping.")
                continue
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev, sample_tokens = batch_data
                B = imgs.shape[0]  # Batch size
                num_samples += B
            except ValueError as e:
                print(f"Error unpacking val batch data: {e}. Skipping.")
                continue

            # Move data to device
            imgs_dev = imgs.to(device)
            rots_dev = rots.to(device)
            trans_dev = trans.to(device)
            intrins_dev = intrins.to(device)
            post_rots_dev = post_rots.to(device)
            post_trans_dev = post_trans.to(device)
            binimgs_dev = binimgs.to(device)
            lidar_bev_dev = lidar_bev.to(device)

            # --- Forward Pass ---
            model_output = model(imgs_dev, rots_dev, trans_dev, intrins_dev,
                                 post_rots_dev, post_trans_dev, lidar_bev_dev)
            depth_prob = None  # Initialize depth_prob
            if isinstance(model_output, tuple) and len(model_output) == 2:
                preds = model_output[0]
                depth_prob = model_output[1]  # Store depth if returned
            elif isinstance(model_output, torch.Tensor):
                preds = model_output
            else:
                print(
                    f"Error: Unexpected model output format: {type(model_output)}. Skipping batch.")
                continue

            # --- 1. Simple Metric Calculation (vs. binimgs) ---
            batch_loss = loss_fn(preds, binimgs_dev)
            if torch.isfinite(batch_loss):
                total_loss_simple += batch_loss.item() * B

            intersect_s, union_s, iou_s, fp_s, fn_s = get_batch_iou_flexible(
                preds, binimgs_dev)
            total_tp_simple += intersect_s
            total_union_simple += union_s
            total_fp_simple += fp_s
            total_fn_simple += fn_s

            # --- 2. Devkit Metric Calculation (vs. Rasterized Map) --- (Only if nusc is available)
            if nusc is not None and nusc_map_explorer is not None:
                for i in range(B):  # Process each sample in the batch
                    sample_token = sample_tokens[i]
                    pred_i = preds[i]  # Prediction for this sample (H, W)
                    devkit_gt_raster_i = None  # Initialize GT raster

                    # Get map location and cache MapAPI
                    try:
                        scene_record = nusc.get('scene', nusc.get(
                            'sample', sample_token)['scene_token'])
                        log_record = nusc.get('log', scene_record['log_token'])
                        log_token = log_record['token']
                        map_location = log_record['location']

                        if log_token not in map_apis:
                            # print(f"Loading NuScenesMap for location: {map_location}")
                            try:
                                map_apis[log_token] = NuScenesMap(
                                    dataroot=nusc.dataroot, map_name=map_location)
                            except Exception as map_e:
                                print(
                                    f"Warning: Error loading map {map_location}: {map_e}. Devkit eval skipped for this location.")
                                map_apis[log_token] = None  # Mark as failed
                    except Exception as e:
                        print(
                            f"Warning: Error getting map location for sample {sample_token}: {e}")
                        log_token = None  # Cannot determine map
                        map_apis[log_token] = None  # Avoid KeyError later

                    current_map_api = map_apis.get(log_token, None)
                    if current_map_api is None:
                        continue  # Skip devkit eval if map couldn't be loaded or determined

                    # Rasterize the ground truth map for this sample
                    devkit_gt_raster_i = rasterize_nusc_map(
                        nusc_map_explorer, current_map_api, sample_token, map_layers, grid_conf)

                    if devkit_gt_raster_i is None or devkit_gt_raster_i.sum() == 0:  # Error or empty map
                        continue

                    # Ensure GT is on the correct device
                    devkit_gt_raster_i = devkit_gt_raster_i.to(device)

                    # Compare prediction with devkit rasterized GT
                    intersect_d, union_d, iou_d, fp_d, fn_d = get_batch_iou_flexible(
                        pred_i.unsqueeze(0), devkit_gt_raster_i.unsqueeze(0))

                    # Accumulate devkit stats
                    total_tp_devkit += intersect_d
                    total_union_devkit += union_d
                    total_fp_devkit += fp_d
                    total_fn_devkit += fn_d
                    num_devkit_samples_ok += 1

            # --- 3. Visualization (Only for first batch with valid data) ---
            if not visualized_batch and writer is not None:
                try:
                    vis_idx = 0  # Visualize first sample
                    # Input Image (use original non-device tensor for CPU ops)
                    writer.add_image('val/input_image_front',
                                     imgs[vis_idx, 1].cpu(), global_step)

                    # Lidar BEV Input (use original non-device tensor)
                    lidar_bev_vis_val = lidar_bev[vis_idx].cpu().sum(
                        0, keepdim=True)
                    lidar_bev_vis_val = (lidar_bev_vis_val - lidar_bev_vis_val.min()) / (
                        lidar_bev_vis_val.max() - lidar_bev_vis_val.min() + 1e-6)
                    writer.add_image('val/input_lidar_bev',
                                     lidar_bev_vis_val, global_step)

                    # Simple GT BEV (use original non-device tensor)
                    writer.add_image(
                        'val/gt_bev_simple', binimgs[vis_idx].cpu().float(), global_step)

                    # Predicted BEV (use prediction tensor)
                    writer.add_image(
                        'val/pred_bev', torch.sigmoid(preds[vis_idx]).cpu(), global_step)

                    # Devkit GT BEV (if available)
                    # Check if devkit GT was processed for this sample
                    if nusc is not None and num_devkit_samples_ok > batchi * B + vis_idx:
                        # Re-rasterize for visualization (or store devkit_gt_raster_i if preferred)
                        sample_token_vis = sample_tokens[vis_idx]
                        scene_record_vis = nusc.get('scene', nusc.get(
                            'sample', sample_token_vis)['scene_token'])
                        log_record_vis = nusc.get(
                            'log', scene_record_vis['log_token'])
                        log_token_vis = log_record_vis['token']
                        current_map_api_vis = map_apis.get(log_token_vis, None)
                        if nusc_map_explorer and current_map_api_vis:
                            devkit_gt_v = rasterize_nusc_map(
                                nusc_map_explorer, current_map_api_vis, sample_token_vis, map_layers, grid_conf)
                            if devkit_gt_v is not None:
                                writer.add_image(
                                    'val/gt_bev_devkit', devkit_gt_v.unsqueeze(0), global_step)

                    # Depth Map (if available and params provided)
                    if depth_prob is not None and D_depth is not None and final_dim_vis is not None:
                        # Assumes CAM_FRONT is index 1
                        front_cam_depth_prob = depth_prob[vis_idx, 1].cpu()
                        depth_indices = torch.argmax(
                            front_cam_depth_prob, dim=0, keepdim=True)
                        vis_depth_map = depth_indices.float() / max(1, D_depth - 1)
                        vis_depth_map_resized = F.interpolate(vis_depth_map.unsqueeze(
                            0), size=final_dim_vis, mode='nearest').squeeze(0)
                        writer.add_image(
                            'val/depth_map_front_maxprob', vis_depth_map_resized, global_step)

                    visualized_batch = True  # Mark as visualized
                except Exception as vis_e:
                    print(
                        f"Warning: Error during validation visualization: {vis_e}")
                    visualized_batch = True  # Avoid retrying visualization if it fails once

            val_pbar.set_postfix(
                loss=f"{batch_loss.item():.4f}", iou_s=f"{iou_s:.4f}")

    # --- Calculate Final Metrics ---
    avg_loss_simple = total_loss_simple / num_samples if num_samples > 0 else 0

    # Simple metrics
    precision_s = total_tp_simple / (total_tp_simple + total_fp_simple + 1e-6)
    recall_s = total_tp_simple / (total_tp_simple + total_fn_simple + 1e-6)
    f1_s = 2 * (precision_s * recall_s) / (precision_s + recall_s + 1e-6)
    iou_s = total_tp_simple / (total_union_simple + 1e-6)

    # Devkit metrics
    precision_d = total_tp_devkit / \
        (total_tp_devkit + total_fp_devkit +
         1e-6) if num_devkit_samples_ok > 0 else 0
    recall_d = total_tp_devkit / \
        (total_tp_devkit + total_fn_devkit +
         1e-6) if num_devkit_samples_ok > 0 else 0
    f1_d = 2 * (precision_d * recall_d) / (precision_d +
                                           recall_d + 1e-6) if num_devkit_samples_ok > 0 else 0
    iou_d = total_tp_devkit / \
        (total_union_devkit + 1e-6) if num_devkit_samples_ok > 0 else 0

    print(
        f"Validation Complete. Samples: {num_samples}. Devkit Samples OK: {num_devkit_samples_ok}")
    print(
        f"  Simple Metrics (vs binimgs): Loss: {avg_loss_simple:.4f}, IoU: {iou_s:.4f}, P: {precision_s:.4f}, R: {recall_s:.4f}, F1: {f1_s:.4f}")
    if num_devkit_samples_ok > 0:
        print(
            f"  Devkit Metrics (vs rasterized {map_layers}): IoU: {iou_d:.4f}, P: {precision_d:.4f}, R: {recall_d:.4f}, F1: {f1_d:.4f}")
    else:
        print("  Devkit Metrics: Not calculated (no successful samples or map loading errors).")

    model.train()

    results = {
        'loss': avg_loss_simple,
        'simple_iou': iou_s,
        'simple_precision': precision_s,
        'simple_recall': recall_s,
        'simple_f1': f1_s,
    }
    if num_devkit_samples_ok > 0:
        results.update({
            'devkit_iou': iou_d,
            'devkit_precision': precision_d,
            'devkit_recall': recall_d,
            'devkit_f1': f1_d,
        })

    return results


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
