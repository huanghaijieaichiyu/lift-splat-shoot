"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from tqdm import tqdm
# 导入nuscenes相关库
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
# EvalBoxes is used for predictions, DetectionBox for ground truth loading (if needed directly)
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import json
from typing import Dict, List, Any
from torch.cuda.amp import autocast
import torch.nn.functional as F  # Add import for max_pool2d

# --- Constants for nuScenes Mapping ---

# Mapping from your model's output class IDs to official nuScenes detection names
# IMPORTANT: Verify this mapping matches your model's training and dataset setup.
# Background class (ID 0) should not be included here.
MODEL_ID_TO_NUSCENES_NAME = {
    0: 'car',
    1: 'truck',
    2: 'bus',
    3: 'trailer',
    4: 'construction_vehicle',
    5: 'pedestrian',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'traffic_cone',
    9: 'barrier'
}

# Mapping from nuScenes detection names to contiguous IDs (0-9 for 10 classes)
# Used internally by nuScenes evaluation if needed, but primarily we use names.
NUSCENES_NAME_TO_DETECTION_ID = {
    v: k-1 for k, v in MODEL_ID_TO_NUSCENES_NAME.items()}  # 0-indexed

# --- Helper Function for nuScenes Box Conversion (Vectorized) ---


def convert_pred_to_nuscenes_box(
    predictions_dict: Dict[str, Any],
    all_sample_tokens: List[str] | None
) -> EvalBoxes:
    """
    Converts a dictionary of concatenated predictions to the nuScenes EvalBoxes format.
    Ensures all sample tokens from the evaluation split are present in the output.

    Args:
        predictions_dict: Dictionary containing concatenated prediction arrays for
                          detections *above* the score threshold.
        all_sample_tokens: A list of all sample_token strings that were processed
                           in the evaluation split.

    Returns:
        EvalBoxes: A container holding nuScenes DetectionBox dictionaries grouped by sample_token.
                   Includes entries with empty lists for samples with no detections.
    """
    nuscenes_boxes = EvalBoxes()

    # Handle case where predictions_dict might be empty if *no* detections passed the threshold
    # across the entire dataset, but we still need to return EvalBoxes with empty lists for all tokens.
    has_valid_detections = (predictions_dict and
                            'sample_tokens' in predictions_dict and
                            predictions_dict['sample_tokens'])

    if has_valid_detections:
        # Extract data for valid detections
        scores = predictions_dict['box_scores']
        # Fix Linter Error: Convert cls_ids to numpy *before* astype
        cls_ids = np.array(predictions_dict['box_cls']).astype(int)
        xyz = predictions_dict['box_xyz']
        wlh = predictions_dict['box_wlh']
        rot_sincos = predictions_dict['box_rot_sincos']
        vel = predictions_dict['box_vel']
        # List of tokens corresponding ONLY to the valid detections
        valid_detection_sample_tokens = predictions_dict['sample_tokens']
        num_dets = len(scores)

        if num_dets > 0:
            print(f"Vectorizing conversion for {num_dets} valid detections...")

            # Vectorized calculations
            yaws = np.arctan2(rot_sincos[:, 0], rot_sincos[:, 1])
            detection_names = np.array(
                [MODEL_ID_TO_NUSCENES_NAME.get(c, None) for c in cls_ids])
            valid_mask = detection_names != None

            if np.any(valid_mask):
                # Apply mask to filter detections with unknown class IDs
                scores = scores[valid_mask]
                xyz = xyz[valid_mask]
                wlh = wlh[valid_mask]
                yaws = yaws[valid_mask]
                vel = vel[valid_mask]
                detection_names = detection_names[valid_mask]
                # Filter the list of sample tokens for valid detections
                valid_tokens_np = np.array(valid_detection_sample_tokens)
                # Back to list after filtering
                sample_tokens_for_boxes = valid_tokens_np[valid_mask].tolist()
                num_valid_dets = len(scores)

                print(
                    f"Processing {num_valid_dets} actual boxes after class mapping...")

                # Create DetectionBox objects for valid detections and group by token
                for i in tqdm(range(num_valid_dets), desc="Creating nuScenes boxes", leave=False):
                    token = sample_tokens_for_boxes[i]
                    quat = Quaternion(axis=[0, 0, 1], angle=yaws[i])
                    detection_box = DetectionBox(
                        sample_token=token,
                        translation=xyz[i].tolist(),
                        size=wlh[i].tolist(),
                        rotation=quat.elements.tolist(),
                        velocity=vel[i].tolist()[:2],
                        detection_name=detection_names[i],
                        detection_score=float(scores[i]),
                        attribute_name=''
                    )
                    if token not in nuscenes_boxes.boxes:
                        nuscenes_boxes.boxes[token] = []
                    nuscenes_boxes.boxes[token].append(detection_box)
            else:
                print("Warning: No valid detections remained after class mapping.")
        else:
            print("Warning: predictions_dict contained zero detections.")

    # Ensure all sample tokens from the split are present in the final EvalBoxes object
    if all_sample_tokens:
        print(
            f"Ensuring all {len(all_sample_tokens)} sample tokens are present in the results...")
        # Use set for efficient lookup
        original_tokens_set = set(all_sample_tokens)
        tokens_with_boxes = set(nuscenes_boxes.boxes.keys())

        missing_tokens = original_tokens_set - tokens_with_boxes
        if missing_tokens:
            print(
                f"Adding {len(missing_tokens)} sample tokens with empty prediction lists.")
            for token in missing_tokens:
                nuscenes_boxes.boxes[token] = []  # Add token with empty list

        # Verify final count matches expected number of samples
        final_token_count = len(nuscenes_boxes.boxes.keys())
        if final_token_count != len(original_tokens_set):
            print(
                f"Warning: Final token count ({final_token_count}) doesn't match expected ({len(original_tokens_set)})! Check for duplicates or other issues.")
        else:
            print(
                f"Final results confirmed to contain {final_token_count} sample tokens.")

    else:
        print("Warning: `all_sample_tokens` list was not provided to `convert_pred_to_nuscenes_box`. Cannot guarantee all samples are included.")

    return nuscenes_boxes


# --- Main Evaluation Function using nuScenes Devkit (Updated Signature) ---

def evaluate_with_nuscenes(
    # Accepts the dict of numpy arrays
    predictions_dict: Dict[str, np.ndarray | List[str]],
    nuscenes_version: str,
    nuscenes_dataroot: str,
    eval_set: str,  # e.g., 'val', 'mini_val', 'test'
    output_dir: str,
    verbose: bool = True,
    # Fix type hint for optional list
    all_sample_tokens: List[str] | None = None
) -> Dict[str, Any]:
    """
    Performs 3D object detection evaluation using the official nuScenes devkit.
    Accepts predictions as a dictionary of concatenated NumPy arrays.

    Args:
        predictions_dict: Dictionary containing concatenated prediction arrays and sample tokens
                          for detections *above* the score threshold.
        nuscenes_version: nuScenes dataset version (e.g., 'v1.0-mini', 'v1.0-trainval').
        nuscenes_dataroot: Path to the nuScenes dataset root directory.
        eval_set: The split to evaluate on (e.g., 'val', 'mini_val').
        output_dir: Directory to save evaluation results and temporary files.
        verbose: Whether to print detailed evaluation progress and results.
        all_sample_tokens: A list of all sample_token strings that were processed
                           in the evaluation split. Needed to ensure results JSON is complete.

    Returns:
        Dict[str, Any]: A dictionary containing the official nuScenes evaluation metrics.
    """

    # 1. Convert predictions dictionary to nuScenes format (vectorized)
    print("Converting predictions to nuScenes format...")
    # Pass the complete token list to ensure all samples are included
    nusc_boxes = convert_pred_to_nuscenes_box(
        predictions_dict, all_sample_tokens)
    print(
        f"Conversion complete. Results generated for {len(nusc_boxes.boxes)} samples.")

    # Check if any boxes were actually generated across all samples
    total_boxes = sum(len(boxes) for boxes in nusc_boxes.boxes.values())
    if total_boxes == 0:
        print("Error: No valid prediction boxes were generated after conversion. Check model output and conversion logic.")
        # Return default/empty metrics
        return {
            'mAP': 0.0, 'NDS': 0.0, 'mATE': float('inf'), 'mASE': float('inf'),
            'mAOE': float('inf'), 'mAVE': float('inf'), 'mAAE': float('inf'),
            'per_class_AP': {name: 0.0 for name in MODEL_ID_TO_NUSCENES_NAME.values()},
            'per_class_TP_Errors': {err: {name: 0.0 for name in MODEL_ID_TO_NUSCENES_NAME.values()}
                                    for err in ['trans_err', 'scale_err', 'orient_err', 'vel_err', 'attr_err']}
        }

    # 2. Prepare nuScenes evaluation configuration
    # Use the standard config for nuScenes detection challenge (CVPR 2019)
    cfg = config_factory('detection_cvpr_2019')
    # Adjust config parameters if needed, e.g., distance thresholds
    # cfg.dist_ths = [0.5, 1.0, 2.0, 4.0] # Example

    # 3. Initialize NuScenes object to access ground truth
    print(
        f"Initializing NuScenes (version: {nuscenes_version}, dataroot: {nuscenes_dataroot})...")
    try:
        nusc = NuScenes(version=nuscenes_version,
                        dataroot=nuscenes_dataroot, verbose=verbose)
    except AssertionError as e:
        print(f"Error initializing NuScenes: {e}")
        print(
            "Please ensure the dataroot path is correct and the specified version exists.")
        raise

    # 4. Prepare output directory and save predictions JSON
    eval_output_dir = os.path.join(output_dir, 'nuscenes_eval_results')
    os.makedirs(eval_output_dir, exist_ok=True)

    # nuScenes evaluation requires predictions saved as a JSON file
    res_path = os.path.join(eval_output_dir, 'results_nuscenes.json')
    meta = {
        "use_camera": True,   # Indicates camera-based detection
        "use_lidar": False,  # Set based on your model's input modalities
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    results_json = {'meta': meta, 'results': nusc_boxes.serialize()}

    print(f"Saving predictions to {res_path}...")
    try:
        with open(res_path, 'w') as f:
            json.dump(results_json, f, indent=4)  # Use indent for readability
        print("Predictions saved successfully.")
    except Exception as e:
        print(f"Error saving predictions JSON: {e}")
        raise

    # 5. Run nuScenes Evaluation
    if verbose:
        print(f"Running nuScenes detection evaluation for set: {eval_set}...")
        print(f"Using configuration: detection_cvpr_2019")
        print(f"Output will be stored in: {eval_output_dir}")

    # Instantiate NuScenesEval
    # Make sure the eval_set matches the split used in your valloader
    # Common sets: 'train', 'val', 'test', 'mini_train', 'mini_val'
    try:
        nusc_eval = NuScenesEval(
            nusc,
            config=cfg,
            result_path=res_path,
            eval_set=eval_set,
            output_dir=eval_output_dir,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error initializing NuScenesEval: {e}")
        print(
            f"Check if eval_set '{eval_set}' is valid for version '{nuscenes_version}'.")
        raise

    # Run evaluation - this computes mAP, NDS, and other metrics
    # render_curves=True generates PR curve plots
    print("Starting NuScenesEval.main()...")
    metrics_summary = nusc_eval.main(render_curves=False)
    print("NuScenesEval.main() finished.")

    # 6. Parse and return metrics
    # nuScenes eval returns metrics in a specific structure. We extract key ones.
    # Use .get() to handle potentially missing keys, returning NaN if missing
    metrics_dict = {
        'mAP': metrics_summary.get('mean_ap', float('nan')),
        'mATE': metrics_summary.get('mean_trans_err', float('nan')),
        'mASE': metrics_summary.get('mean_scale_err', float('nan')),
        'mAOE': metrics_summary.get('mean_orient_err', float('nan')),
        'mAVE': metrics_summary.get('mean_vel_err', float('nan')),
        'mAAE': metrics_summary.get('mean_attr_err', float('nan')),
        'NDS': metrics_summary.get('nd_score', float('nan')),
        # AP per class might also be missing if no GT/Preds for any class
        'per_class_AP': metrics_summary.get('mean_dist_aps', {}),
        # TP errors per class might also be missing
        'per_class_TP_Errors': metrics_summary.get('tp_errors', {}),
    }

    if verbose:
        print("===== Official nuScenes Evaluation Results =====")
        print(f"mAP: {metrics_dict['mAP']:.4f}")
        print(f"NDS: {metrics_dict['NDS']:.4f}")
        print(f"mATE (Translation): {metrics_dict['mATE']:.4f} (meters)")
        print(f"mASE (Scale): {metrics_dict['mASE']:.4f} (1-IoU)")
        print(f"mAOE (Orientation): {metrics_dict['mAOE']:.4f} (radians)")
        print(f"mAVE (Velocity): {metrics_dict['mAVE']:.4f} (m/s)")
        print(f"mAAE (Attribute): {metrics_dict['mAAE']:.4f} (1-acc)")
        print("----- Per-class AP -----")
        for class_name, ap in metrics_dict['per_class_AP'].items():
            print(f"{class_name}: {ap:.4f}")
        # Optionally print TP errors per class if needed
        # print("----- Per-class TP Errors -----")
        # for tp_name, class_errors in metrics_dict['per_class_TP_Errors'].items():
        #     print(f"--- {tp_name} ---")
        #     for class_name, error in class_errors.items():
        #         print(f"{class_name}: {error:.4f}")

    return metrics_dict


# --- Modified decode functions ---

def _gather_feat(feat, ind, mask=None):
    """Gather feature based on index"""
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

# --- Function to perform NMS on heatmap ---


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep  # Return heatmap with non-maximum suppressed values zeroed out


def _topk(scores, K=100):
    batch, cat, height, width = scores.size()

    # Perform NMS first to reduce redundant peaks
    scores = _nms(scores)

    # Find top K scores across all classes and spatial locations
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # Find top K scores *per category*
    topk_score, topk_ind = torch.topk(
        scores.view(batch, cat, -1).permute(0, 2, 1), K)
    # topk_score: [B, K] (top scores regardless of class)
    # topk_ind: [B, K] (class index for each top score)
    topk_clses = topk_ind.int()
    # Get the spatial indices corresponding to these top scores
    # Need to map topk_score back to original spatial index and class index
    # Re-calculate indices based on topk scores across all classes
    topk_inds = _gather_feat(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, K)

    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

# --- Decode function modified for CenterPoint ---


def decode_predictions(
    preds_dict: Dict[str, torch.Tensor],
    device: torch.device,  # Device is implicitly handled by input tensors
    score_thresh: float,
    grid_conf: Dict[str, Any],
    K: int = 100  # Number of top predictions to consider
):
    """
    Decodes predictions from CenterPoint heads into 3D bounding boxes.

    Args:
        preds_dict (Dict[str, torch.Tensor]): Dictionary of raw predictions from the model.
            Expected keys: 'heatmap', 'offset', 'z_coord', 'dimension', 'rotation', 'velocity'.
        score_thresh (float): Threshold to filter detections based on heatmap score.
        grid_conf (Dict[str, Any]): Grid configuration (xbound, ybound, dx, bx).
        K (int): Maximum number of detections to return per sample.

    Returns:
        List[Dict[str, torch.Tensor]]: A list of dictionaries, one per batch sample.
            Each dictionary contains keys for 'box_scores', 'box_cls', 'box_xyz', 
            'box_wlh', 'box_rot_sincos', 'box_vel'.
    """
    heatmap = preds_dict['heatmap'].sigmoid_()
    offset = preds_dict['offset']
    z_coord = preds_dict['z_coord']
    dimension = preds_dict['dimension']  # Log scale
    rotation = preds_dict['rotation']  # sin, cos
    velocity = preds_dict.get('velocity')  # Optional

    B, C, H, W = heatmap.shape
    dx_bev = torch.tensor(grid_conf['xbound'][2], device=heatmap.device)
    dy_bev = torch.tensor(grid_conf['ybound'][2], device=heatmap.device)
    bx_bev = torch.tensor(
        grid_conf['xbound'][0] + grid_conf['xbound'][2] / 2.0, device=heatmap.device)
    by_bev = torch.tensor(
        grid_conf['ybound'][0] + grid_conf['ybound'][2] / 2.0, device=heatmap.device)

    # 1. Find top K peaks in the heatmap after NMS
    scores, inds, clses, ys, xs = _topk(heatmap, K=K)
    # scores, inds, clses, ys, xs are all [B, K]

    # 2. Gather regression predictions at peak locations
    # Transpose and gather needs indices in [B, K] format
    offset = _transpose_and_gather_feat(offset, inds)  # [B, K, 2]
    z_coord = _transpose_and_gather_feat(z_coord, inds)  # [B, K, 1]
    dimension = _transpose_and_gather_feat(dimension, inds)  # [B, K, 3]
    rotation = _transpose_and_gather_feat(rotation, inds)  # [B, K, 2]
    if velocity is not None:
        velocity = _transpose_and_gather_feat(velocity, inds)  # [B, K, 2]
    else:  # Create dummy zero velocity if head doesn't exist
        velocity = torch.zeros_like(offset)  # [B, K, 2]

    # 3. Convert grid coordinates + offset to original coordinates
    # Add offset to integer grid coordinates
    xs = xs.view(B, K, 1) + offset[:, :, 0:1]
    ys = ys.view(B, K, 1) + offset[:, :, 1:2]

    # Convert grid coords to meters
    # x_m = (xs * dx_bev) + bx_bev - dx_bev / 2.0 # Center of the bin adjustment
    # y_m = (ys * dy_bev) + by_bev - dy_bev / 2.0 # Center of the bin adjustment
    # Simpler: Add grid center offset (bx_bev, by_bev includes half-bin offset)
    x_m = xs * dx_bev + bx_bev
    y_m = ys * dy_bev + by_bev

    # Combine with z coordinate
    xyz = torch.cat([x_m, y_m, z_coord], dim=2)  # [B, K, 3]

    # 4. Decode dimensions (exp for log scale)
    wlh = dimension.exp()  # [B, K, 3] - w, l, h

    # 5. Prepare outputs per batch item
    detections = []
    for b in range(B):
        scores_b = scores[b]  # [K]
        clses_b = clses[b]  # [K]
        xyz_b = xyz[b]  # [K, 3]
        wlh_b = wlh[b]  # [K, 3]
        rotation_b = rotation[b]  # [K, 2]
        velocity_b = velocity[b]  # [K, 2]

        # Apply score threshold
        keep_mask = scores_b >= score_thresh
        if keep_mask.sum() == 0:
            # Add empty dict if no detections pass threshold
            detections.append({
                'box_scores': torch.empty(0, device=heatmap.device),
                'box_cls': torch.empty(0, dtype=torch.long, device=heatmap.device),
                'box_xyz': torch.empty(0, 3, device=heatmap.device),
                'box_wlh': torch.empty(0, 3, device=heatmap.device),
                'box_rot_sincos': torch.empty(0, 2, device=heatmap.device),
                'box_vel': torch.empty(0, 2, device=heatmap.device)
            })
            continue

        # Filter based on score
        scores_b = scores_b[keep_mask]
        clses_b = clses_b[keep_mask]
        xyz_b = xyz_b[keep_mask]
        wlh_b = wlh_b[keep_mask]
        rotation_b = rotation_b[keep_mask]
        velocity_b = velocity_b[keep_mask]

        detections.append({
            'box_scores': scores_b,
            'box_cls': clses_b,
            'box_xyz': xyz_b,
            'box_wlh': wlh_b,
            'box_rot_sincos': rotation_b,
            'box_vel': velocity_b
        })

    return detections

# --- Original decode_predictions (KEEP FOR REFERENCE OR REMOVE) ---
# def decode_predictions_original(...):
#    ... (Previous implementation) ...

# --- Function to decode GT targets (Optional, might be useful for debugging/vis) ---


# --- Deprecated/Removed Functions ---

# def compute_map(...) - Replaced by evaluate_with_nuscenes
# def calculate_map(...) - Replaced by evaluate_with_nuscenes
# def calculate_ap_for_class(...) - Replaced by evaluate_with_nuscenes
# def compute_3d_iou(...) - Replaced by nuScenes official IoU calculation within NuScenesEval


# --- Main Evaluation Entry Point ---

def evaluate_3d_detection(
    modelf: str,
    version: str = 'v1.0-mini',
    dataroot: str = '/data/nuscenes',  # IMPORTANT: Set correct path
    gpuid: int = 0,
    # Image dimensions (used for data loading config)
    H: int = 900, W: int = 1600,
    # Data Augmentation settings (match training)
    resize_lim: tuple = (0.193, 0.225),
    final_dim: tuple = (128, 352),
    bot_pct_lim: tuple = (0.0, 0.22),
    rot_lim: tuple = (-5.4, 5.4),
    rand_flip: bool = False,
    ncams: int = 6,  # Default nuScenes setup uses 6 cameras
    # Grid configuration (match training)
    xbound: list = [-50.0, 50.0, 0.5],  # [min, max, resolution]
    ybound: list = [-50.0, 50.0, 0.5],
    zbound: list = [-5.0, 3.0, 0.5],    # Corrected: [min, max, resolution]
    dbound: list = [4.0, 45.0, 1.0],    # Depth bins [min, max, step]
    # Dataloader settings
    bsz: int = 4,
    nworkers: int = 4,  # Adjust based on your system
    # Model settings
    # Number of classes model predicts (excluding background)
    num_classes: int = 10,
    # Type of model architecture ('lss', 'beve', 'fusion', ...)
    model_type: str = 'beve',
    # Evaluation settings
    amp: bool = True,  # Use Automatic Mixed Precision
    output_dir: str = './eval_output',  # Directory for saving results
    verbose: bool = True,
    score_thresh: float = 0.01,  # Add score threshold parameter
):
    """
   评估使用已保存的检查点和官方 nuScenes 评估协议的 3D 对象检测模型。

    Args:
        checkpoint_path: 模型检查点 (.pth) 文件的路径。
        version: nuScenes 数据集版本（例如，"v1.0-mini"，"v1.0-trainval"）。
        dataroot: nuScenes 数据集的根目录。
        gpuid: 用于评估的 GPU ID（-1 表示 CPU）。
        H, W: 原始图像的高度和宽度。
        resize_lim, final_dim, bot_pct_lim, rot_lim, rand_flip: 数据增强参数
                                                               (应与训练匹配)。
        ncams: 每个样本使用的相机数量。
        xbound, ybound, zbound, dbound: BEV 网格配置。
        bsz: 验证数据加载器的批量大小。
        nworkers: 数据加载器的工作进程数。
        num_classes: 模型预测的对象类别数量（不包括背景）。
        model_type: 要加载的模型的架构类型。
        amp: 是否在推理期间使用自动混合精度。
        output_dir: 将保存评估结果（JSON、日志）的目录。
        verbose: 如果为 True，则打印详细的进度和结果。
        score_thresh: 保持检测的最小分数阈值。

    Returns:
        Dict[str, Any]: 包含官方 nuScenes 指标（mAP、NDS 等）的字典。

    """
    from .models import compile_model
    from .data import compile_data  # Ensure this returns sample_tokens

    # --- Setup ---
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    grid_conf = {
        'xbound': xbound, 'ybound': ybound, 'zbound': zbound, 'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim, 'final_dim': final_dim, 'rot_lim': rot_lim,
        'H': H, 'W': W, 'rand_flip': rand_flip, 'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'][:ncams],  # Select correct cameras
        'Ncams': ncams,
    }

    # Determine evaluation set name based on version string
    eval_set = 'val' if 'mini' not in version else 'mini_val'
    print(
        f"Evaluating on nuScenes dataset version: {version}, split: {eval_set}")

    # --- Data Loading ---
    print("Loading validation data...")
    # IMPORTANT: Ensure your compile_data function and the specified parser
    # return the sample_token for each sample in the batch.
    # The parser name 'nuScenes' is assumed here. Modify if yours is different.
    try:
        # Ensure compile_data does not return trainloader if only valloader is needed
        # Assuming compile_data can return just valloader based on flags or arguments
        # Modify this call if compile_data always returns both
        _, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                    grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                    parser_name='nuScenes',  # Ensure this parser yields sample_token
                                    # shuffle_train=False # Fix Linter Error: Remove unknown argument
                                    )
        print("Validation data loaded.")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure 'compile_data' with parser_name='nuScenes' is correctly implemented")
        print(
            "and returns (trainloader, valloader) where valloader yields batches containing")
        print(
            "(imgs, rots, trans, intrins, post_rots, post_trans, target_maps, sample_tokens)")
        raise

    # --- Model Loading ---
    print(f"Loading model checkpoint from: {modelf}")
    try:
        checkpoint = torch.load(
            modelf, map_location='cpu')  # Load to CPU first
        print("Checkpoint loaded.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {modelf}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Compile model architecture - outC is handled internally by BEVENet based on num_classes
    # Pass the correct model_type specified by the user
    print(
        f"Compiling model architecture: {model_type} with {num_classes} classes...")
    model = compile_model(grid_conf, data_aug_conf,
                          outC=num_classes + 9 + 1,  # Pass expected outC, model will adapt
                          model=model_type,
                          num_classes=num_classes)

    if model is None:
        raise ValueError(
            f"Model compilation failed for type '{model_type}'. Check 'compile_model'.")
    print("Model compiled.")

    # Load model state dictionary
    print("Loading model weights...")
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint  # Assume checkpoint is the state_dict itself

    # Handle potential DataParallel/DistributedDataParallel prefixes
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            cleaned_state_dict[k] = v

    try:
        model.load_state_dict(cleaned_state_dict)
        print("Model weights loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to architecture mismatch or incorrect checkpoint format.")
        # Optionally print model keys and checkpoint keys for comparison
        # print("Model Keys:", list(model.state_dict().keys()))
        # print("Checkpoint Keys:", list(cleaned_state_dict.keys()))
        raise

    model.to(device)
    model.eval()  # Set model to evaluation mode

    # --- Run Inference and Collect Predictions (Modified for Batch Processing) ---
    print("Starting inference on validation set...")
    all_pred_scores = []
    all_pred_cls = []
    all_pred_xyz = []
    all_pred_wlh = []
    all_pred_rot_sincos = []
    all_pred_vel = []
    all_pred_sample_tokens = []
    all_tokens_processed = []  # New list to store *all* sample tokens processed

    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(valloader, desc="Evaluating", disable=not verbose)):
            # Ensure the batch structure matches expectations (including sample_token)
            try:
                # Adjust unpacking based on what valloader yields
                imgs, rots, trans, intrins, post_rots, post_trans, target_maps, sample_tokens = data_batch
            except ValueError as e:
                print(
                    f"Error unpacking data batch {i}. Expected 8 items, got {len(data_batch)}.")
                print(f"Original error: {e}")
                print("Make sure your dataloader yields: ")
                print(
                    "(imgs, rots, trans, intrins, post_rots, post_trans, target_maps, sample_tokens)")
                raise ValueError("Dataloader batch format mismatch.") from e

            # Add all sample tokens from this batch to the master list
            all_tokens_processed.extend(sample_tokens)

            # Move data to device
            imgs = imgs.to(device)
            rots = rots.to(device)
            trans = trans.to(device)
            intrins = intrins.to(device)
            post_rots = post_rots.to(device)
            post_trans = post_trans.to(device)
            # Targets are not used for inference

            # Perform inference with AMP if enabled
            if amp and device.type == 'cuda':
                # Fix Linter Error: Correct autocast usage with explicit import
                # Removed device_type as it's inferred for cuda
                with autocast(dtype=torch.float16):
                    preds_dict = model(imgs, rots, trans,
                                       intrins, post_rots, post_trans)
            else:
                preds_dict = model(imgs, rots, trans,
                                   intrins, post_rots, post_trans)

            # Call decode_predictions with the grid configuration
            batch_dets_list = decode_predictions(
                preds_dict, device, score_thresh, grid_conf
            )

            # Collect tensors and map detections to sample tokens
            for sample_idx, dets_dict in enumerate(batch_dets_list):
                num_dets_in_sample = dets_dict['box_scores'].shape[0]
                if num_dets_in_sample > 0:
                    all_pred_scores.append(dets_dict['box_scores'])
                    all_pred_cls.append(dets_dict['box_cls'])
                    all_pred_xyz.append(dets_dict['box_xyz'])
                    all_pred_wlh.append(dets_dict['box_wlh'])
                    all_pred_rot_sincos.append(dets_dict['box_rot_sincos'])
                    all_pred_vel.append(dets_dict['box_vel'])
                    # Repeat the sample token for each detection in this sample
                    all_pred_sample_tokens.extend(
                        [sample_tokens[sample_idx]] * num_dets_in_sample)

    # Concatenate all collected tensors into a dictionary
    if not all_pred_scores:
        print("Warning: No detections found across the entire validation set.")
        all_predictions_dict = {}  # Use the new name - dict will be empty
    else:
        # Store as numpy arrays directly after concatenation and moving to CPU
        all_predictions_dict = {
            'box_scores': torch.cat(all_pred_scores).cpu().numpy(),
            'box_cls': torch.cat(all_pred_cls).cpu().numpy(),
            'box_xyz': torch.cat(all_pred_xyz).cpu().numpy(),
            'box_wlh': torch.cat(all_pred_wlh).cpu().numpy(),
            'box_rot_sincos': torch.cat(all_pred_rot_sincos).cpu().numpy(),
            'box_vel': torch.cat(all_pred_vel).cpu().numpy(),
            'sample_tokens': all_pred_sample_tokens  # Keep as list
        }
        print(
            f"Inference complete. Collected {len(all_predictions_dict['sample_tokens'])} total detections across samples.")

    # --- Run Official nuScenes Evaluation ---
    print("Starting official nuScenes evaluation...")
    # Construct the full version string required by NuScenes class
    full_version = f'v1.0-{version}'
    # Updated call to evaluate_with_nuscenes
    results_dict = evaluate_with_nuscenes(
        predictions_dict=all_predictions_dict,
        nuscenes_version=full_version,  # Pass the constructed full version string
        nuscenes_dataroot=dataroot,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
        all_sample_tokens=all_tokens_processed  # Pass the complete list of tokens
    )
    print("Evaluation finished.")

    return results_dict
# --- End of File ---
