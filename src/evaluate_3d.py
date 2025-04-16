"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import numpy as np
from tqdm import tqdm


def compute_map(model, valloader, device, amp=True):
    """
    计算模型在验证集上的mAP
    Args:
        model: 模型
        valloader: 验证数据加载器
        device: 设备
        amp: 是否使用混合精度
    Returns:
        float: mAP值
    """
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for imgs, rots, trans, intrins, post_rots, post_trans, targets in tqdm(valloader, desc="Evaluating mAP"):
            # 使用上下文管理器进行混合精度计算
            if amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    preds = model(imgs.to(device),
                                  rots.to(device),
                                  trans.to(device),
                                  intrins.to(device),
                                  post_rots.to(device),
                                  post_trans.to(device),
                                  )
            else:
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )

            # 收集预测和目标
            batch_dets = decode_predictions(preds, device)
            batch_gts = decode_targets(targets, device)

            all_predictions.extend(batch_dets)
            all_targets.extend(batch_gts)

    # 计算mAP
    mean_ap = calculate_map(all_predictions, all_targets)
    return mean_ap


def decode_predictions(preds, device):
    """
    解码模型预测结果为3D检测框
    Args:
        preds: 模型预测
        device: 设备
    Returns:
        list: 包含预测结果的列表
    """
    batch_size = preds['cls_pred'].shape[0]
    predictions = []

    for b in range(batch_size):
        # 获取当前批次的预测
        cls_pred = preds['cls_pred'][b]  # [C, H, W]
        reg_pred = preds['reg_pred'][b]  # [9, H, W]
        iou_pred = preds['iou_pred'][b]  # [1, H, W]

        # 获取分类分数
        cls_scores, cls_ids = torch.max(torch.softmax(
            cls_pred, dim=0), dim=0)  # [H, W], [H, W]

        # 获取IoU分数
        iou_scores = torch.sigmoid(iou_pred.squeeze(0))  # [H, W]

        # 合并分数 = 分类分数 * IoU分数
        scores = cls_scores * iou_scores  # [H, W]

        # 过滤掉背景 (类别ID=0)
        mask = cls_ids > 0  # [H, W]

        if mask.sum() > 0:
            # 获取非背景的预测
            filtered_cls_ids = cls_ids[mask]  # [N]
            filtered_scores = scores[mask]  # [N]

            # 获取回归预测值
            h, w = mask.shape
            # 修复meshgrid警告，添加indexing参数
            ys, xs = torch.meshgrid(torch.arange(
                h, device=device), torch.arange(w, device=device), indexing='ij')
            xs, ys = xs[mask], ys[mask]  # [N], [N]

            reg_pred = reg_pred.permute(1, 2, 0)  # [H, W, 9]
            filtered_reg = reg_pred[mask]  # [N, 9]

            # 解析回归值: x, y, z, w, l, h, sin(θ), cos(θ), 速度
            dets = {
                'box_cls': filtered_cls_ids,               # [N] 类别ID
                'box_scores': filtered_scores,             # [N] 分数
                'box_reg': filtered_reg,                   # [N, 9] 回归值
                'box_xyz': filtered_reg[:, :3],            # [N, 3] 中心点坐标
                'box_wlh': filtered_reg[:, 3:6],           # [N, 3] 尺寸
                # [N, 1] 旋转角
                'box_rot': torch.atan2(filtered_reg[:, 6], filtered_reg[:, 7]).unsqueeze(1),
                'box_vel': filtered_reg[:, 8].unsqueeze(1)  # [N, 1] 速度
            }

            predictions.append(dets)
        else:
            # 没有有效检测
            empty_dets = {
                'box_cls': torch.zeros(0, dtype=torch.long, device=device),
                'box_scores': torch.zeros(0, device=device),
                'box_reg': torch.zeros(0, 9, device=device),
                'box_xyz': torch.zeros(0, 3, device=device),
                'box_wlh': torch.zeros(0, 3, device=device),
                'box_rot': torch.zeros(0, 1, device=device),
                'box_vel': torch.zeros(0, 1, device=device)
            }
            predictions.append(empty_dets)

    return predictions


def decode_targets(targets_list, device):
    """
    解码目标数据为3D检测框
    Args:
        targets_list: 目标数据列表，包含 [cls_map, reg_map, reg_weight, iou_map]
        device: 设备
    Returns:
        list: 包含目标数据的列表
    """
    # 解包目标列表
    cls_map = targets_list[0].to(device)        # 类别地图
    reg_map = targets_list[1].to(device)        # 回归地图
    reg_weight = targets_list[2].to(device)     # 回归权重
    iou_map = targets_list[3].to(device)        # IoU地图

    batch_size = cls_map.shape[0]
    gt_list = []

    for b in range(batch_size):
        # 获取当前批次的目标
        cls_targets = cls_map[b]  # [H, W]
        reg_targets = reg_map[b]  # [9, H, W]

        # 过滤掉背景 (类别ID=0)
        mask = cls_targets > 0  # [H, W]

        if mask.sum() > 0:
            # 获取非背景的目标
            filtered_cls_ids = cls_targets[mask]  # [N]

            # 获取回归目标值
            reg_targets = reg_targets.permute(1, 2, 0)  # [H, W, 9]
            filtered_reg = reg_targets[mask]  # [N, 9]

            # 解析回归值: x, y, z, w, l, h, sin(θ), cos(θ), 速度
            gt = {
                'box_cls': filtered_cls_ids,               # [N] 类别ID
                'box_reg': filtered_reg,                   # [N, 9] 回归值
                'box_xyz': filtered_reg[:, :3],            # [N, 3] 中心点坐标
                'box_wlh': filtered_reg[:, 3:6],           # [N, 3] 尺寸
                # [N, 1] 旋转角
                'box_rot': torch.atan2(filtered_reg[:, 6], filtered_reg[:, 7]).unsqueeze(1),
                'box_vel': filtered_reg[:, 8].unsqueeze(1)  # [N, 1] 速度
            }

            gt_list.append(gt)
        else:
            # 没有有效目标
            empty_gt = {
                'box_cls': torch.zeros(0, dtype=torch.long, device=device),
                'box_reg': torch.zeros(0, 9, device=device),
                'box_xyz': torch.zeros(0, 3, device=device),
                'box_wlh': torch.zeros(0, 3, device=device),
                'box_rot': torch.zeros(0, 1, device=device),
                'box_vel': torch.zeros(0, 1, device=device)
            }
            gt_list.append(empty_gt)

    return gt_list


def calculate_map(predictions, targets, iou_thresh=0.5, num_classes=10):
    """
    计算平均精度 (mAP)
    Args:
        predictions: 预测结果列表
        targets: 目标数据列表
        iou_thresh: IoU阈值
        num_classes: 类别数量
    Returns:
        float: mAP值
    """
    aps = []

    # 按类别计算AP
    for cls_id in range(1, num_classes):  # 忽略背景类(0)
        all_preds = []
        all_gts = []

        # 收集所有预测和目标
        for pred, gt in zip(predictions, targets):
            # 过滤当前类别的预测
            pred_mask = pred['box_cls'] == cls_id
            pred_boxes = {
                'box_xyz': pred['box_xyz'][pred_mask],
                'box_wlh': pred['box_wlh'][pred_mask],
                'box_rot': pred['box_rot'][pred_mask],
                'box_scores': pred['box_scores'][pred_mask]
            }
            all_preds.append(pred_boxes)

            # 过滤当前类别的目标
            gt_mask = gt['box_cls'] == cls_id
            gt_boxes = {
                'box_xyz': gt['box_xyz'][gt_mask],
                'box_wlh': gt['box_wlh'][gt_mask],
                'box_rot': gt['box_rot'][gt_mask]
            }
            all_gts.append(gt_boxes)

        # 如果没有该类别的目标，则跳过
        if sum(len(gt['box_xyz']) for gt in all_gts) == 0:
            continue

        # 计算该类别的AP
        ap = calculate_ap_for_class(all_preds, all_gts, iou_thresh)
        aps.append(ap)

    # 计算mAP
    if len(aps) > 0:
        mean_ap = sum(aps) / len(aps)
    else:
        mean_ap = 0.0

    return mean_ap


def calculate_ap_for_class(predictions, targets, iou_thresh=0.5):
    """
    计算单个类别的AP
    Args:
        predictions: 该类别的预测结果
        targets: 该类别的目标数据
        iou_thresh: IoU阈值
    Returns:
        float: AP值
    """
    # 收集所有预测结果
    all_scores = []
    all_matches = []

    # 遍历每个样本
    for pred, gt in zip(predictions, targets):
        scores = pred['box_scores']
        pred_boxes = torch.cat([
            pred['box_xyz'],
            pred['box_wlh'],
            pred['box_rot']
        ], dim=1)  # [N, 7] - x, y, z, w, l, h, θ

        gt_boxes = torch.cat([
            gt['box_xyz'],
            gt['box_wlh'],
            gt['box_rot']
        ], dim=1) if len(gt['box_xyz']) > 0 else None  # [M, 7] - x, y, z, w, l, h, θ

        # 如果没有预测或没有目标
        if len(scores) == 0 or gt_boxes is None or len(gt_boxes) == 0:
            if len(scores) > 0:
                all_scores.extend(scores.cpu().numpy())
                all_matches.extend([0] * len(scores))
            continue

        # 计算IoU
        ious = compute_3d_iou(pred_boxes, gt_boxes)  # [N, M]

        # 对每个预测找到最大IoU的目标
        max_ious, max_idx = ious.max(dim=1)

        # 确定匹配
        is_match = max_ious >= iou_thresh

        # 处理重复匹配
        gt_matched = torch.zeros(
            len(gt_boxes), dtype=torch.bool, device=pred_boxes.device)
        for i in range(len(pred_boxes)):
            if is_match[i]:
                gt_idx = max_idx[i]
                if not gt_matched[gt_idx]:
                    gt_matched[gt_idx] = True
                else:
                    is_match[i] = False

        # 收集结果
        all_scores.extend(scores.cpu().numpy())
        all_matches.extend(is_match.cpu().numpy())

    # 如果没有预测
    if len(all_scores) == 0:
        return 0.0

    # 按分数降序排列
    indices = np.argsort(-np.array(all_scores))
    all_matches = np.array(all_matches)[indices]

    # 计算精度和召回率
    cum_tp = np.cumsum(all_matches)
    cum_fp = np.cumsum(1 - all_matches)
    precision = cum_tp / (cum_tp + cum_fp)

    total_gt = sum(len(gt['box_xyz']) for gt in targets)
    recall = cum_tp / total_gt if total_gt > 0 else np.zeros_like(cum_tp)

    # 11点插值AP计算
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return ap


def compute_3d_iou(boxes1, boxes2):
    """
    计算两组3D边界框之间的IoU
    Args:
        boxes1: [N, 7] - x, y, z, w, l, h, θ
        boxes2: [M, 7] - x, y, z, w, l, h, θ
    Returns:
        Tensor: [N, M] IoU矩阵
    """
    # 简化计算，暂时不考虑旋转，只计算轴对齐的3D IoU
    def get_box_min_max(boxes):
        # boxes: [..., 7] - x, y, z, w, l, h, θ
        centers = boxes[..., :3]
        dimensions = boxes[..., 3:6]

        # 计算边界框的最小/最大坐标
        half_dimensions = dimensions / 2
        mins = centers - half_dimensions
        maxs = centers + half_dimensions
        return mins, maxs

    # 获取两组边界框的最小/最大坐标
    mins1, maxs1 = get_box_min_max(boxes1)  # [N, 3], [N, 3]
    mins2, maxs2 = get_box_min_max(boxes2)  # [M, 3], [M, 3]

    # 计算交集边界框的最小/最大坐标
    mins1 = mins1.unsqueeze(1)  # [N, 1, 3]
    maxs1 = maxs1.unsqueeze(1)  # [N, 1, 3]
    mins2 = mins2.unsqueeze(0)  # [1, M, 3]
    maxs2 = maxs2.unsqueeze(0)  # [1, M, 3]

    intersect_mins = torch.max(mins1, mins2)  # [N, M, 3]
    intersect_maxs = torch.min(maxs1, maxs2)  # [N, M, 3]

    # 计算交集体积
    intersect_dimensions = torch.clamp(
        intersect_maxs - intersect_mins, min=0)  # [N, M, 3]
    intersect_volumes = intersect_dimensions.prod(dim=2)  # [N, M]

    # 计算两组边界框的体积
    volumes1 = boxes1[:, 3:6].prod(dim=1).unsqueeze(1)  # [N, 1]
    volumes2 = boxes2[:, 3:6].prod(dim=1).unsqueeze(0)  # [1, M]

    # 计算并集体积
    union_volumes = volumes1 + volumes2 - intersect_volumes  # [N, M]

    # 计算IoU
    iou = intersect_volumes / (union_volumes + 1e-7)  # [N, M]

    return iou


def evaluate_3d_detection(
    checkpoint_path,
    version='v1.0-mini',
    dataroot='/data/nuscenes',
    gpuid=0,
    H=900, W=1600,
    resize_lim=(0.193, 0.225),
    final_dim=(128, 352),
    bot_pct_lim=(0.0, 0.22),
    rot_lim=(-5.4, 5.4),
    rand_flip=False,
    ncams=5,
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 45.0, 1.0],
    bsz=4,
    nworkers=0,
    num_classes=10,
    amp=True,
):
    """
    使用保存的模型检查点评估3D检测性能
    """
    from .models import compile_model
    from .data import compile_data

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }

    # 加载验证数据
    _, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                parser_name='detection3d')

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 加载模型
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 使用正确的参数初始化模型
    # 对于3D检测任务，outC应该是num_classes*9
    model = compile_model(grid_conf, data_aug_conf,
                          outC=num_classes*9, model='beve', num_classes=num_classes)

    if model is None:
        raise ValueError("模型初始化失败，请检查compile_model函数")

    # 加载权重
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # 计算mAP
    map_value = compute_map(model, valloader, device, amp=amp)
    print(f"mAP: {map_value:.4f}")

    return map_value
