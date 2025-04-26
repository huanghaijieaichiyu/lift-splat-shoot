import torch
import torch.nn as nn
import time  # 直接导入time模块
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler  # 正确导入
from tqdm import tqdm
import numpy as np
import os
from .models import compile_model
from .data import compile_data
from .tools import save_path
from .tools import get_batch_iou, get_val_info
from contextlib import nullcontext  # 导入nullcontext
from .nuscenes_info import load_nuscenes_infos  # 导入数据集缓存加载函数
from .tools import CenterPointLoss, DetectionBEVLoss  # 导入新的损失函数
from .evaluate_3d import decode_predictions, evaluate_with_nuscenes  # Add import


def distance_based_nms(boxes_xyz, scores, classes, dist_threshold, score_threshold):
    """
    执行基于中心点距离的简化NMS (同一类别内).

    Args:
        boxes_xyz (torch.Tensor): 检测框中心坐标 [N, 3] or [N, 2]. 使用BEV (x, y) 进行距离计算.
        scores (torch.Tensor): 检测框置信度 [N].
        classes (torch.Tensor): 检测框类别 [N].
        dist_threshold (float): 同一类别内，移除框的中心点距离阈值 (BEV平面).
        score_threshold (float): 低于此分数的框将被首先过滤掉.

    Returns:
        torch.Tensor: 保留的检测框的原始索引.
    """
    # 1. 按分数过滤
    keep_indices_score = torch.where(scores >= score_threshold)[0]
    if keep_indices_score.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=scores.device)

    boxes_xyz_filtered = boxes_xyz[keep_indices_score]
    scores_filtered = scores[keep_indices_score]
    classes_filtered = classes[keep_indices_score]

    if boxes_xyz_filtered.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes_xyz.device)

    # 2. 按分数排序 (在每个类别内独立排序，或者全局排序后处理也可以)
    # 这里我们将在类别循环内排序

    keep = []
    unique_classes = torch.unique(classes_filtered)

    # 3. 按类别进行NMS
    for cls in unique_classes:
        cls_mask = (classes_filtered == cls)
        if not torch.any(cls_mask):
            continue

        cls_indices_in_filtered = torch.where(cls_mask)[0]
        cls_boxes_xyz = boxes_xyz_filtered[cls_indices_in_filtered]
        cls_scores = scores_filtered[cls_indices_in_filtered]
        # 获取这些框在过滤前（应用分数阈值后）的原始索引
        original_indices_for_class = keep_indices_score[cls_indices_in_filtered]

        # 按分数降序排序当前类别的框
        cls_order = torch.argsort(cls_scores, descending=True)
        cls_active = torch.ones(
            cls_boxes_xyz.shape[0], dtype=torch.bool, device=cls_boxes_xyz.device)

        for i in range(cls_boxes_xyz.shape[0]):
            idx_in_class_sorted = cls_order[i]  # 获取排序后的索引对应的在类别内的索引
            if cls_active[idx_in_class_sorted]:
                # 保留当前框 (使用它在过滤前列表中的原始索引)
                keep.append(original_indices_for_class[idx_in_class_sorted])
                # 获取当前框的 BEV 坐标
                current_box_bev = cls_boxes_xyz[idx_in_class_sorted, :2].unsqueeze(
                    0)
                # 计算当前框与该类别中其他框的 BEV 距离
                distances = torch.norm(
                    cls_boxes_xyz[:, :2] - current_box_bev, p=2, dim=1)
                # 找到需要抑制的框 (距离小于阈值，且不是自身，且还未被抑制)
                # 注意：比较的是 cls_active 内的索引
                suppress_mask = (distances < dist_threshold) & cls_active
                suppress_mask[idx_in_class_sorted] = False  # 确保不会抑制自身
                cls_active[suppress_mask] = False

    if not keep:
        return torch.tensor([], dtype=torch.long, device=boxes_xyz.device)

    # 返回保留框的原始索引列表（相对于输入boxes_xyz的索引）
    return torch.tensor(keep, dtype=torch.long, device=boxes_xyz.device)


def check_and_ensure_cache(dataroot, version):
    """
    检查并确保数据集缓存存在

    Args:
        dataroot: 数据集根目录
        version: 数据集版本，如'mini'、'trainval'
    """
    # 设置NuScenes数据集版本
    version_str = f'v1.0-{version}'
    max_sweeps = 10  # 设置最大扫描帧数

    try:
        print(f"训练前检查NuScenes缓存信息...")
        from .nuscenes_info import load_nuscenes_infos

        # 这里将直接触发缓存的创建（如果不存在）
        nusc_infos = load_nuscenes_infos(
            dataroot, version=version_str, max_sweeps=max_sweeps)

        # 检查缓存是否包含必要信息
        if nusc_infos and 'infos' in nusc_infos and 'train' in nusc_infos['infos'] and 'val' in nusc_infos['infos']:
            print(
                f"缓存验证成功！包含{len(nusc_infos['infos']['train'])}个训练样本和{len(nusc_infos['infos']['val'])}个验证样本")
            return True
        else:
            print("缓存数据结构不完整，训练过程可能无法使用缓存")
            return False
    except FileNotFoundError as e:
        print(f"找不到缓存文件或相关目录: {e}")
        print("请确保数据集路径正确，并且有写入权限以创建缓存")
        return False
    except Exception as e:
        print(f"训练前准备缓存信息失败：{e}")
        print("训练过程将自动创建缓存，但可能会减慢初始加载速度")
        return False


def train(version,
          dataroot='/data/nuscenes',
          nepochs=10000,
          gpuid=0,
          cuDNN=False,
          resume='',
          load_weight='',
          amp=True,
          H=900, W=1600,
          resize_lim=(0.193, 0.225),
          final_dim=(128, 352),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=5,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-50.0, 50.0, 0.5],
          ybound=[-50.0, 50.0, 0.5],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 45.0, 1.0],

          bsz=4,
          nworkers=4,
          lr=1e-3,
          weight_decay=1e-7,
          ):
    # 训练前验证并确保数据集缓存存在
    check_and_ensure_cache(dataroot, version)

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
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    loss_seg = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight)).to(device)  # 分割损失
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    if resume != '':
        path = os.path.dirname(resume)
    else:
        path = save_path(logdir)
    writer = SummaryWriter(logdir=path)
    val_step = 10 if version == 'mini' else 30
    model.train()
    counter = 0
    epoch = 0
    if resume != '':
        print("Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)
        opt.load_state_dict(checkpoint['optimizer'])

    while epoch < nepochs:
        np.random.seed()
        Iou = [0.0]  # 初始化为浮点数
        t0, t1 = 0.0, 0.0  # Initialize time variables
        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in pbar:
            t0 = time.time()  # 确保是浮点数
            opt.zero_grad()
            with autocast(enabled=amp):
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )
            binimgs = binimgs.to(device)

            with autocast(enabled=amp):
                loss = loss_seg(preds, binimgs)  # 使用分割损失变量名
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time.time()  # 确保是浮点数

            pbar.set_description('||Epoch: [%d/%d]|-----|-----||Batch: [%d/%d]||-----|-----|| Loss: %.4f||'
                                 % (epoch + 1, nepochs, batchi + 1, len(trainloader), loss.item()))

        # Save the last model for resume
        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
        }
        last_model = os.path.join(path, "last.pt")
        torch.save(last_checkpoint, last_model)

        if (epoch + 1) % 10 == 0 and (epoch + 1) >= 10:
            writer.add_scalar('train/loss', loss, counter)

        if (epoch + 1) % val_step == 0 and (epoch + 1) >= val_step:
            intersect, union, iou = get_batch_iou(preds, binimgs)
            # Save the bast model in iou
            if float(iou) > max(Iou):
                best_model = os.path.join(path, "best.pt")
                torch.save(model, best_model)

            Iou.append(float(iou))  # 确保添加的是浮点数
            writer.add_scalar('train/iou', iou, epoch + 1)
            step_time = t1 - t0  # 两个time.time()函数返回值相减得到的就是浮点数
            writer.add_scalar('train/step_time', step_time, epoch + 1)
            val_info = get_val_info(model, valloader, loss_seg, device)
            print(
                '||val/loss: {} ||-----|-----||val/iou: {}||'.format(val_info['loss'], val_info['iou']))
            writer.add_scalar('val/loss', val_info['loss'], epoch + 1)
            writer.add_scalar('val/iou', val_info['iou'], epoch + 1)
            print('---------|Debug data print here|-----------')
            print(
                '|intersect: {}|-----|-----|union: {}|-----|-----|iou: {}|'.format(intersect, union, iou))
            print('-----------------|done|--------------------')

        epoch += 1
        pbar.close()
    writer.close()


def train_3d(version,
             dataroot='/data/nuscenes',
             nepochs=100,
             gpuid=0,
             cuDNN=False,
             resume='',
             load_weight='',
             H=900, W=1600,
             resize_lim=(0.193, 0.225),
             final_dim=(128, 352),
             bot_pct_lim=(0.0, 0.22),
             rot_lim=(-5.4, 5.4),
             rand_flip=True,
             ncams=6,
             max_grad_norm=10.0,
             logdir='./runs_3d',

             xbound=[-50.0, 50.0, 0.5],
             ybound=[-50.0, 50.0, 0.5],
             zbound=[-5.0, 3.0, 0.5],
             dbound=[4.0, 45.0, 1.0],

             bsz=4,
             nworkers=4,
             lr=2e-4,
             weight_decay=1e-7,
             num_classes=10,
             enable_multiscale=True,
             use_enhanced_bev=True,
             bev_loss_type='ciou',
             loss_alpha=0.25,
             loss_gamma=2.0,
             loss_beta=1.0,
             dwa_temperature: float = 2.0,
             dwa_loss_keys: list = ['heatmap', 'offset',
                                    'z_coord', 'dimension', 'rotation', 'velocity'],
             eps: float = 1e-8  # Epsilon for numerical stability
             ):
    """
    训练用于3D目标检测的BEVENet模型
    增加多尺度特征训练支持和增强的BEV投影
    使用 Dynamic Weight Averaging (DWA) 进行动态损失加权
    """
    # 训练前验证并确保数据集缓存存在
    check_and_ensure_cache(dataroot, version)

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': (0.2, 0.2),  # 禁用随机 resize/crop
        'final_dim': final_dim,
        'rot_lim': (0.0, 0.0),    # 禁用随机旋转
        'H': H, 'W': W,
        'rand_flip': False,       # 禁用随机翻转
        'bot_pct_lim': (0.0, 0.0),  # 禁用底部随机裁剪/填充
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='detection3d')

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 使用BEVENet模型用于3D目标检测
    # 确保输出通道数至少为1
    actual_num_classes = max(1, num_classes)
    # --- 修改开始 ---
    # 计算模型总输出通道数 (分类 + 回归 + IoU)
    # 回归参数: x,y,z, w,l,h, sin,cos, vel (9)
    # IoU预测: 1 (假设存在)
    total_output_channels = actual_num_classes + 9 + 1  # 假设包含IoU头
    # 检查并确保通道数大于0
    if total_output_channels <= 0:
        raise ValueError(f"计算出的总输出通道数必须大于0，但得到 {total_output_channels}")

    print(
        f"配置BEVE模型，num_classes={actual_num_classes}, total_output_channels={total_output_channels}")
    model = compile_model(grid_conf, data_aug_conf,
                          outC=total_output_channels,
                          model='beve',
                          num_classes=actual_num_classes,
                          backbone_type='resnet18')
    # --- 修改结束 ---

    # 记录模型配置
    model_configs = {
        'grid_conf': grid_conf,
        'data_aug_conf': data_aug_conf,
        'num_classes': actual_num_classes,
        'enable_multiscale': enable_multiscale,
        'use_enhanced_bev': use_enhanced_bev
    }

    # 应用增强的BEV投影设置
    if hasattr(model, 'feat_enhancement'):
        # 检查feat_enhancement属性的类型
        if isinstance(model.feat_enhancement, (torch.Tensor, nn.Module)):
            print("Warning: feat_enhancement属性不是布尔值，跳过设置")
        else:
            # 安全地设置布尔值
            setattr(model, 'feat_enhancement', bool(use_enhanced_bev))

    model.to(device)

    if cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # === MODIFICATION START: Change Loss Function to CenterPointLoss ===
    # Define loss weights for CenterPointLoss
    cp_loss_weights = {
        'heatmap': 1.0,
        'offset': 1.0,
        'z_coord': 1.0,
        'dimension': 1.0,
        'rotation': 1.0,
        'velocity': 1.0
    }

    # Instantiate CenterPointLoss
    loss_fn = CenterPointLoss(num_classes=actual_num_classes,
                              loss_weights=cp_loss_weights
                              ).to(device)
    # === MODIFICATION END ===

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )

    if resume != '':
        path = os.path.dirname(resume)
    else:
        path = save_path(logdir)

    writer = SummaryWriter(logdir=path)
    val_step = 10 if version == 'mini' else 30

    model.train()
    counter = 0
    epoch = 0
    best_map = 0

    # === MODIFICATION START: Update possible loss keys ===
    all_possible_loss_keys = ['heatmap', 'offset',
                              'z_coord', 'dimension', 'rotation', 'velocity']
    # === MODIFICATION END ===
    prev_epoch_raw_losses = {k: torch.tensor(
        1.0, device=device) for k in all_possible_loss_keys}
    dwa_weights = {k: torch.tensor(1.0, device=device)
                   for k in all_possible_loss_keys}

    if resume != '':
        print("Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0)
        resume = ''

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)

    amp_scaler = GradScaler() if torch.cuda.is_available() else None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_raw_losses_sum = {k: 0.0 for k in all_possible_loss_keys}
        num_batches = len(trainloader)

        if epoch > 0:
            current_avg_raw_losses = {k: epoch_raw_losses_sum[k] / num_batches
                                      for k in all_possible_loss_keys}
            loss_ratios = {k: current_avg_raw_losses[k] / (prev_epoch_raw_losses[k] + eps)
                           for k in all_possible_loss_keys}

            exps = {k: torch.exp(
                loss_ratios[k] / dwa_temperature) for k in all_possible_loss_keys}
            sum_exps = sum(exps.values())

            num_tasks = len(all_possible_loss_keys)
            dwa_weights = {k: exps[k] / (sum_exps + eps) * num_tasks
                           for k in all_possible_loss_keys}

            prev_epoch_raw_losses = {k: torch.tensor(current_avg_raw_losses[k], device=device)
                                     for k in all_possible_loss_keys}
        else:
            dwa_weights = {k: torch.tensor(1.0, device=device)
                           for k in all_possible_loss_keys}

        pbar = tqdm(enumerate(trainloader),
                    total=num_batches, colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sample_tokens) in pbar:
            opt.zero_grad()
            B = imgs.size(0)  # Get batch size

            with autocast(enabled=(amp_scaler is not None)):
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )

                # === MODIFICATION START: Correctly use collated targets ===
                # targets_list IS the collated dictionary from custom_collate
                # NO NEED to stack or pad again here.
                # Just move the tensors in the dictionary to the correct device.
                targets_on_device = {}
                for key, value in targets_list.items():  # Iterate through the collated dictionary
                    if isinstance(value, torch.Tensor):
                        targets_on_device[key] = value.to(device)
                    else:
                        # Keep non-tensor items (like sample_tokens if they were part of dict)
                        targets_on_device[key] = value
                # === MODIFICATION END ===

                # Calculate loss using CenterPointLoss (expects the collated dictionary with tensors on device)
                raw_losses = loss_fn(preds, targets_on_device)

                # Calculate total weighted loss (remains the same logic)
                total_loss = torch.tensor(0.0, device=device)
                for key in all_possible_loss_keys:
                    loss_key = f'raw_{key}_loss'
                    if loss_key in raw_losses and raw_losses[loss_key] is not None:
                        total_loss += dwa_weights[key] * raw_losses[loss_key]

            scaled_loss = total_loss

            if amp_scaler is not None:
                amp_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (batchi + 1) % num_batches == 0:
                if amp_scaler is not None:
                    amp_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)

                if amp_scaler is not None:
                    amp_scaler.step(opt)
                    amp_scaler.update()
                else:
                    opt.step()

            epoch_loss += total_loss.item()
            for key in all_possible_loss_keys:
                loss_key = f'raw_{key}_loss'
                if loss_key in raw_losses and raw_losses[loss_key] is not None:
                    epoch_raw_losses_sum[key] += raw_losses[loss_key].item()

            counter += 1
            t1 = time.time()

            desc_losses = " | ".join([f"{k[:4].upper()}: {raw_losses.get(f'raw_{k}_loss', torch.tensor(0.0)).item():.3f}"
                                      for k in all_possible_loss_keys])
            desc = f'||Epoch: [{epoch + 1}/{nepochs}]|WLoss: {total_loss.item():.3f}|{desc_losses}||'
            pbar.set_description(desc)

        avg_epoch_loss = epoch_loss / num_batches
        avg_raw_losses = {
            k: epoch_raw_losses_sum[k] / num_batches for k in all_possible_loss_keys}

        writer.add_scalar('train/total_weighted_loss',
                          avg_epoch_loss, epoch + 1)
        for key in all_possible_loss_keys:
            writer.add_scalar(
                f'train/raw_{key}_loss', avg_raw_losses[key], epoch + 1)
            if key in dwa_weights:
                writer.add_scalar(
                    f'dwa_weights/{key}_weight', dwa_weights[key].item(), epoch + 1)

        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
            'best_map': best_map,
            'model_configs': model_configs,
            'amp_scaler': amp_scaler.state_dict() if amp_scaler is not None else None,
        }
        last_model = os.path.join(path, "last.pt")
        torch.save(last_checkpoint, last_model)

        if (epoch + 1) % val_step == 0 and (epoch + 1) >= val_step:
            model.eval()
            val_raw_losses_sum = {k: 0.0 for k in all_possible_loss_keys}

            all_pred_scores_list = []
            all_pred_cls_list = []
            all_pred_xyz_list = []
            all_pred_wlh_list = []
            all_pred_rot_sincos_list = []
            all_pred_vel_list = []
            all_pred_token_list = []
            all_val_tokens_processed = []

            with torch.no_grad():
                for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sample_tokens) in enumerate(valloader):
                    all_val_tokens_processed.extend(sample_tokens)
                    B_val = imgs.size(0)

                    with autocast(enabled=(amp_scaler is not None)):
                        preds_dict = model(imgs.to(device),
                                           rots.to(device),
                                           trans.to(device),
                                           intrins.to(device),
                                           post_rots.to(device),
                                           post_trans.to(device),
                                           )

                        targets_on_device = {}
                        try:
                            targets_on_device['target_heatmap'] = torch.stack(
                                [t['target_heatmap'] for t in targets_list]).to(device)
                            targets_on_device['target_mask'] = torch.stack(
                                [t['target_mask'] for t in targets_list]).to(device)
                        except Exception as e:
                            print(
                                f"[Validation] Error stacking dense targets: {e}")
                            continue

                        sparse_keys = ['target_indices', 'target_offset', 'target_z_coord',
                                       'target_dimension', 'target_rotation', 'target_velocity', 'num_objs']
                        for key in sparse_keys:
                            try:
                                targets_on_device[key] = [t[key].to(device) if isinstance(
                                    t[key], torch.Tensor) else t[key] for t in targets_list]
                            except Exception as e:
                                print(
                                    f"[Validation] Error processing sparse key '{key}': {e}")
                                targets_on_device[key] = [
                                    None] * len(targets_list)

                        raw_losses = loss_fn(preds_dict, targets_on_device)
                        for key in all_possible_loss_keys:
                            loss_key = f'raw_{key}_loss'
                            if loss_key in raw_losses and raw_losses[loss_key] is not None:
                                val_raw_losses_sum[key] += raw_losses[loss_key].item()

                    batch_dets_list = decode_predictions(
                        preds_dict, device, score_thresh=0.1, grid_conf=grid_conf, K=500
                    )
                    for sample_idx, dets_dict in enumerate(batch_dets_list):
                        num_dets_in_sample = dets_dict['box_scores'].shape[0]
                        if num_dets_in_sample > 0:
                            all_pred_scores_list.append(
                                dets_dict['box_scores'].cpu())
                            all_pred_cls_list.append(
                                dets_dict['box_cls'].cpu())
                            all_pred_xyz_list.append(
                                dets_dict['box_xyz'].cpu())
                            all_pred_wlh_list.append(
                                dets_dict['box_wlh'].cpu())
                            all_pred_rot_sincos_list.append(
                                dets_dict['box_rot_sincos'].cpu())
                            all_pred_vel_list.append(
                                dets_dict['box_vel'].cpu())
                            sample_token = sample_tokens[sample_idx]
                            all_pred_token_list.extend(
                                [sample_token] * num_dets_in_sample)

            avg_val_raw_losses = {
                k: val_raw_losses_sum[k] /
                len(valloader) if len(valloader) > 0 else 0.0
                for k in all_possible_loss_keys
            }
            for key in all_possible_loss_keys:
                writer.add_scalar(
                    f'val/raw_{key}_loss', avg_val_raw_losses[key], epoch + 1)

            val_map_score = 0.0
            val_nds_score = 0.0
            metrics_dict = {}

            if not all_pred_scores_list:
                print(
                    "Warning: No detections found across validation set for nuScenes evaluation.")
            else:
                all_predictions_dict = {
                    'box_scores': torch.cat(all_pred_scores_list).numpy(),
                    'box_cls': torch.cat(all_pred_cls_list).numpy(),
                    'box_xyz': torch.cat(all_pred_xyz_list).numpy(),
                    'box_wlh': torch.cat(all_pred_wlh_list).numpy(),
                    'box_rot_sincos': torch.cat(all_pred_rot_sincos_list).numpy(),
                    'box_vel': torch.cat(all_pred_vel_list).numpy(),
                    'sample_tokens': all_pred_token_list
                }

                eval_set = 'val' if 'mini' not in version else 'mini_val'
                full_version = f'v1.0-{version}'

                try:
                    print("Starting official nuScenes evaluation...")
                    metrics_dict = evaluate_with_nuscenes(
                        predictions_dict=all_predictions_dict,
                        nuscenes_version=full_version,
                        nuscenes_dataroot=dataroot,
                        eval_set=eval_set,
                        output_dir=path,
                        verbose=True,
                        all_sample_tokens=list(set(all_val_tokens_processed))
                    )
                    val_map_score = metrics_dict.get('mAP', 0.0)
                    val_nds_score = metrics_dict.get('NDS', 0.0)
                    print("nuScenes evaluation finished.")
                except Exception as e:
                    print(f"Error during nuScenes evaluation: {e}")
                    print("Skipping mAP/NDS reporting for this epoch.")
                    val_map_score = 0.0
                    val_nds_score = 0.0
                    metrics_dict = {}

            val_loss_str = " | ".join(
                [f"R-{k[:4].upper()}: {avg_val_raw_losses[k]:.3f}" for k in all_possible_loss_keys])
            print(
                f'||Val {val_loss_str} | mAP: {val_map_score:.4f} | NDS: {val_nds_score:.4f} ||')
            writer.add_scalar('val/mAP', val_map_score, epoch + 1)
            writer.add_scalar('val/NDS', val_nds_score, epoch + 1)
            for k, v in metrics_dict.items():
                if k not in ['per_class_AP', 'per_class_TP_Errors']:
                    writer.add_scalar(f'val_nusc/{k}', v, epoch + 1)

            if val_nds_score > best_map:
                best_map = val_nds_score
                best_checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': epoch,
                    'best_map': best_map,
                    'model_configs': model_configs,
                    'amp_scaler': amp_scaler.state_dict() if amp_scaler is not None else None,
                }
                best_model_path = os.path.join(
                    path, "best_map.pt")
                torch.save(best_checkpoint, best_model_path)
                print(f"Best model saved... NDS: {best_map:.4f}")

        epoch += 1
        pbar.close()

    writer.close()


def train_fusion(version,
                 dataroot='/data/nuscenes',
                 nepochs=10000,
                 gpuid=0,
                 cuDNN=False,
                 resume='',
                 load_weight='',
                 amp=True,
                 H=900, W=1600,
                 resize_lim=(0.193, 0.225),
                 final_dim=(128, 352),
                 bot_pct_lim=(0.0, 0.22),
                 rot_lim=(-5.4, 5.4),
                 rand_flip=True,
                 ncams=5,
                 max_grad_norm=5.0,
                 logdir='./runs_multimodal_detection',
                 xbound=[-50.0, 50.0, 0.5],
                 ybound=[-50.0, 50.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[4.0, 45.0, 1.0],
                 bsz=1,
                 nworkers=0,
                 lr=1e-3,
                 weight_decay=1e-7,
                 num_classes=10,
                 lidar_channels=18,
                 enable_multiscale=True,
                 use_enhanced_fusion=True,
                 grad_accum_steps=2,
                 bev_loss_type='ciou',
                 loss_alpha=0.25,
                 loss_gamma=2.0,
                 loss_beta=1.0,
                 dwa_temperature: float = 2.0,
                 dwa_loss_keys: list = ['cls', 'bev', 'z', 'h', 'vel', 'iou'],
                 eps: float = 1e-8  # Epsilon for numerical stability
                 ):
    """
    训练用于3D目标检测的多模态融合模型
    结合相机和LiDAR特征进行训练
    使用 Dynamic Weight Averaging (DWA) 进行动态损失加权
    """
    # 训练前验证并确保数据集缓存存在
    check_and_ensure_cache(dataroot, version)

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

    # 使用多模态数据加载器
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='multimodal_detection')

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 使用多模态融合模型 (BEVENet with fusion handling)
    actual_num_classes = max(1, num_classes)
    # total_output_channels is not directly relevant for CenterPoint head
    # The underlying BEVENet model (when model='fusion') should handle the fusion internally
    # and use the BEVEncoderCenterPointHead which defines its own outputs.
    print(
        f"配置 Fusion (BEVENet) 模型，num_classes={actual_num_classes}, lidar_channels={lidar_channels}")
    # Assuming 'fusion' model uses the BEVENet we modified, which has CenterPoint head
    model = compile_model(grid_conf, data_aug_conf,
                          outC=1,  # Placeholder outC, not used by CenterPoint head
                          model='fusion',  # This should ideally configure BEVENet internally
                          num_classes=actual_num_classes,
                          lidar_channels=lidar_channels)  # Pass lidar_channels if BEVENet needs it for fusion

    # 记录模型配置
    model_configs = {
        'grid_conf': grid_conf,
        'data_aug_conf': data_aug_conf,
        'num_classes': actual_num_classes,
        'enable_multiscale': enable_multiscale,  # These might be flags for the model?
        'use_enhanced_fusion': use_enhanced_fusion,
        'lidar_channels': lidar_channels
    }

    model.to(device)

    if cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # --- CenterPoint Loss Setup ---
    # Define loss weights
    cp_loss_weights = {
        'heatmap': 1.0,
        'offset': 1.0,
        'z_coord': 1.0,
        'dimension': 1.0,
        'rotation': 1.0,
        'velocity': 1.0
    }
    # === ADDITION START: Re-introduce cp_dwa_loss_keys ===
    # DWA loss keys specific to CenterPointLoss
    cp_dwa_loss_keys = ['heatmap', 'offset',
                        'z_coord', 'dimension', 'rotation', 'velocity']
    # === ADDITION END ===

    loss_fn = CenterPointLoss(num_classes=actual_num_classes,
                              loss_weights=cp_loss_weights
                              ).to(device)
    # ---

    opt = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 设置学习率调度器 - 使用余弦退火调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepochs *
        (len(trainloader) // grad_accum_steps),
        pct_start=0.2, div_factor=10, final_div_factor=10,
        anneal_strategy='cos'
    )

    if resume != '':
        path = os.path.dirname(resume)
    else:
        path = save_path(logdir)

    writer = SummaryWriter(logdir=path)
    val_step = 10 if version == 'mini' else 30

    model.train()
    counter = 0
    epoch = 0
    best_map = 0

    # === MODIFICATION START: Update possible loss keys (should match cp_dwa_loss_keys) ===
    all_possible_loss_keys = cp_dwa_loss_keys
    prev_epoch_raw_losses = {k: torch.tensor(
        1.0, device=device) for k in all_possible_loss_keys}
    dwa_weights = {k: torch.tensor(1.0, device=device)
                   for k in all_possible_loss_keys}
    # === MODIFICATION END ===

    if resume != '':
        print("Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0)
        resume = ''

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)

    amp_scaler = GradScaler() if amp and torch.cuda.is_available() else None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_raw_losses_sum = {k: 0.0 for k in all_possible_loss_keys}
        num_batches = len(trainloader)
        num_opt_steps = num_batches // grad_accum_steps

        if epoch > 0:
            # DWA calculation (uses cp_dwa_loss_keys)
            current_avg_raw_losses = {k: epoch_raw_losses_sum[k] / num_batches
                                      for k in cp_dwa_loss_keys}
            loss_ratios = {k: current_avg_raw_losses[k] / (prev_epoch_raw_losses[k] + eps)
                           for k in cp_dwa_loss_keys}

            exps = {k: torch.exp(
                loss_ratios[k] / dwa_temperature) for k in cp_dwa_loss_keys}
            sum_exps = sum(exps.values())

            num_tasks = len(cp_dwa_loss_keys)
            dwa_weights = {k: exps[k] / (sum_exps + eps) * num_tasks
                           for k in cp_dwa_loss_keys}

            prev_epoch_raw_losses = {k: torch.tensor(current_avg_raw_losses[k], device=device)
                                     for k in cp_dwa_loss_keys}
        else:
            # Initialize weights (uses cp_dwa_loss_keys)
            dwa_weights = {k: torch.tensor(1.0, device=device)
                           for k in cp_dwa_loss_keys}

        opt.zero_grad()
        pbar = tqdm(enumerate(trainloader),
                    total=num_batches, colour='#8762A5')

        # Training loop (uses cp_dwa_loss_keys for loss calculation and logging)
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list, sample_tokens) in pbar:
            t0 = time.time()
            B = imgs.size(0)

            with autocast(enabled=(amp_scaler is not None)):
                preds_dict = model(imgs.to(device),
                                   rots.to(device),
                                   trans.to(device),
                                   intrins.to(device),
                                   post_rots.to(device),
                                   post_trans.to(device),
                                   )

                # === MODIFICATION START: Pad and Stack Sparse Targets ===
                targets_on_device = {}
                # Stack dense maps
                try:
                    targets_on_device['target_heatmap'] = torch.stack(
                        [t['target_heatmap'] for t in targets_list]).to(device)
                    targets_on_device['target_mask'] = torch.stack(
                        [t['target_mask'] for t in targets_list]).to(device)
                except Exception as e:
                    print(f"Error stacking dense targets: {e}")
                    continue  # Skip batch if targets are bad

                # Initialize padded sparse tensors and mask
                padded_indices = torch.zeros(
                    (B, 500), dtype=torch.long, device=device)
                padded_offset = torch.zeros(
                    (B, 500, 2), dtype=torch.float32, device=device)
                padded_z = torch.zeros(
                    (B, 500, 1), dtype=torch.float32, device=device)
                padded_dim = torch.zeros(
                    (B, 500, 3), dtype=torch.float32, device=device)
                padded_rot = torch.zeros(
                    (B, 500, 2), dtype=torch.float32, device=device)
                padded_vel = torch.zeros(
                    (B, 500, 2), dtype=torch.float32, device=device)
                reg_padding_mask = torch.zeros(
                    (B, 500), dtype=torch.bool, device=device)

                sparse_keys_to_pad = ['target_indices', 'target_offset', 'target_z_coord',
                                      'target_dimension', 'target_rotation', 'target_velocity']
                padded_tensors = {
                    'target_indices': padded_indices,
                    'target_offset': padded_offset,
                    'target_z_coord': padded_z,
                    'target_dimension': padded_dim,
                    'target_rotation': padded_rot,
                    'target_velocity': padded_vel
                }

                for i in range(B):
                    try:
                        num_obj = targets_list[i]['num_objs'].item()
                        if num_obj > 500:
                            num_obj = 500  # Clip

                        if num_obj > 0:
                            reg_padding_mask[i, :num_obj] = True
                            for key in sparse_keys_to_pad:
                                source_tensor = targets_list[i][key]
                                padded_tensors[key][i, :num_obj] = source_tensor[:num_obj].to(
                                    device)

                    except KeyError as e:
                        print(
                            f"KeyError accessing targets for sample {i}: {e}.")
                        continue
                    except Exception as e:
                        print(
                            f"Error padding sparse targets for sample {i}: {e}")
                        continue

                # Add padded sparse targets and mask to targets_on_device
                targets_on_device.update(padded_tensors)
                targets_on_device['reg_mask'] = reg_padding_mask
                targets_on_device['num_objs'] = torch.tensor(
                    [t['num_objs'].item() for t in targets_list], device=device)
                # === MODIFICATION END ===

                # Calculate loss
                raw_losses = loss_fn(preds_dict, targets_on_device)

                # Calculate weighted total loss
                total_loss = torch.tensor(0.0, device=device)
                for key in cp_dwa_loss_keys:
                    loss_key = f'raw_{key}_loss'
                    if loss_key in raw_losses and raw_losses[loss_key] is not None:
                        total_loss += dwa_weights[key] * raw_losses[loss_key]
