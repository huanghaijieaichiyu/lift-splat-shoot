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
    loss = nn.BCEWithLogitsLoss()
    model.to(device)
    if cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    loss_fn = loss(pos_weight).cuda(gpuid)

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
        loss_fn.load_state_dict(checkpoint['loss'])
        epoch = checkpoint['epoch']

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)
        opt.load_state_dict(checkpoint['optimizer'])
        loss_fn.load_state_dict(checkpoint['loss'])

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
                loss = loss_fn(preds, binimgs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)  # 梯度裁减
                opt.step()
            counter += 1
            t1 = time.time()  # 确保是浮点数

            # print(counter, loss.item())
            pbar.set_description('||Epoch: [%d/%d]|-----|-----||Batch: [%d/%d]||-----|-----|| Loss: %.4f||'
                                 % (epoch + 1, nepochs, batchi + 1, len(trainloader), loss.item()))

        # Save the last model for resume
        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
            'loss': loss_fn.state_dict()
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
            val_info = get_val_info(model, valloader, loss_fn, device)
            print(
                '||val/loss: {} ||-----|-----||val/iou: {}||'.format(val_info['loss'], val_info['iou']))
            writer.add_scalar('val/loss: %.4f', val_info['loss'], epoch + 1)
            writer.add_scalar('val/iou: %.4f', val_info['iou'], epoch + 1)
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
             load_weight='',  # 默认启用混合精度训练，但在验证时需要注意类型一致性
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
             enable_multiscale=True,  # 支持多尺度特征训练
             use_enhanced_bev=True,   # 使用增强的BEV投影
             ):
    """
    训练用于3D目标检测的BEVENet模型
    增加多尺度特征训练支持和增强的BEV投影
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
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='detection3d')  # 使用3D检测数据解析器

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
                          outC=total_output_channels,  # 传递总通道数
                          model='beve',
                          num_classes=actual_num_classes)  # num_classes 参数可能仍用于内部层
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

    # 优化器设置
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           betas=(0.5, 0.999),
                           weight_decay=weight_decay)

    # 设置学习率调度器 - 使用余弦退火调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepochs * len(trainloader),
        pct_start=0.2, div_factor=10, final_div_factor=10,
        anneal_strategy='cos'  # 使用余弦退火
    )

    # 检查如果启用了多尺度训练，打印一条警告信息但不影响程序运行

    # 损失函数
    from .tools import DetectionBEVLoss
    # 初始化损失函数，只使用支持的参数
    loss_fn = DetectionBEVLoss(num_classes=num_classes,
                               cls_weight=1.0,  # Increased weight from 1.0
                               iou_weight=4.0).to(device)

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

    # 恢复训练
    if resume != '':
        print("Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0)

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)

    # 自动混合精度训练
    amp_scaler = GradScaler() if torch.cuda.is_available() else None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_iou_loss = 0
        # Add accumulators for new regression loss components
        epoch_bev_diou_loss = 0.0
        epoch_z_loss = 0.0
        epoch_h_loss = 0.0
        epoch_vel_loss = 0.0

        # 多尺度训练相关指标
        if enable_multiscale:
            epoch_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')
        # batchi: 当前批次索引
        # imgs: 当前批次图像
        # rots: 当前批次旋转矩阵
        # trans: 当前批次平移矩阵
        # intrins: 当前批次内参矩阵
        # post_rots: 当前批次后处理旋转矩阵
        # post_trans: 当前批次后处理平移矩阵
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sparse_gts_batch, sample_tokens) in pbar:
            # 只有在新的累积周期开始时才清零梯度
            opt.zero_grad()

            # 处理当前批次
            preds = model(imgs.to(device),
                          rots.to(device),
                          trans.to(device),
                          intrins.to(device),
                          post_rots.to(device),
                          post_trans.to(device),
                          )

            # Ensure targets is a dictionary and move to device (for loss calculation)
            targets_dict = {}
            # targets_list is the dictionary yielded by the DataLoader
            for key, val in targets_list.items():
                if isinstance(val, torch.Tensor):
                    targets_dict[key] = val.to(device)
                else:
                    targets_dict[key] = val

            # Calculate loss using the correctly processed dictionary
            losses = loss_fn(preds, targets_dict)

            # --- FIX: Define loss components before accumulating/logging ---
            total_loss = losses['total_loss']
            cls_loss = losses['cls_loss']
            # Use .get with default tensor to avoid KeyError if loss is missing
            iou_loss = losses.get('iou_loss', torch.tensor(0.0, device=device))
            bev_diou_loss = losses.get(
                'bev_diou_loss', torch.tensor(0.0, device=device))
            z_loss = losses.get('z_loss', torch.tensor(0.0, device=device))
            h_loss = losses.get('h_loss', torch.tensor(0.0, device=device))
            vel_loss = losses.get('vel_loss', torch.tensor(0.0, device=device))
            # ---

            # 多尺度损失
            if enable_multiscale and 'scale_losses' in losses:
                scale_losses = losses['scale_losses']

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 反向传播
            losses['total_loss'].backward()

            # 更新参数
            opt.step()

            # Scheduler step should happen after optimizer step
            # This should be outside the scaler check, called every step where optimizer is stepped
            scheduler.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_iou_loss += iou_loss.item()
            epoch_bev_diou_loss += bev_diou_loss.item()
            epoch_z_loss += z_loss.item()
            epoch_h_loss += h_loss.item()
            epoch_vel_loss += vel_loss.item()

            # 多尺度训练相关指标
            if enable_multiscale and 'scale_losses' in losses:
                for i, loss in enumerate(scale_losses):
                    epoch_scale_losses[i] += loss.item()

            counter += 1
            t1 = time.time()  # 确保是浮点数

            # 构建进度条描述
            desc = '||Epoch: [%d/%d]|Loss: %.4f|Cls: %.4f|BEV: %.4f|Z: %.4f|H: %.4f|Vel: %.4f|IoU: %.4f||' % (
                epoch + 1, nepochs, total_loss.item(), cls_loss.item(),
                bev_diou_loss.item(), z_loss.item(), h_loss.item(),
                vel_loss.item(), iou_loss.item()
            )

            # 如果启用多尺度训练，添加多尺度损失信息
            if enable_multiscale and 'scale_losses' in losses:
                desc += ' MS: [%.2f,%.2f,%.2f]' % (
                    scale_losses[0].item(),
                    scale_losses[1].item(),
                    scale_losses[2].item()
                )

            pbar.set_description(desc)

        # 每个epoch结束后记录训练损失
        writer.add_scalar('train/loss', epoch_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/cls_loss', epoch_cls_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/iou_loss', epoch_iou_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/bev_diou_loss', epoch_bev_diou_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/z_loss', epoch_z_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/h_loss', epoch_h_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/vel_loss', epoch_vel_loss /
                          len(trainloader), epoch + 1)

        # 记录多尺度训练损失
        if enable_multiscale:
            for i, loss in enumerate(epoch_scale_losses):
                writer.add_scalar(
                    f'train/scale{i+3}_loss', loss / len(trainloader), epoch + 1)

        # 记录学习率
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch + 1)

        # 保存最后的模型，用于恢复训练
        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_map': best_map,
            'model_configs': model_configs,
            'amp_scaler': amp_scaler.state_dict() if amp_scaler is not None else None,  # 修复None调用问题
        }
        last_model = os.path.join(path, "last.pt")
        torch.save(last_checkpoint, last_model)

        # 验证
        if (epoch + 1) % val_step == 0 and (epoch + 1) >= val_step:
            model.eval()
            val_loss = 0
            val_cls_loss = 0
            val_iou_loss = 0

            # 多尺度验证相关指标
            if enable_multiscale:
                val_scale_losses = [0, 0, 0]

            # === 修改：使用新的变量名收集解码结果和稀疏GT ===
            all_predictions = []
            all_targets_sparse = []  # 用于存储从dataloader加载的稀疏GT
            # ---

            visualized_this_epoch = False  # Flag to visualize only once per epoch

            with torch.no_grad():
                # --- 修改：更新循环变量以接收 sparse_gts_batch --- #
                for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sparse_gts_batch, sample_tokens) in enumerate(valloader):
                    # ---
                    context_manager = nullcontext()
                    with context_manager:
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )

                        # Ensure targets is a dictionary and move to device (for loss calculation)
                        targets_dict = {}
                        for key, val in targets_list.items():
                            if isinstance(val, torch.Tensor):
                                targets_dict[key] = val.to(device)
                            else:
                                # Keep non-tensors as is (e.g., potential metadata)
                                targets_dict[key] = val

                        # 1. 计算和累加损失 (using dense targets_dict)
                        losses = loss_fn(preds, targets_dict)
                        # ... (accumulate validation losses - similar to train_3d) ...
                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        # Assuming reg_loss and iou_loss are present in fusion loss output
                        val_iou_loss += losses.get('iou_loss',
                                                   torch.tensor(0.0)).item()
                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

                    # Decode predictions outside context
                    # --- FIX: Ensure preds_dict is defined correctly --- #
                    if not isinstance(preds, dict) and isinstance(preds, torch.Tensor):
                        B, C, H, W_bev = preds.shape
                        cls_channels = num_classes
                        reg_channels = 9
                        iou_channels = 1
                        expected_channels = cls_channels + reg_channels + iou_channels
                        preds_dict = {}  # Initialize
                        if C >= expected_channels:
                            preds_dict = {
                                'cls_pred': preds[:, :cls_channels],
                                'reg_pred': preds[:, cls_channels: cls_channels + reg_channels],
                                'iou_pred': preds[:, -iou_channels:]
                            }
                        else:
                            print(
                                f"Warning: Pred tensor channels ({C}) < expected ({expected_channels}). Skipping decode.")
                            preds_dict = {'cls_pred': torch.empty(B, 0, H, W_bev), 'reg_pred': torch.empty(
                                B, 0, H, W_bev), 'iou_pred': torch.empty(B, 0, H, W_bev)}
                        preds = preds_dict
                    # ---

                    from .evaluate_3d import decode_predictions  # Reuse decode
                    batch_dets = decode_predictions(
                        preds, device, score_thresh=0.2, grid_conf=grid_conf)

                    # === 修改：直接使用从dataloader加载的 sparse_gts_batch ===
                    processed_sparse_gts_for_eval = []
                    for sample_gt_list in sparse_gts_batch:  # Iterate samples in the batch
                        # Need GTs in the format expected by calculate_simple_ap
                        # which is a list of dicts, each dict holding tensors for ONE sample.
                        if not sample_gt_list:
                            # Handle case with no GT boxes for a sample
                            processed_sparse_gts_for_eval.append({
                                'box_cls': torch.empty(0, dtype=torch.long, device=device),
                                'box_xyz': torch.empty(0, 3, device=device),
                                'box_wlh': torch.empty(0, 3, device=device),
                                'box_rot_sincos': torch.empty(0, 2, device=device),
                                'box_vel': torch.empty(0, 2, device=device)
                            })
                            continue

                        # Combine the list of GT dicts for the sample into a single dict with batched tensors
                        keys = sample_gt_list[0].keys()
                        sample_batched_gts = {}
                        for k in keys:
                            sample_batched_gts[k] = torch.stack(
                                [gt[k] for gt in sample_gt_list], dim=0).to(device)
                        processed_sparse_gts_for_eval.append(
                            sample_batched_gts)

                    # Extend the main lists for evaluation after the loop
                    # batch_dets is already list[dict_per_sample]
                    all_predictions.extend(batch_dets)
                    # Use the processed sparse GTs
                    all_targets_sparse.extend(processed_sparse_gts_for_eval)
                    # ---

                    # --- 可视化 (unchanged) --- #
                    # ... (visualization code) ...

            # --- 循环结束后 --- #

            # ... (calculate and log average validation losses - unchanged) ...

            # --- 计算简化的 mAP --- #
            def calculate_simple_ap(preds_list, targets_list, num_classes, iou_threshold=0.5, dist_threshold=2.0):
                class_aps = {}
                map_score = 0.0
                for cls_id in range(num_classes):
                    # --- FIX: Restore TP/FP calculation logic --- #
                    tp = 0
                    fp = 0
                    total_gt_for_class = 0
                    for sample_preds, sample_targets in zip(preds_list, targets_list):
                        if 'box_cls' not in sample_targets or 'box_xyz' not in sample_targets:
                            print(
                                f"Warning: Skipping sample in AP calc due to missing keys in targets: {sample_targets.keys()}")
                            continue
                        if 'box_cls' not in sample_preds or 'box_xyz' not in sample_preds or 'box_scores' not in sample_preds:
                            print(
                                f"Warning: Skipping sample in AP calc due to missing keys in preds: {sample_preds.keys()}")
                            continue

                        gt_mask_sample = sample_targets['box_cls'] == cls_id
                        total_gt_for_class += gt_mask_sample.sum().item()

                        pred_mask_sample = sample_preds['box_cls'] == cls_id
                        num_preds_sample = pred_mask_sample.sum().item()

                        if num_preds_sample == 0:
                            continue

                        pred_boxes_sample = sample_preds['box_xyz'][pred_mask_sample]
                        pred_scores_sample = sample_preds['box_scores'][pred_mask_sample]
                        gt_boxes_sample = sample_targets['box_xyz'][gt_mask_sample]

                        if gt_boxes_sample.shape[0] == 0:
                            fp += num_preds_sample
                            continue

                        # Simplified distance-based matching
                        sample_tp = 0
                        sample_fp = 0
                        sorted_indices = torch.argsort(
                            pred_scores_sample, descending=True)
                        pred_boxes_sorted = pred_boxes_sample[sorted_indices]
                        gt_matched_flags = torch.zeros(
                            gt_boxes_sample.shape[0], dtype=torch.bool, device=device)

                        for p_idx, p_box in enumerate(pred_boxes_sorted):
                            match_found_for_pred = False
                            if gt_boxes_sample.shape[0] > 0:
                                distances = torch.norm(
                                    gt_boxes_sample - p_box.unsqueeze(0), p=2, dim=1)
                                min_dist, best_gt_idx = torch.min(
                                    distances, dim=0)

                                if min_dist < dist_threshold and not gt_matched_flags[best_gt_idx]:
                                    sample_tp += 1
                                    gt_matched_flags[best_gt_idx] = True
                                    match_found_for_pred = True

                            if not match_found_for_pred:
                                sample_fp += 1

                        tp += sample_tp
                        fp += sample_fp
                    # --- End TP/FP calculation logic for class --- #

                    # Calculate AP for the class
                    ap = 0.0
                    if total_gt_for_class == 0:
                        ap = 0.0
                    elif tp + fp == 0:
                        ap = 0.0
                    else:
                        precision = tp / (tp + fp)
                        recall = tp / total_gt_for_class
                        if (precision + recall) > 0:
                            ap = 2 * (precision * recall) / \
                                (precision + recall)
                        else:
                            ap = 0.0
                    class_aps[cls_id] = ap if not np.isnan(ap) else 0.0

                valid_aps = [v for v in class_aps.values() if not np.isnan(v)]
                map_score = sum(valid_aps) / \
                    len(valid_aps) if valid_aps else 0.0
                print(
                    f"Simplified AP calculation results (dist_thresh={dist_threshold}): mAP: {map_score:.4f}")
                return map_score, class_aps

            # 调用简化的AP计算
            # --- 修改：使用 all_targets_sparse --- #
            if all_predictions and all_targets_sparse:
                map_dist_2, class_aps_dist_2 = calculate_simple_ap(
                    all_predictions, all_targets_sparse, num_classes, dist_threshold=2.0)
                map_dist_1, class_aps_dist_1 = calculate_simple_ap(
                    all_predictions, all_targets_sparse, num_classes, dist_threshold=1.0)
            # ---
                val_map_score = map_dist_2  # 使用2米距离阈值作为主要指标

                # ... (log mAP scores, print, save best model based on mAP - unchanged) ...
            else:
                print(
                    "Warning: No predictions/targets collected during validation, skipping simple mAP calculation.")
                val_map_score = 0.0  # Assign a default value

            # --- 简化评估结束 --- #

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
                 amp=True,  # 启用混合精度训练
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
                 bsz=1,  # 减小批量大小
                 nworkers=0,
                 lr=1e-3,
                 weight_decay=1e-7,
                 num_classes=10,
                 lidar_channels=18,
                 enable_multiscale=True,
                 use_enhanced_fusion=True,
                 grad_accum_steps=2):  # 添加梯度累积步数
    """
    训练用于3D目标检测的多模态融合模型
    结合相机和LiDAR特征进行训练
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

    # 使用多模态融合模型
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
                          outC=total_output_channels,  # 传递总通道数
                          model='fusion',  # 使用融合模型
                          num_classes=actual_num_classes,
                          lidar_channels=lidar_channels)  # 传递LiDAR通道数
    # --- 修改结束 ---

    # 记录模型配置
    model_configs = {
        'grid_conf': grid_conf,
        'data_aug_conf': data_aug_conf,
        'num_classes': actual_num_classes,
        'enable_multiscale': enable_multiscale,
        'use_enhanced_fusion': use_enhanced_fusion,
        'lidar_channels': lidar_channels
    }

    model.to(device)

    if cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True

    # 优化器设置
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    # 设置学习率调度器 - 使用余弦退火调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepochs * len(trainloader),
        pct_start=0.2, div_factor=10, final_div_factor=10,
        anneal_strategy='cos'
    )

    # 损失函数
    from .tools import DetectionBEVLoss
    loss_fn = DetectionBEVLoss(num_classes=num_classes,
                               cls_weight=2.0,  # Increased weight from 1.0
                               iou_weight=1.0).to(device)

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

    # 恢复训练
    if resume != '':
        print("Loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['net'])
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        best_map = checkpoint.get('best_map', 0)

    if load_weight != '':
        print("Loading weight '{}'".format(load_weight))
        checkpoint = torch.load(load_weight)
        model.load_state_dict(checkpoint['net'], strict=False)

    # 自动混合精度训练
    if amp and torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_iou_loss = 0
        # Add accumulators for new regression loss components

        # 多尺度训练相关指标
        if enable_multiscale:
            epoch_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list, sparse_gts_batch, sample_tokens) in pbar:
            t0 = time.time()
            # 只有在新的累积周期开始时才清零梯度
            if batchi % grad_accum_steps == 0:
                opt.zero_grad()

            # 处理当前批次
            # 使用autocast进行混合精度训练
            with autocast(enabled=amp) if scaler else nullcontext():
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              lidar_bev.to(device)
                              )

                # Ensure targets is a dictionary and move to device (for loss calculation)
                targets_dict = {}
                # targets_list is the dictionary yielded by the DataLoader
                for key, val in targets_list.items():
                    if isinstance(val, torch.Tensor):
                        targets_dict[key] = val.to(device)
                    else:
                        targets_dict[key] = val

                # Calculate loss using the correctly processed dictionary
                losses = loss_fn(preds, targets_dict)

                # --- FIX: Define loss components before scaling/accumulating --- #
                total_loss = losses['total_loss']
                cls_loss = losses['cls_loss']
                reg_loss = losses.get(
                    'reg_loss', torch.tensor(0.0, device=device))
                iou_loss = losses.get(
                    'iou_loss', torch.tensor(0.0, device=device))
                # ---

                # Scale loss for gradient accumulation
                # --- FIX: Ensure grad_accum_steps is defined and used --- #
                # grad_accum_steps is an argument to train_fusion, should be defined
                scaled_loss = total_loss / grad_accum_steps
                # ---

            # Backward pass with scaler
            if scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Step optimizer only after accumulating gradients
            # --- FIX: Ensure grad_accum_steps is defined and used --- #
            if (batchi + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                scheduler.step()

            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_iou_loss += iou_loss.item()

            # 多尺度训练相关指标
            if enable_multiscale and 'scale_losses' in losses:
                scale_losses = losses['scale_losses']
                for i, loss in enumerate(scale_losses):
                    epoch_scale_losses[i] += loss.item()

            counter += 1
            t1 = time.time()  # 确保是浮点数

            # 构建进度条描述
            desc = '||Epoch: [%d/%d]|Loss: %.4f|Cls: %.4f|Reg: %.4f|IoU: %.4f||' % (
                epoch + 1, nepochs, total_loss.item(), cls_loss.item(),
                reg_loss.item(), iou_loss.item()
            )

            # 如果启用多尺度训练，添加多尺度损失信息
            if enable_multiscale and 'scale_losses' in losses:
                desc += ' MS: [%.2f,%.2f,%.2f]' % (
                    scale_losses[0].item(),
                    scale_losses[1].item(),
                    scale_losses[2].item()
                )

            pbar.set_description(desc)

        # 每个epoch结束后记录训练损失
        writer.add_scalar('train/loss', epoch_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/cls_loss', epoch_cls_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/reg_loss', epoch_reg_loss /
                          len(trainloader), epoch + 1)
        writer.add_scalar('train/iou_loss', epoch_iou_loss /
                          len(trainloader), epoch + 1)

        # 记录多尺度训练损失
        if enable_multiscale:
            for i, loss in enumerate(epoch_scale_losses):
                writer.add_scalar(
                    f'train/scale{i+3}_loss', loss / len(trainloader), epoch + 1)

        # 记录学习率
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch + 1)

        # 保存最后的模型，用于恢复训练
        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_map': best_map,
            'model_configs': model_configs,
            'amp_scaler': scaler.state_dict() if scaler is not None else None,  # 修复None调用问题
        }
        last_model = os.path.join(path, "last.pt")
        torch.save(last_checkpoint, last_model)

        # 验证
        if (epoch + 1) % val_step == 0 and (epoch + 1) >= val_step:
            model.eval()
            val_loss = 0
            val_cls_loss = 0
            val_reg_loss = 0
            val_iou_loss = 0

            # 多尺度验证相关指标
            if enable_multiscale:
                val_scale_losses = [0, 0, 0]

            # === 修改：使用新的变量名收集解码结果和稀疏GT ===
            all_predictions = []
            all_targets_sparse = []  # 用于存储从dataloader加载的稀疏GT
            # ---

            visualized_this_epoch = False

            with torch.no_grad():
                # --- 修改：更新循环变量以接收 sparse_gts_batch --- #
                for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list, sparse_gts_batch, sample_tokens) in enumerate(valloader):
                    # ---
                    context_manager = nullcontext()
                    with context_manager:
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )

                        # Ensure targets is a dictionary and move to device (for loss calculation)
                        targets_dict = {}
                        for key, val in targets_list.items():
                            if isinstance(val, torch.Tensor):
                                targets_dict[key] = val.to(device)
                            else:
                                # Keep non-tensors as is (e.g., potential metadata)
                                targets_dict[key] = val

                        # 1. 计算和累加损失 (using dense targets_dict)
                        losses = loss_fn(preds, targets_dict)
                        # ... (accumulate validation losses - similar to train_3d) ...
                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        # Assuming reg_loss and iou_loss are present in fusion loss output
                        val_reg_loss += losses.get('reg_loss',
                                                   torch.tensor(0.0)).item()
                        val_iou_loss += losses.get('iou_loss',
                                                   torch.tensor(0.0)).item()
                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

                    # Decode predictions outside context
                    # --- FIX: Ensure preds_dict is defined correctly --- #
                    if not isinstance(preds, dict) and isinstance(preds, torch.Tensor):
                        B, C, H, W_bev = preds.shape
                        cls_channels = num_classes
                        reg_channels = 9
                        iou_channels = 1
                        expected_channels = cls_channels + reg_channels + iou_channels
                        preds_dict = {}  # Initialize
                        if C >= expected_channels:
                            preds_dict = {
                                'cls_pred': preds[:, :cls_channels],
                                'reg_pred': preds[:, cls_channels: cls_channels + reg_channels],
                                'iou_pred': preds[:, -iou_channels:]
                            }
                        else:
                            print(
                                f"Warning: Pred tensor channels ({C}) < expected ({expected_channels}). Skipping decode.")
                            preds_dict = {'cls_pred': torch.empty(B, 0, H, W_bev), 'reg_pred': torch.empty(
                                B, 0, H, W_bev), 'iou_pred': torch.empty(B, 0, H, W_bev)}
                        preds = preds_dict
                    # ---

                    from .evaluate_3d import decode_predictions  # Reuse decode
                    batch_dets = decode_predictions(
                        preds, device, score_thresh=0.2, grid_conf=grid_conf)

                    # === 修改：直接使用从dataloader加载的 sparse_gts_batch ===
                    processed_sparse_gts_for_eval = []
                    for sample_gt_list in sparse_gts_batch:
                        if not sample_gt_list:
                            processed_sparse_gts_for_eval.append({
                                'box_cls': torch.empty(0, dtype=torch.long, device=device),
                                'box_xyz': torch.empty(0, 3, device=device),
                                'box_wlh': torch.empty(0, 3, device=device),
                                'box_rot_sincos': torch.empty(0, 2, device=device),
                                'box_vel': torch.empty(0, 2, device=device)
                            })
                            continue
                        keys = sample_gt_list[0].keys()
                        sample_batched_gts = {}
                        for k in keys:
                            sample_batched_gts[k] = torch.stack(
                                [gt[k] for gt in sample_gt_list], dim=0).to(device)
                        processed_sparse_gts_for_eval.append(
                            sample_batched_gts)
                    # --- End sparse GT processing --- #

                    # Extend lists using NMS preds and sparse GTs
                    all_predictions.extend(batch_dets)
                    # FIX: Use the defined variable
                    all_targets_sparse.extend(processed_sparse_gts_for_eval)

                    # ... (visualization) ...

            # Outside validation loop
            # ... (calculate and log average validation losses - unchanged) ...

            # Calculate simplified mAP
            def calculate_simple_ap(preds_list, targets_list, num_classes, iou_threshold=0.5, dist_threshold=2.0):
                class_aps = {}
                map_score = 0.0
                for cls_id in range(num_classes):
                    # --- Restore TP/FP calculation logic --- #
                    tp = 0
                    fp = 0
                    total_gt_for_class = 0
                    for sample_preds, sample_targets in zip(preds_list, targets_list):
                        if 'box_cls' not in sample_targets or 'box_xyz' not in sample_targets:
                            print(
                                f"Warning: Skipping sample in AP calc due to missing keys in targets: {sample_targets.keys()}")
                            continue
                        if 'box_cls' not in sample_preds or 'box_xyz' not in sample_preds or 'box_scores' not in sample_preds:
                            print(
                                f"Warning: Skipping sample in AP calc due to missing keys in preds: {sample_preds.keys()}")
                            continue

                        gt_mask_sample = sample_targets['box_cls'] == cls_id
                        total_gt_for_class += gt_mask_sample.sum().item()

                        pred_mask_sample = sample_preds['box_cls'] == cls_id
                        num_preds_sample = pred_mask_sample.sum().item()

                        if num_preds_sample == 0:
                            continue

                        pred_boxes_sample = sample_preds['box_xyz'][pred_mask_sample]
                        pred_scores_sample = sample_preds['box_scores'][pred_mask_sample]
                        gt_boxes_sample = sample_targets['box_xyz'][gt_mask_sample]

                        if gt_boxes_sample.shape[0] == 0:
                            fp += num_preds_sample
                            continue

                        # Simplified distance-based matching
                        sample_tp = 0
                        sample_fp = 0
                        sorted_indices = torch.argsort(
                            pred_scores_sample, descending=True)
                        pred_boxes_sorted = pred_boxes_sample[sorted_indices]
                        gt_matched_flags = torch.zeros(
                            gt_boxes_sample.shape[0], dtype=torch.bool, device=device)

                        for p_idx, p_box in enumerate(pred_boxes_sorted):
                            match_found_for_pred = False
                            if gt_boxes_sample.shape[0] > 0:
                                distances = torch.norm(
                                    gt_boxes_sample - p_box.unsqueeze(0), p=2, dim=1)
                                min_dist, best_gt_idx = torch.min(
                                    distances, dim=0)
                                if min_dist < dist_threshold and not gt_matched_flags[best_gt_idx]:
                                    sample_tp += 1
                                    gt_matched_flags[best_gt_idx] = True
                                    match_found_for_pred = True
                            if not match_found_for_pred:
                                sample_fp += 1

                        tp += sample_tp
                        fp += sample_fp
                    # --- End TP/FP calculation logic --- #

                    ap = 0.0
                    if total_gt_for_class == 0:
                        ap = 0.0
                    elif tp + fp == 0:
                        ap = 0.0
                    else:
                        precision = tp / (tp + fp)
                        recall = tp / total_gt_for_class
                        if (precision + recall) > 0:
                            ap = 2 * (precision * recall) / \
                                (precision + recall)
                        else:
                            ap = 0.0
                    class_aps[cls_id] = ap if not np.isnan(ap) else 0.0

                valid_aps = [v for v in class_aps.values() if not np.isnan(v)]
                map_score = sum(valid_aps) / \
                    len(valid_aps) if valid_aps else 0.0
                print(
                    f"Simplified AP calculation results (dist_thresh={dist_threshold}): mAP: {map_score:.4f}")
                return map_score, class_aps

            # Call calculate_simple_ap and use results
            if all_predictions and all_targets_sparse:
                map_dist_2, class_aps_dist_2 = calculate_simple_ap(
                    all_predictions, all_targets_sparse, num_classes, dist_threshold=2.0)
                map_dist_1, class_aps_dist_1 = calculate_simple_ap(
                    all_predictions, all_targets_sparse, num_classes, dist_threshold=1.0)
                val_map_score = map_dist_2

                # Log and print results
                writer.add_scalar('val/simple_mAP_dist_2m',
                                  val_map_score, epoch + 1)
                writer.add_scalar('val/simple_mAP_dist_1m',
                                  map_dist_1, epoch + 1)
                print('||Val Loss: %.4f | Simple mAP@dist=2m: %.4f | Simple mAP@dist=1m: %.4f||' %
                      (val_loss, val_map_score, map_dist_1))

                # Save best model based on mAP
                if val_map_score > best_map:
                    best_map = val_map_score
                    best_checkpoint = {
                        'net': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_map': best_map,
                        'model_configs': model_configs,
                        'amp_scaler': scaler.state_dict() if scaler is not None else None,
                    }
                    best_model_path = os.path.join(path, "best_map.pt")
                    torch.save(best_checkpoint, best_model_path)
                    print(f"Best model saved... mAP@dist=2m: {best_map:.4f}")
            else:
                print(
                    "Warning: No predictions/targets collected during validation, skipping simple mAP calculation.")
                val_map_score = 0.0  # Assign default value if no calculation happened

        epoch += 1
        pbar.close()

    writer.close()
