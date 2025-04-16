"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

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
from .tools import SimpleLoss, get_batch_iou, get_val_info
from contextlib import nullcontext  # 导入nullcontext


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

    model.to(device)
    if cuDNN:
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    loss_fn = SimpleLoss(pos_weight).cuda(gpuid)

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
             nepochs=10000,
             gpuid=0,
             cuDNN=False,
             resume='',
             load_weight='',
             amp=True,  # 默认启用混合精度训练，但在验证时需要注意类型一致性
             H=900, W=1600,
             resize_lim=(0.193, 0.225),
             final_dim=(128, 352),
             bot_pct_lim=(0.0, 0.22),
             rot_lim=(-5.4, 5.4),
             rand_flip=True,
             ncams=5,
             max_grad_norm=5.0,
             logdir='./runs_3d',

             xbound=[-50.0, 50.0, 0.5],
             ybound=[-50.0, 50.0, 0.5],
             zbound=[-10.0, 10.0, 20.0],
             dbound=[4.0, 45.0, 1.0],

             bsz=4,
             nworkers=0,
             lr=1e-3,
             weight_decay=1e-7,
             num_classes=10,
             enable_multiscale=True,  # 支持多尺度特征训练
             use_enhanced_bev=True,   # 使用增强的BEV投影
             ):
    """
    训练用于3D目标检测的BEVENet模型
    增加多尺度特征训练支持和增强的BEV投影
    """
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
    model = compile_model(grid_conf, data_aug_conf,
                          outC=actual_num_classes, model='beve',
                          num_classes=actual_num_classes)

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
                           weight_decay=weight_decay)

    # 设置学习率调度器 - 使用余弦退火调度
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=nepochs * len(trainloader),
        pct_start=0.2, div_factor=10, final_div_factor=10,
        anneal_strategy='cos'  # 使用余弦退火
    )

    # 检查如果启用了多尺度训练，打印一条警告信息但不影响程序运行

    # 损失函数
    from .tools import Detection3DLoss
    # 初始化损失函数，只使用支持的参数
    loss_fn = Detection3DLoss(num_classes=num_classes,
                              cls_weight=1.0,
                              reg_weight=1.0,
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
    if amp:
        # 使用正确的API，注意torch.cuda.amp仍然有效
        scaler = GradScaler() if torch.cuda.is_available() else None
    else:
        scaler = None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        epoch_iou_loss = 0

        # 多尺度训练相关指标
        if enable_multiscale:
            epoch_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list) in pbar:
            t0 = time.time()  # 确保是浮点数
            # 只有在新的累积周期开始时才清零梯度
            opt.zero_grad()

            # 处理当前批次
            # 使用autocast进行混合精度训练
            with autocast(enabled=amp) if amp and torch.cuda.is_available() else nullcontext():
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )

                # Unpack the list and create the target dictionary
                cls_map = targets_list[0].to(device)        # 类别地图
                reg_map = targets_list[1].to(device)        # 回归地图
                reg_weight = targets_list[2].to(device)     # 回归权重
                iou_map = targets_list[3].to(device)        # IoU地图

                targets = {
                    'cls_targets': cls_map,
                    'reg_targets': reg_map,
                    'reg_weights': reg_weight,
                    'iou_targets': iou_map
                }

                losses = loss_fn(preds, targets)

                total_loss = losses['total_loss']
                cls_loss = losses['cls_loss']
                reg_loss = losses['reg_loss']
                iou_loss = losses['iou_loss']

                # 多尺度损失
                if enable_multiscale and 'scale_losses' in losses:
                    scale_losses = losses['scale_losses']

            # 使用混合精度训练
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)  # 反向传播前解缩放，以便进行梯度裁剪

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler:
                scaler.step(opt)
                scaler.update()
                scheduler.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_iou_loss += iou_loss.item()

            # 多尺度训练相关指标
            if enable_multiscale and 'scale_losses' in losses:
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
            'model_configs': model_configs,  # 保存模型配置
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
                val_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

            with torch.no_grad():
                for imgs, rots, trans, intrins, post_rots, post_trans, targets_list in valloader:
                    # 使用autocast进行混合精度训练
                    with autocast(enabled=amp) if amp and torch.cuda.is_available() else nullcontext():
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )

                        # Unpack the list and create the target dictionary
                        cls_map = targets_list[0].to(device)        # 类别地图
                        reg_map = targets_list[1].to(device)        # 回归地图
                        reg_weight = targets_list[2].to(device)     # 回归权重
                        iou_map = targets_list[3].to(device)        # IoU地图

                        targets = {
                            'cls_targets': cls_map,
                            'reg_targets': reg_map,
                            'reg_weights': reg_weight,
                            'iou_targets': iou_map
                        }

                        losses = loss_fn(preds, targets)

                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        val_reg_loss += losses['reg_loss'].item()
                        val_iou_loss += losses['iou_loss'].item()

                        # 多尺度验证相关指标
                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

            # 计算平均损失
            val_loss /= len(valloader)
            val_cls_loss /= len(valloader)
            val_reg_loss /= len(valloader)
            val_iou_loss /= len(valloader)

            # 记录验证损失
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/cls_loss', val_cls_loss, epoch + 1)
            writer.add_scalar('val/reg_loss', val_reg_loss, epoch + 1)
            writer.add_scalar('val/iou_loss', val_iou_loss, epoch + 1)

            # 记录多尺度验证损失
            if enable_multiscale:
                for i, loss in enumerate(val_scale_losses):
                    avg_loss = loss / len(valloader)
                    writer.add_scalar(
                        f'val/scale{i+3}_loss', avg_loss, epoch + 1)

            # 评估mAP
            from .evaluate_3d import compute_map, calculate_map

            # 使用compute_map获取预测结果和目标信息
            with torch.no_grad():
                all_predictions = []
                all_targets = []

                for imgs, rots, trans, intrins, post_rots, post_trans, targets_list in valloader:
                    # 不使用混合精度进行评估，确保数据类型一致性
                    preds = model(imgs.to(device),
                                  rots.to(device),
                                  trans.to(device),
                                  intrins.to(device),
                                  post_rots.to(device),
                                  post_trans.to(device))

                    # 确保预测结果为float32类型，避免类型不匹配
                    if hasattr(preds, 'float'):
                        preds = preds.float()
                    elif isinstance(preds, dict):
                        for k in preds:
                            if hasattr(preds[k], 'float'):
                                preds[k] = preds[k].float()
                    elif isinstance(preds, (list, tuple)):
                        preds = [p.float() if hasattr(p, 'float')
                                 else p for p in preds]

                    # 解码预测和目标
                    from .evaluate_3d import decode_predictions, decode_targets
                    batch_dets = decode_predictions(preds, device)
                    batch_gts = decode_targets(targets_list, device)

                    all_predictions.extend(batch_dets)
                    all_targets.extend(batch_gts)

            # 计算mAP，确保结果类型正确
            try:
                val_results = calculate_map(all_predictions, all_targets,
                                            iou_thresholds=[0.5, 0.7],
                                            num_classes=num_classes,
                                            consider_rotation=True)

                # 记录多个IoU阈值下的mAP
                writer.add_scalar(
                    'val/mAP@0.5', float(val_results['mAP@0.5']), epoch + 1)
                writer.add_scalar(
                    'val/mAP@0.7', float(val_results['mAP@0.7']), epoch + 1)

                # 记录每个类别的AP，确保数据类型正确
                if isinstance(val_results['class_aps'], dict) and 'AP@0.5' in val_results['class_aps']:
                    for cls_id, ap in val_results['class_aps']['AP@0.5'].items():
                        if isinstance(val_results.get('class_names', {}), dict):
                            cls_name = val_results['class_names'].get(
                                cls_id, f"Class {cls_id}")
                        else:
                            cls_name = f"Class {cls_id}"
                        writer.add_scalar(
                            f'val/AP@0.5/{cls_name}', float(ap), epoch + 1)

                # 使用0.5阈值的mAP作为主要指标
                val_map_score = float(val_results['mAP@0.5'])
            except Exception as e:
                print(f"计算mAP时出错: {e}")
                val_map_score = 0.0
                val_results = {'mAP@0.5': 0.0, 'mAP@0.7': 0.0}  # 创建默认结果字典

            print('||Val Loss: %.4f|mAP@0.5: %.4f|mAP@0.7: %.4f||' %
                  (val_loss, val_map_score, float(val_results['mAP@0.7'])))

            # 保存最佳模型
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
                best_model = os.path.join(path, "best.pt")
                torch.save(best_checkpoint, best_model)
                print(f"Best model saved with mAP: {best_map:.4f}")

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
    model = compile_model(grid_conf, data_aug_conf,
                          outC=actual_num_classes*9,  # 每个类别9个通道
                          model='fusion',  # 使用融合模型
                          num_classes=actual_num_classes,
                          lidar_channels=lidar_channels)  # 传递LiDAR通道数

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
    from .tools import Detection3DLoss
    loss_fn = Detection3DLoss(num_classes=num_classes,
                              cls_weight=1.0,
                              reg_weight=1.0,
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
    if amp:
        # 使用正确的API
        scaler = GradScaler() if torch.cuda.is_available() else None
    else:
        scaler = None

    while epoch < nepochs:
        np.random.seed()
        model.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_reg_loss = 0
        epoch_iou_loss = 0

        # 多尺度训练相关指标
        if enable_multiscale:
            epoch_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list) in pbar:
            t0 = time.time()
            # 只有在新的累积周期开始时才清零梯度
            if batchi % grad_accum_steps == 0:
                opt.zero_grad()

            # 处理当前批次
            # 使用autocast进行混合精度训练
            with autocast(enabled=amp) if amp and torch.cuda.is_available() else nullcontext():
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              lidar_bev.to(device)
                              )

                # Unpack the list and create the target dictionary
                cls_map = targets_list[0].to(device)        # 类别地图
                reg_map = targets_list[1].to(device)        # 回归地图
                reg_weight = targets_list[2].to(device)     # 回归权重
                iou_map = targets_list[3].to(device)        # IoU地图

                targets = {
                    'cls_targets': cls_map,
                    'reg_targets': reg_map,
                    'reg_weights': reg_weight,
                    'iou_targets': iou_map
                }

                losses = loss_fn(preds, targets)

                total_loss = losses['total_loss']
                cls_loss = losses['cls_loss']
                reg_loss = losses['reg_loss']
                iou_loss = losses['iou_loss']

                # 根据累积步数缩放损失
                scaled_loss = total_loss / grad_accum_steps

                # 多尺度损失
                if enable_multiscale and 'scale_losses' in losses:
                    scale_losses = losses['scale_losses']

            # 使用混合精度训练
            if scaler:
                scaler.scale(scaled_loss).backward()

            # 只有在累积周期结束时才更新参数
            if (batchi + 1) % grad_accum_steps == 0 or (batchi + 1) == len(trainloader):
                if scaler:
                    scaler.unscale_(opt)  # 反向传播前解缩放，以便进行梯度裁剪
                    # 梯度裁剪，避免梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()
            epoch_iou_loss += iou_loss.item()

            # 多尺度训练相关指标
            if enable_multiscale and 'scale_losses' in losses:
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
                val_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

            with torch.no_grad():
                for imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list in valloader:
                    # 使用autocast进行混合精度训练
                    with autocast(enabled=amp) if amp and torch.cuda.is_available() else nullcontext():
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      lidar_bev.to(device))

                        # Unpack the list and create the target dictionary
                        cls_map = targets_list[0].to(device)        # 类别地图
                        reg_map = targets_list[1].to(device)        # 回归地图
                        reg_weight = targets_list[2].to(device)     # 回归权重
                        iou_map = targets_list[3].to(device)        # IoU地图

                        targets = {
                            'cls_targets': cls_map,
                            'reg_targets': reg_map,
                            'reg_weights': reg_weight,
                            'iou_targets': iou_map
                        }

                        losses = loss_fn(preds, targets)

                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        val_reg_loss += losses['reg_loss'].item()
                        val_iou_loss += losses['iou_loss'].item()

                        # 多尺度验证相关指标
                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

            # 计算平均损失
            val_loss /= len(valloader)
            val_cls_loss /= len(valloader)
            val_reg_loss /= len(valloader)
            val_iou_loss /= len(valloader)

            # 记录验证损失
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/cls_loss', val_cls_loss, epoch + 1)
            writer.add_scalar('val/reg_loss', val_reg_loss, epoch + 1)
            writer.add_scalar('val/iou_loss', val_iou_loss, epoch + 1)

            # 记录多尺度验证损失
            if enable_multiscale:
                for i, loss in enumerate(val_scale_losses):
                    avg_loss = loss / len(valloader)
                    writer.add_scalar(
                        f'val/scale{i+3}_loss', avg_loss, epoch + 1)

            # 评估mAP
            from .evaluate_3d import compute_map, calculate_map

            # 使用compute_map获取预测结果和目标信息
            with torch.no_grad():
                all_predictions = []
                all_targets = []

                for imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list in valloader:
                    # 不使用混合精度进行评估，确保数据类型一致性
                    preds = model(imgs.to(device),
                                  rots.to(device),
                                  trans.to(device),
                                  intrins.to(device),
                                  post_rots.to(device),
                                  post_trans.to(device),
                                  lidar_bev.to(device))

                    # 确保预测结果为float32类型，避免类型不匹配
                    if hasattr(preds, 'float'):
                        preds = preds.float()
                    elif isinstance(preds, dict):
                        for k in preds:
                            if hasattr(preds[k], 'float'):
                                preds[k] = preds[k].float()
                    elif isinstance(preds, (list, tuple)):
                        preds = [p.float() if hasattr(p, 'float')
                                 else p for p in preds]

                    # 解码预测和目标
                    from .evaluate_3d import decode_predictions, decode_targets
                    batch_dets = decode_predictions(preds, device)
                    batch_gts = decode_targets(targets_list, device)

                    all_predictions.extend(batch_dets)
                    all_targets.extend(batch_gts)

            # 计算mAP，确保结果类型正确
            try:
                val_results = calculate_map(all_predictions, all_targets,
                                            iou_thresholds=[0.5, 0.7],
                                            num_classes=num_classes,
                                            consider_rotation=True)

                # 记录多个IoU阈值下的mAP
                writer.add_scalar(
                    'val/mAP@0.5', float(val_results['mAP@0.5']), epoch + 1)
                writer.add_scalar(
                    'val/mAP@0.7', float(val_results['mAP@0.7']), epoch + 1)

                # 记录每个类别的AP，确保数据类型正确
                if isinstance(val_results['class_aps'], dict) and 'AP@0.5' in val_results['class_aps']:
                    for cls_id, ap in val_results['class_aps']['AP@0.5'].items():
                        if isinstance(val_results.get('class_names', {}), dict):
                            cls_name = val_results['class_names'].get(
                                cls_id, f"Class {cls_id}")
                        else:
                            cls_name = f"Class {cls_id}"
                        writer.add_scalar(
                            f'val/AP@0.5/{cls_name}', float(ap), epoch + 1)

                # 使用0.5阈值的mAP作为主要指标
                val_map_score = float(val_results['mAP@0.5'])
            except Exception as e:
                print(f"计算mAP时出错: {e}")
                val_map_score = 0.0
                val_results = {'mAP@0.5': 0.0, 'mAP@0.7': 0.0}  # 创建默认结果字典

            print('||Val Loss: %.4f|mAP@0.5: %.4f|mAP@0.7: %.4f||' %
                  (val_loss, val_map_score, float(val_results['mAP@0.7'])))

            # 保存最佳模型
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
                best_model = os.path.join(path, "best.pt")
                torch.save(best_checkpoint, best_model)
                print(f"Best model saved with mAP: {best_map:.4f}")

        epoch += 1
        pbar.close()

    writer.close()
