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
from .nuscenes_info import load_nuscenes_infos  # 导入数据集缓存加载函数


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
             load_weight='',
             amp=True,  # 默认启用混合精度训练，但在验证时需要注意类型一致性
             H=900, W=1600,
             resize_lim=(0.193, 0.225),
             final_dim=(128, 352),
             bot_pct_lim=(0.0, 0.22),
             rot_lim=(-5.4, 5.4),
             rand_flip=True,
             ncams=6,
             max_grad_norm=5.0,
             logdir='./runs_3d',

             xbound=[-50.0, 50.0, 0.5],
             ybound=[-50.0, 50.0, 0.5],
             zbound=[-5.0, 3.0, 0.5],
             dbound=[4.0, 45.0, 1.0],

             bsz=4,
             nworkers=4,
             lr=1e-3,
             weight_decay=5e-5,
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
                               cls_weight=1.0,
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

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sample_tokens) in pbar:
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

                # Ensure targets is a dictionary and move to device
                targets_dict = {}
                # targets_list is the dictionary yielded by the DataLoader
                for key, val in targets_list.items():
                    if isinstance(val, torch.Tensor):
                        targets_dict[key] = val.to(device)
                    else:
                        targets_dict[key] = val

                # Calculate loss using the correctly processed dictionary
                losses = loss_fn(preds, targets_dict)

                total_loss = losses['total_loss']
                cls_loss = losses['cls_loss']
                iou_loss = losses.get('iou_loss', torch.tensor(0.0))
                # Get individual regression losses
                bev_diou_loss = losses.get('bev_diou_loss', torch.tensor(0.0))
                z_loss = losses.get('z_loss', torch.tensor(0.0))
                h_loss = losses.get('h_loss', torch.tensor(0.0))
                vel_loss = losses.get('vel_loss', torch.tensor(0.0))

                # 多尺度损失
                if enable_multiscale and 'scale_losses' in losses:
                    scale_losses = losses['scale_losses']

            # 使用混合精度训练
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(opt)  # 反向传播前解缩放，以便进行梯度裁剪

            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Step optimizer and scheduler
            if scaler:
                scaler.step(opt)   # Calls optimizer.step() implicitly
                scaler.update()
            else:  # Handle case where AMP is disabled
                opt.step()

            # Scheduler step should happen after optimizer step
            # This should be outside the scaler check, called every step where optimizer is stepped
            scheduler.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_iou_loss += iou_loss.item()
            # Accumulate new regression losses
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
        # Log new regression losses
        writer.add_scalar('train/bev_diou_loss',
                          epoch_bev_diou_loss / len(trainloader), epoch + 1)
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
            val_iou_loss = 0
            val_bev_diou_loss = 0.0
            val_z_loss = 0.0
            val_h_loss = 0.0
            val_vel_loss = 0.0

            # 多尺度验证相关指标
            if enable_multiscale:
                val_scale_losses = [0, 0, 0]

            # === 新增：用于收集解码结果 ===
            all_predictions = []
            all_targets = []
            # ---

            visualized_this_epoch = False  # Flag to visualize only once per epoch

            with torch.no_grad():
                # 只遍历一次验证集
                for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sample_tokens) in enumerate(valloader):
                    # 使用autocast进行混合精度训练 (如果验证时也启用AMP)
                    # 注意：为了评估指标的稳定性，通常建议验证时不使用autocast，除非内存非常受限
                    context_manager = autocast(
                        enabled=amp) if amp and torch.cuda.is_available() else nullcontext()
                    with context_manager:
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )

                        # Ensure targets is a dictionary and move to device
                        targets_dict = {}
                        for key, val in targets_list.items():
                            if isinstance(val, torch.Tensor):
                                targets_dict[key] = val.to(device)
                            else:
                                # Keep non-tensors as is
                                targets_dict[key] = val

                        # 1. 计算和累加损失
                        losses = loss_fn(preds, targets_dict)

                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        val_iou_loss += losses.get('iou_loss',
                                                   torch.tensor(0.0)).item()
                        val_bev_diou_loss += losses.get(
                            'bev_diou_loss', torch.tensor(0.0)).item()
                        val_z_loss += losses.get('z_loss',
                                                 torch.tensor(0.0)).item()
                        val_h_loss += losses.get('h_loss',
                                                 torch.tensor(0.0)).item()
                        val_vel_loss += losses.get('vel_loss',
                                                   torch.tensor(0.0)).item()

                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

                    # --- 在autocast之外解码，确保使用float32进行解码 ---
                    # 确保preds是预期的字典格式 (如果模型输出不是字典需要转换)
                    if not isinstance(preds, dict) and isinstance(preds, torch.Tensor):
                        B, C, H, W = preds.shape
                        cls_channels = num_classes
                        reg_channels = 9  # x,y,z,w,l,h,sin,cos,vel - 检查是否与模型输出匹配
                        iou_channels = 1  # 检查是否与模型输出匹配

                        # 检查通道数是否足够
                        expected_channels = cls_channels + reg_channels + iou_channels
                        if C < expected_channels:
                            print(
                                f"Warning: Unexpected number of channels in prediction tensor ({C}). Expected at least {expected_channels}. Skipping decoding for this batch.")
                            continue  # 跳过这个批次的解码

                        preds_dict = {
                            'cls_pred': preds[:, :cls_channels],
                            # 注意：这里假设回归头预测所有类的参数，需要根据实际模型调整
                            # 如果模型为每个类输出独立的回归头，结构会不同
                            # 例如，如果形状是 B, (cls+reg+iou)*num_tasks, H, W
                            # 假设共享回归头
                            'reg_pred': preds[:, cls_channels:cls_channels+reg_channels],
                            'iou_pred': preds[:, -iou_channels:]  # 假设共享IoU头
                        }
                        #  如果你的模型为每个类别输出了独立的回归头，需要调整这里的解析逻辑
                        # 例如: preds_dict['reg_pred'] = preds[:, cls_channels:-iou_channels].view(B, num_classes, reg_channels, H, W)

                        preds = preds_dict  # 更新为字典格式

                    # 2. 解码预测和目标 (在循环内部进行)
                    # 确保 evaluate_3d 或当前文件中有 decode_predictions
                    from .evaluate_3d import decode_predictions  # 仅导入 decode_predictions
                    # score_thresh 应该是一个超参数，这里暂时用0.2
                    # --- 修改：降低 score_thresh 用于调试 ---
                    debug_score_thresh = 0.01
                    print(
                        f"DEBUG: Using lowered score_thresh={debug_score_thresh} for decode_predictions")
                    batch_dets = decode_predictions(
                        preds, device, score_thresh=debug_score_thresh, grid_conf=grid_conf)
                    # --- 修改结束 ---

                    # --- Debug: Print keys of targets_dict (保留用于调试) ---
                    # print(f"DEBUG: Keys in targets_dict: {list(targets_dict.keys())}")
                    # --- End Debug ---

                    # --- 手动处理 GT ---
                    # 从 targets_dict 中提取 GT box 信息，格式与 batch_dets 保持一致
                    batch_gts = []
                    # 假设 targets_dict 包含 'cls_targets' [B, N_max] 和 'reg_targets' [B, N_max, 9]
                    gt_cls = targets_dict['cls_targets']  # [B, N_max]
                    gt_reg = targets_dict['reg_targets']  # [B, N_max, 9]
                    B = gt_cls.shape[0]

                    for b in range(B):
                        sample_cls = gt_cls[b]  # [N_max]
                        sample_reg = gt_reg[b]  # [N_max, 9]

                        # 过滤掉 padding (假设类别ID <= 0 为 padding)
                        valid_mask = sample_cls > 0
                        if valid_mask.sum() == 0:
                            # 如果没有 GT box，添加空字典
                            gts = {
                                'box_cls': torch.zeros(0, dtype=torch.long, device=device),
                                'box_xyz': torch.zeros(0, 3, device=device),
                                'box_wlh': torch.zeros(0, 3, device=device),
                                'box_rot_sincos': torch.zeros(0, 2, device=device),
                                'box_vel': torch.zeros(0, 2, device=device)
                            }
                        else:
                            valid_cls = sample_cls[valid_mask]
                            valid_reg = sample_reg[valid_mask]  # [N_valid, 9]

                            # 从 valid_reg 中提取坐标、尺寸、旋转、速度
                            # 索引需要与 decode_predictions 中的回归头输出顺序一致!
                            # 假设顺序: x,y,z, w,l,h, sin,cos, vel_xy
                            gt_xyz = valid_reg[:, :3]
                            gt_wlh = valid_reg[:, 3:6]
                            gt_rot_sincos = valid_reg[:, 6:8]
                            gt_vel = valid_reg[:, 8:]  # 可能只有1维或2维
                            # 确保速度为2维
                            if gt_vel.shape[1] == 1:
                                gt_vel = torch.cat(
                                    [gt_vel, torch.zeros_like(gt_vel)], dim=1)
                            elif gt_vel.shape[1] > 2:
                                gt_vel = gt_vel[:, :2]
                            elif gt_vel.shape[1] == 0:  # 处理没有速度的情况
                                gt_vel = torch.zeros(
                                    gt_vel.shape[0], 2, device=device)

                            gts = {
                                'box_cls': valid_cls,
                                # 'box_scores': torch.ones_like(valid_cls, dtype=torch.float), # GT 没有分数, 但AP计算可能需要
                                'box_xyz': gt_xyz,
                                'box_wlh': gt_wlh,
                                'box_rot_sincos': gt_rot_sincos,
                                'box_vel': gt_vel
                            }
                        batch_gts.append(gts)
                    # --- GT 处理结束 ---

                    # # 使用移动到device后的targets_dict (注释掉原始调用)
                    # batch_gts = decode_targets(targets_dict, device)

                    all_predictions.extend(batch_dets)
                    all_targets.extend(batch_gts)
                    # ---

                    # --- 可视化 (只在第一个验证批次进行一次) ---
                    if not visualized_this_epoch and batch_idx == 0:  # Visualize first batch
                        try:
                            vis_idx = 0  # Visualize the first sample in the batch
                            # 1. Visualize Input Image (Front Camera)
                            front_cam_idx = 1
                            if imgs.shape[1] > front_cam_idx:
                                # Ensure float
                                input_img_sample = imgs[vis_idx,
                                                        front_cam_idx].cpu().float()
                                writer.add_image(
                                    'val/input_front_camera', input_img_sample, global_step=epoch + 1)

                            # 2. Visualize BEV Prediction Heatmap
                            if 'cls_pred' in preds and preds['cls_pred'] is not None and preds['cls_pred'].numel() > 0:
                                cls_pred_sample = preds['cls_pred'][vis_idx].detach(
                                ).cpu().float()  # Ensure float
                                if cls_pred_sample.numel() > 0 and cls_pred_sample.dim() > 0:  # Check if not empty and has dimensions
                                    bev_heatmap = torch.max(torch.softmax(
                                        cls_pred_sample, dim=0), dim=0)[0]
                                    if bev_heatmap.numel() > 0:  # Check heatmap is not empty
                                        bev_heatmap = (
                                            bev_heatmap - bev_heatmap.min()) / (bev_heatmap.max() - bev_heatmap.min() + 1e-6)
                                        writer.add_image(
                                            'val/bev_prediction_heatmap', bev_heatmap.unsqueeze(0), global_step=epoch + 1)
                                else:
                                    print(
                                        f"Warning: cls_pred_sample for visualization is empty or invalid (shape: {cls_pred_sample.shape}).")
                            else:
                                print(
                                    "Warning: 'cls_pred' not found or empty in preds for visualization.")

                            # 3. Visualize Depth Map (Optional)
                            # ...

                            visualized_this_epoch = True
                        except Exception as e:
                            print(f"Warning: Failed to add visualization: {e}")
                    # --- 可视化结束 ---

            # --- 循环结束后 ---

            # 计算平均损失
            num_val_batches = len(valloader)
            if num_val_batches > 0:
                val_loss /= num_val_batches
                val_cls_loss /= num_val_batches
                val_iou_loss /= num_val_batches
                val_bev_diou_loss /= num_val_batches
                val_z_loss /= num_val_batches
                val_h_loss /= num_val_batches
                val_vel_loss /= num_val_batches
            else:
                print("Warning: Validation loader is empty.")
                # Set losses to NaN or zero if loader is empty
                val_loss, val_cls_loss, val_iou_loss = float(
                    'nan'), float('nan'), float('nan')
                val_bev_diou_loss, val_z_loss, val_h_loss, val_vel_loss = float(
                    'nan'), float('nan'), float('nan'), float('nan')

            # 记录验证损失
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/cls_loss', val_cls_loss, epoch + 1)
            writer.add_scalar('val/iou_loss', val_iou_loss, epoch + 1)
            writer.add_scalar('val/bev_diou_loss',
                              val_bev_diou_loss, epoch + 1)
            writer.add_scalar('val/z_loss', val_z_loss, epoch + 1)
            writer.add_scalar('val/h_loss', val_h_loss, epoch + 1)
            writer.add_scalar('val/vel_loss', val_vel_loss, epoch + 1)

            # 记录多尺度验证损失
            if enable_multiscale and num_val_batches > 0:
                for i, loss in enumerate(val_scale_losses):
                    avg_loss = loss / num_val_batches
                    writer.add_scalar(
                        f'val/scale{i+3}_loss', avg_loss, epoch + 1)

            # --- 计算简化的 mAP ---
            # 定义 calculate_simple_ap (如果尚未定义或导入)
            # 注意：这个函数需要在此作用域内可用

            def calculate_simple_ap(preds_list, targets_list, num_classes, iou_threshold=0.5, dist_threshold=2.0):
                """计算简化的平均精度 (基于距离匹配)"""
                if not preds_list or not targets_list:
                    print(
                        "Warning: Empty predictions or targets list for simple AP calculation.")
                    return 0.0, {}  # Return 0 mAP and empty class APs

                assert len(preds_list) == len(
                    targets_list), "Prediction and target list lengths must match"

                class_aps = {}
                # 从1到num_classes (不包括背景类0)
                for cls_id in range(1, num_classes + 1):
                    tp = 0
                    fp = 0
                    total_gt_for_class = 0
                    scores_for_class = []
                    match_for_class = []  # 0 for fp, 1 for tp

                    # 遍历每个样本
                    for sample_preds, sample_targets in zip(preds_list, targets_list):
                        # 当前样本中该类的GT数量
                        gt_mask_sample = sample_targets['box_cls'] == cls_id
                        total_gt_for_class += gt_mask_sample.sum().item()

                        # 当前样本中该类的预测
                        pred_mask_sample = sample_preds['box_cls'] == cls_id
                        num_preds_sample = pred_mask_sample.sum().item()

                        if num_preds_sample == 0:
                            continue  # 没有这个类的预测，跳到下一个样本

                        # 获取该类的预测框、得分和GT框
                        pred_boxes_sample = sample_preds['box_xyz'][pred_mask_sample]
                        pred_scores_sample = sample_preds['box_scores'][pred_mask_sample]
                        gt_boxes_sample = sample_targets['box_xyz'][gt_mask_sample]

                        scores_for_class.append(pred_scores_sample)

                        if gt_boxes_sample.shape[0] == 0:
                            # 这个样本没有该类的GT，所有预测都是FP
                            fp += num_preds_sample
                            match_for_class.extend([0] * num_preds_sample)
                            continue

                        # 使用距离进行匹配
                        # matched_gt = torch.zeros(gt_boxes_sample.shape[0], dtype=torch.bool, device=device)
                        # 简化：我们只需要统计TP/FP，不需要精确的AP曲线，所以可以用更简单的方式
                        # 对每个预测框，找到最近的GT框，如果距离小于阈值，则为TP (假设一对一匹配简化)
                        # TODO: 这种简化可能导致一个GT匹配多个预测，更精确的方法需要匈牙利算法或类似方法
                        sample_tp = 0
                        sample_fp = 0
                        # 对预测按分数排序（简化AP不需要，但保留逻辑）
                        sorted_indices = torch.argsort(
                            pred_scores_sample, descending=True)
                        pred_boxes_sorted = pred_boxes_sample[sorted_indices]

                        # 记录哪些GT已经被匹配过 (简化版)
                        gt_matched_flags = torch.zeros(
                            gt_boxes_sample.shape[0], dtype=torch.bool, device=device)

                        for p_box in pred_boxes_sorted:
                            min_dist = float('inf')
                            best_gt_idx = -1
                            match_found_for_pred = False

                            # Ensure GT boxes exist
                            if gt_boxes_sample.shape[0] > 0:
                                distances = torch.norm(
                                    gt_boxes_sample - p_box.unsqueeze(0), p=2, dim=1)
                                min_dist, best_gt_idx = torch.min(
                                    distances, dim=0)

                                # 检查距离和是否已被匹配
                                if min_dist < dist_threshold and not gt_matched_flags[best_gt_idx]:
                                    sample_tp += 1
                                    # 标记为已匹配
                                    gt_matched_flags[best_gt_idx] = True
                                    match_for_class.append(1)  # 标记为TP
                                    match_found_for_pred = True
                                else:
                                    sample_fp += 1
                                    match_for_class.append(0)  # 标记为FP
                            else:  # No GT boxes of this class in the sample
                                sample_fp += 1
                                match_for_class.append(0)  # 标记为FP

                        tp += sample_tp
                        fp += sample_fp

                    if total_gt_for_class == 0:
                        class_aps[cls_id] = 0.0  # 没有GT，AP为0
                        continue

                    if tp + fp == 0:
                        # 没有预测或者所有预测都匹配了GT (不太可能在真实场景发生)
                        # 如果 tp=0, fp=0, 没有预测，precision=0
                        # 如果 tp>0, fp=0, 所有预测都是TP，precision=1
                        precision = 1.0 if tp > 0 else 0.0
                    else:
                        precision = tp / (tp + fp)

                    recall = tp / total_gt_for_class if total_gt_for_class > 0 else 0.0

                    # 使用 F1 分数作为简化的 AP 替代 (或者直接用 Precision * Recall)
                    # ap = precision * recall
                    ap = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0.0
                    class_aps[cls_id] = ap if not np.isnan(
                        ap) else 0.0  # Handle potential NaN

                # 计算 mAP
                valid_aps = [v for v in class_aps.values() if not np.isnan(v)]
                map_score = sum(valid_aps) / \
                    len(valid_aps) if valid_aps else 0.0

                print(
                    f"Simplified AP calculation results (dist_thresh={dist_threshold}):")
                # print(f" Class APs: {class_aps}") # Optional: print per-class APs
                print(f" mAP: {map_score:.4f}")

                return map_score, class_aps  # 返回 mAP 和各类别 AP

            # 调用简化的AP计算
            # 确保 all_predictions 和 all_targets 不为空
            if all_predictions and all_targets:
                # 使用距离阈值0.5和2.0计算
                map_dist_2, class_aps_dist_2 = calculate_simple_ap(
                    all_predictions, all_targets, num_classes, dist_threshold=2.0)
                map_dist_1, class_aps_dist_1 = calculate_simple_ap(
                    all_predictions, all_targets, num_classes, dist_threshold=1.0)  # 更严格的阈值
                val_map_score = map_dist_2  # 使用2米距离阈值作为主要指标

                # 记录指标
                writer.add_scalar('val/simple_mAP_dist_2m',
                                  val_map_score, epoch + 1)
                writer.add_scalar('val/simple_mAP_dist_1m',
                                  map_dist_1, epoch + 1)  # 记录1米阈值的结果

                print('||Val Loss: %.4f | Simple mAP@dist=2m: %.4f | Simple mAP@dist=1m: %.4f||' %
                      (val_loss, val_map_score, map_dist_1))

                # 保存最佳模型 (基于 dist=2m 的 mAP)
                if val_map_score > best_map:
                    best_map = val_map_score
                    best_checkpoint = {
                        'net': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_map': best_map,  # Store the best mAP score
                        'model_configs': model_configs,
                        'amp_scaler': scaler.state_dict() if scaler is not None else None,
                    }
                    best_model_path = os.path.join(
                        path, "best_map.pt")  # Use a distinct name
                    torch.save(best_checkpoint, best_model_path)
                    print(
                        f"Best model saved to {best_model_path} with simple mAP@dist=2m: {best_map:.4f}")
            else:
                print(
                    "Warning: No predictions/targets collected during validation, skipping simple mAP calculation.")
                val_map_score = 0.0  # Assign a default value if no calculation happened

            # --- 简化评估结束 ---

            # --- 新增 DEBUG 打印 ---
            print(
                f"\nDEBUG: === AP Calculation Inputs (Epoch {epoch + 1}) ===")
            print(f"DEBUG: Num Predictions Collected: {len(all_predictions)}")
            print(f"DEBUG: Num Targets Collected: {len(all_targets)}")
            if all_predictions:
                print(
                    f"DEBUG: Example Prediction keys: {list(all_predictions[0].keys())}")
                print(f"DEBUG: Example Prediction shapes/types: "
                      f"cls={all_predictions[0]['box_cls'].shape}/{all_predictions[0]['box_cls'].dtype}, "
                      f"scores={all_predictions[0]['box_scores'].shape}/{all_predictions[0]['box_scores'].dtype}, "
                      f"xyz={all_predictions[0]['box_xyz'].shape}/{all_predictions[0]['box_xyz'].dtype}")
            else:
                print("DEBUG: all_predictions list is empty.")
            if all_targets:
                print(
                    f"DEBUG: Example Target keys: {list(all_targets[0].keys())}")
                print(f"DEBUG: Example Target shapes/types: "
                      f"cls={all_targets[0]['box_cls'].shape}/{all_targets[0]['box_cls'].dtype}, "
                      # f"scores={all_targets[0]['box_scores'].shape}/{all_targets[0]['box_scores'].dtype}, " # GT usually doesn't have scores
                      f"xyz={all_targets[0]['box_xyz'].shape}/{all_targets[0]['box_xyz'].dtype}")
            else:
                print("DEBUG: all_targets list is empty.")
            print("DEBUG: =========================================\n")
            # --- DEBUG 打印结束 ---

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
                               cls_weight=1.0,
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
        epoch_iou_loss = 0
        # Add accumulators for new regression loss components

        # 多尺度训练相关指标
        if enable_multiscale:
            epoch_scale_losses = [0, 0, 0]  # P3, P4, P5尺度的损失

        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5')

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list, sample_tokens) in pbar:
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

                # Ensure targets is a dictionary and move to device
                targets_dict = {}
                # targets_list is the dictionary yielded by the DataLoader
                for key, val in targets_list.items():
                    if isinstance(val, torch.Tensor):
                        targets_dict[key] = val.to(device)
                    else:
                        targets_dict[key] = val

                # Calculate loss using the correctly processed dictionary
                losses = loss_fn(preds, targets_dict)

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

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Step optimizer and scheduler
            if scaler:
                scaler.step(opt)   # Calls optimizer.step() implicitly
                scaler.update()
            else:  # Handle case where AMP is disabled
                opt.step()

            # Scheduler step should happen after optimizer step
            # This should be outside the scaler check, called every step where optimizer is stepped
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
                val_scale_losses = [0, 0, 0]

            # === 新增：用于收集解码结果 ===
            all_predictions = []
            all_targets = []
            # ---

            visualized_this_epoch = False  # Flag to visualize only once per epoch

            with torch.no_grad():
                # 只遍历一次验证集
                for batch_idx, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list, sample_tokens) in enumerate(valloader):
                    # 使用autocast进行混合精度训练 (如果验证时也启用AMP)
                    # 注意：为了评估指标的稳定性，通常建议验证时不使用autocast，除非内存非常受限
                    context_manager = autocast(
                        enabled=amp) if amp and torch.cuda.is_available() else nullcontext()
                    with context_manager:
                        preds = model(imgs.to(device),
                                      rots.to(device),
                                      trans.to(device),
                                      intrins.to(device),
                                      post_rots.to(device),
                                      post_trans.to(device),
                                      )

                        # Ensure targets is a dictionary and move to device
                        targets_dict = {}
                        for key, val in targets_list.items():
                            if isinstance(val, torch.Tensor):
                                targets_dict[key] = val.to(device)
                            else:
                                # Keep non-tensors as is
                                targets_dict[key] = val

                        # 1. 计算和累加损失
                        losses = loss_fn(preds, targets_dict)

                        val_loss += losses['total_loss'].item()
                        val_cls_loss += losses['cls_loss'].item()
                        val_reg_loss += losses['reg_loss'].item()
                        val_iou_loss += losses['iou_loss'].item()

                        if enable_multiscale and 'scale_losses' in losses:
                            for i, loss in enumerate(losses['scale_losses']):
                                val_scale_losses[i] += loss.item()

                    # --- 在autocast之外解码，确保使用float32进行解码 ---
                    # 确保preds是预期的字典格式 (如果模型输出不是字典需要转换)
                    if not isinstance(preds, dict) and isinstance(preds, torch.Tensor):
                        B, C, H, W = preds.shape
                        cls_channels = num_classes
                        reg_channels = 9  # x,y,z,w,l,h,sin,cos,vel - 检查是否与模型输出匹配
                        iou_channels = 1  # 检查是否与模型输出匹配

                        # 检查通道数是否足够
                        expected_channels = cls_channels + reg_channels + iou_channels
                        if C < expected_channels:
                            print(
                                f"Warning: Unexpected number of channels in prediction tensor ({C}). Expected at least {expected_channels}. Skipping decoding for this batch.")
                            continue  # 跳过这个批次的解码

                        preds_dict = {
                            'cls_pred': preds[:, :cls_channels],
                            # 注意：这里假设回归头预测所有类的参数，需要根据实际模型调整
                            # 如果模型为每个类输出独立的回归头，结构会不同
                            # 例如，如果形状是 B, (cls+reg+iou)*num_tasks, H, W
                            # 假设共享回归头
                            'reg_pred': preds[:, cls_channels:cls_channels+reg_channels],
                            'iou_pred': preds[:, -iou_channels:]  # 假设共享IoU头
                        }
                        #  如果你的模型为每个类别输出了独立的回归头，需要调整这里的解析逻辑
                        # 例如: preds_dict['reg_pred'] = preds[:, cls_channels:-iou_channels].view(B, num_classes, reg_channels, H, W)

                        preds = preds_dict  # 更新为字典格式

                    # 2. 解码预测和目标 (在循环内部进行)
                    # 确保 evaluate_3d 或当前文件中有 decode_predictions
                    from .evaluate_3d import decode_predictions  # 仅导入 decode_predictions
                    # score_thresh 应该是一个超参数，这里暂时用0.2
                    # --- 修改：降低 score_thresh 用于调试 ---
                    debug_score_thresh = 0.01
                    print(
                        f"DEBUG: Using lowered score_thresh={debug_score_thresh} for decode_predictions")
                    batch_dets = decode_predictions(
                        preds, device, score_thresh=debug_score_thresh, grid_conf=grid_conf)
                    # --- 修改结束 ---

                    # --- Debug: Print keys of targets_dict (保留用于调试) ---
                    # print(f"DEBUG: Keys in targets_dict: {list(targets_dict.keys())}")
                    # --- End Debug ---

                    # --- 手动处理 GT ---
                    # 从 targets_dict 中提取 GT box 信息，格式与 batch_dets 保持一致
                    batch_gts = []
                    # 假设 targets_dict 包含 'cls_targets' [B, N_max] 和 'reg_targets' [B, N_max, 9]
                    gt_cls = targets_dict['cls_targets']  # [B, N_max]
                    gt_reg = targets_dict['reg_targets']  # [B, N_max, 9]
                    B = gt_cls.shape[0]

                    for b in range(B):
                        sample_cls = gt_cls[b]  # [N_max]
                        sample_reg = gt_reg[b]  # [N_max, 9]

                        # 过滤掉 padding (假设类别ID <= 0 为 padding)
                        valid_mask = sample_cls > 0
                        if valid_mask.sum() == 0:
                            # 如果没有 GT box，添加空字典
                            gts = {
                                'box_cls': torch.zeros(0, dtype=torch.long, device=device),
                                'box_xyz': torch.zeros(0, 3, device=device),
                                'box_wlh': torch.zeros(0, 3, device=device),
                                'box_rot_sincos': torch.zeros(0, 2, device=device),
                                'box_vel': torch.zeros(0, 2, device=device)
                            }
                        else:
                            valid_cls = sample_cls[valid_mask]
                            valid_reg = sample_reg[valid_mask]  # [N_valid, 9]

                            # 从 valid_reg 中提取坐标、尺寸、旋转、速度
                            # 索引需要与 decode_predictions 中的回归头输出顺序一致!
                            # 假设顺序: x,y,z, w,l,h, sin,cos, vel_xy
                            gt_xyz = valid_reg[:, :3]
                            gt_wlh = valid_reg[:, 3:6]
                            gt_rot_sincos = valid_reg[:, 6:8]
                            gt_vel = valid_reg[:, 8:]  # 可能只有1维或2维
                            # 确保速度为2维
                            if gt_vel.shape[1] == 1:
                                gt_vel = torch.cat(
                                    [gt_vel, torch.zeros_like(gt_vel)], dim=1)
                            elif gt_vel.shape[1] > 2:
                                gt_vel = gt_vel[:, :2]
                            elif gt_vel.shape[1] == 0:  # 处理没有速度的情况
                                gt_vel = torch.zeros(
                                    gt_vel.shape[0], 2, device=device)

                            gts = {
                                'box_cls': valid_cls,
                                # 'box_scores': torch.ones_like(valid_cls, dtype=torch.float), # GT 没有分数, 但AP计算可能需要
                                'box_xyz': gt_xyz,
                                'box_wlh': gt_wlh,
                                'box_rot_sincos': gt_rot_sincos,
                                'box_vel': gt_vel
                            }
                        batch_gts.append(gts)
                    # --- GT 处理结束 ---

                    all_predictions.extend(batch_dets)
                    all_targets.extend(batch_gts)
                    # ---

                    # --- 可视化 (只在第一个验证批次进行一次) ---
                    if not visualized_this_epoch and batch_idx == 0:  # Visualize first batch
                        try:
                            vis_idx = 0  # Visualize the first sample in the batch
                            # 1. Visualize Input Image (Front Camera)
                            front_cam_idx = 1
                            if imgs.shape[1] > front_cam_idx:
                                # Ensure float
                                input_img_sample = imgs[vis_idx,
                                                        front_cam_idx].cpu().float()
                                writer.add_image(
                                    'val/input_front_camera', input_img_sample, global_step=epoch + 1)

                            # 2. Visualize BEV Prediction Heatmap
                            if 'cls_pred' in preds and preds['cls_pred'] is not None and preds['cls_pred'].numel() > 0:
                                cls_pred_sample = preds['cls_pred'][vis_idx].detach(
                                ).cpu().float()  # Ensure float
                                if cls_pred_sample.numel() > 0 and cls_pred_sample.dim() > 0:  # Check if not empty and has dimensions
                                    bev_heatmap = torch.max(torch.softmax(
                                        cls_pred_sample, dim=0), dim=0)[0]
                                    if bev_heatmap.numel() > 0:  # Check heatmap is not empty
                                        bev_heatmap = (
                                            bev_heatmap - bev_heatmap.min()) / (bev_heatmap.max() - bev_heatmap.min() + 1e-6)
                                        writer.add_image(
                                            'val/bev_prediction_heatmap', bev_heatmap.unsqueeze(0), global_step=epoch + 1)
                                else:
                                    print(
                                        f"Warning: cls_pred_sample for visualization is empty or invalid (shape: {cls_pred_sample.shape}).")
                            else:
                                print(
                                    "Warning: 'cls_pred' not found or empty in preds for visualization.")

                            # 3. Visualize Depth Map (Optional)
                            # ...

                            visualized_this_epoch = True
                        except Exception as e:
                            print(f"Warning: Failed to add visualization: {e}")
                    # --- 可视化结束 ---

            # --- 循环结束后 ---

            # 计算平均损失
            num_val_batches = len(valloader)
            if num_val_batches > 0:
                val_loss /= num_val_batches
                val_cls_loss /= num_val_batches
                val_reg_loss /= num_val_batches
                val_iou_loss /= num_val_batches
            else:
                print("Warning: Validation loader is empty.")
                # Set losses to NaN or zero if loader is empty
                val_loss, val_cls_loss, val_reg_loss, val_iou_loss = float(
                    'nan'), float('nan'), float('nan'), float('nan')

            # 记录验证损失
            writer.add_scalar('val/loss', val_loss, epoch + 1)
            writer.add_scalar('val/cls_loss', val_cls_loss, epoch + 1)
            writer.add_scalar('val/reg_loss', val_reg_loss, epoch + 1)
            writer.add_scalar('val/iou_loss', val_iou_loss, epoch + 1)

            # 记录多尺度验证损失
            if enable_multiscale and num_val_batches > 0:
                for i, loss in enumerate(val_scale_losses):
                    avg_loss = loss / num_val_batches
                    writer.add_scalar(
                        f'val/scale{i+3}_loss', avg_loss, epoch + 1)

            # --- 计算简化的 mAP ---
            # 定义 calculate_simple_ap (如果尚未定义或导入)
            # 注意：这个函数需要在此作用域内可用

            def calculate_simple_ap(preds_list, targets_list, num_classes, iou_threshold=0.5, dist_threshold=2.0):
                """计算简化的平均精度 (基于距离匹配)"""
                if not preds_list or not targets_list:
                    print(
                        "Warning: Empty predictions or targets list for simple AP calculation.")
                    return 0.0, {}  # Return 0 mAP and empty class APs

                assert len(preds_list) == len(
                    targets_list), "Prediction and target list lengths must match"

                class_aps = {}
                # 从1到num_classes (不包括背景类0)
                for cls_id in range(1, num_classes + 1):
                    tp = 0
                    fp = 0
                    total_gt_for_class = 0
                    scores_for_class = []
                    match_for_class = []  # 0 for fp, 1 for tp

                    # 遍历每个样本
                    for sample_preds, sample_targets in zip(preds_list, targets_list):
                        # 当前样本中该类的GT数量
                        gt_mask_sample = sample_targets['box_cls'] == cls_id
                        total_gt_for_class += gt_mask_sample.sum().item()

                        # 当前样本中该类的预测
                        pred_mask_sample = sample_preds['box_cls'] == cls_id
                        num_preds_sample = pred_mask_sample.sum().item()

                        if num_preds_sample == 0:
                            continue  # 没有这个类的预测，跳到下一个样本

                        # 获取该类的预测框、得分和GT框
                        pred_boxes_sample = sample_preds['box_xyz'][pred_mask_sample]
                        pred_scores_sample = sample_preds['box_scores'][pred_mask_sample]
                        gt_boxes_sample = sample_targets['box_xyz'][gt_mask_sample]

                        scores_for_class.append(pred_scores_sample)

                        if gt_boxes_sample.shape[0] == 0:
                            # 这个样本没有该类的GT，所有预测都是FP
                            fp += num_preds_sample
                            match_for_class.extend([0] * num_preds_sample)
                            continue

                        # 使用距离进行匹配
                        # matched_gt = torch.zeros(gt_boxes_sample.shape[0], dtype=torch.bool, device=device)
                        # 简化：我们只需要统计TP/FP，不需要精确的AP曲线，所以可以用更简单的方式
                        # 对每个预测框，找到最近的GT框，如果距离小于阈值，则为TP (假设一对一匹配简化)
                        # TODO: 这种简化可能导致一个GT匹配多个预测，更精确的方法需要匈牙利算法或类似方法
                        sample_tp = 0
                        sample_fp = 0
                        # 对预测按分数排序（简化AP不需要，但保留逻辑）
                        sorted_indices = torch.argsort(
                            pred_scores_sample, descending=True)
                        pred_boxes_sorted = pred_boxes_sample[sorted_indices]

                        # 记录哪些GT已经被匹配过 (简化版)
                        gt_matched_flags = torch.zeros(
                            gt_boxes_sample.shape[0], dtype=torch.bool, device=device)

                        for p_box in pred_boxes_sorted:
                            min_dist = float('inf')
                            best_gt_idx = -1
                            match_found_for_pred = False

                            # Ensure GT boxes exist
                            if gt_boxes_sample.shape[0] > 0:
                                distances = torch.norm(
                                    gt_boxes_sample - p_box.unsqueeze(0), p=2, dim=1)
                                min_dist, best_gt_idx = torch.min(
                                    distances, dim=0)

                                # 检查距离和是否已被匹配
                                if min_dist < dist_threshold and not gt_matched_flags[best_gt_idx]:
                                    sample_tp += 1
                                    # 标记为已匹配
                                    gt_matched_flags[best_gt_idx] = True
                                    match_for_class.append(1)  # 标记为TP
                                    match_found_for_pred = True
                                else:
                                    sample_fp += 1
                                    match_for_class.append(0)  # 标记为FP
                            else:  # No GT boxes of this class in the sample
                                sample_fp += 1
                                match_for_class.append(0)  # 标记为FP

                        tp += sample_tp
                        fp += sample_fp

                    if total_gt_for_class == 0:
                        class_aps[cls_id] = 0.0  # 没有GT，AP为0
                        continue

                    if tp + fp == 0:
                        # 没有预测或者所有预测都匹配了GT (不太可能在真实场景发生)
                        # 如果 tp=0, fp=0, 没有预测，precision=0
                        # 如果 tp>0, fp=0, 所有预测都是TP，precision=1
                        precision = 1.0 if tp > 0 else 0.0
                    else:
                        precision = tp / (tp + fp)

                    recall = tp / total_gt_for_class if total_gt_for_class > 0 else 0.0

                    # 使用 F1 分数作为简化的 AP 替代 (或者直接用 Precision * Recall)
                    # ap = precision * recall
                    ap = 2 * (precision * recall) / (precision +
                                                     recall) if (precision + recall) > 0 else 0.0
                    class_aps[cls_id] = ap if not np.isnan(
                        ap) else 0.0  # Handle potential NaN

                # 计算 mAP
                valid_aps = [v for v in class_aps.values() if not np.isnan(v)]
                map_score = sum(valid_aps) / \
                    len(valid_aps) if valid_aps else 0.0

                print(
                    f"Simplified AP calculation results (dist_thresh={dist_threshold}):")
                # print(f" Class APs: {class_aps}") # Optional: print per-class APs
                print(f" mAP: {map_score:.4f}")

                return map_score, class_aps  # 返回 mAP 和各类别 AP

            # 调用简化的AP计算
            # 确保 all_predictions 和 all_targets 不为空
            if all_predictions and all_targets:
                # 使用距离阈值0.5和2.0计算
                map_dist_2, class_aps_dist_2 = calculate_simple_ap(
                    all_predictions, all_targets, num_classes, dist_threshold=2.0)
                map_dist_1, class_aps_dist_1 = calculate_simple_ap(
                    all_predictions, all_targets, num_classes, dist_threshold=1.0)  # 更严格的阈值
                val_map_score = map_dist_2  # 使用2米距离阈值作为主要指标

                # 记录指标
                writer.add_scalar('val/simple_mAP_dist_2m',
                                  val_map_score, epoch + 1)
                writer.add_scalar('val/simple_mAP_dist_1m',
                                  map_dist_1, epoch + 1)  # 记录1米阈值的结果

                print('||Val Loss: %.4f | Simple mAP@dist=2m: %.4f | Simple mAP@dist=1m: %.4f||' %
                      (val_loss, val_map_score, map_dist_1))

                # 保存最佳模型 (基于 dist=2m 的 mAP)
                if val_map_score > best_map:
                    best_map = val_map_score
                    best_checkpoint = {
                        'net': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'best_map': best_map,  # Store the best mAP score
                        'model_configs': model_configs,
                        'amp_scaler': scaler.state_dict() if scaler is not None else None,
                    }
                    best_model_path = os.path.join(
                        path, "best_map.pt")  # Use a distinct name
                    torch.save(best_checkpoint, best_model_path)
                    print(
                        f"Best model saved to {best_model_path} with simple mAP@dist=2m: {best_map:.4f}")
            else:
                print(
                    "Warning: No predictions/targets collected during validation, skipping simple mAP calculation.")
                val_map_score = 0.0  # Assign a default value if no calculation happened

            # --- 简化评估结束 ---

        epoch += 1
        pbar.close()

    writer.close()
