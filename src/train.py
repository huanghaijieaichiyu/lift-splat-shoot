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
from .tools import get_batch_iou, get_val_info, get_val_info_fusion
from contextlib import nullcontext  # 导入nullcontext
from .nuscenes_info import load_nuscenes_infos  # 导入数据集缓存加载函数
import torchvision  # 导入 torchvision 用于可视化
import torch.autograd  # 导入 autograd


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
                 ncams=6,
                 max_grad_norm=5.0,
                 pos_weight=2.13,
                 logdir='./runs_fusion',

                 xbound=[-50.0, 50.0, 0.5],
                 ybound=[-50.0, 50.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[4.0, 45.0, 1.0],

                 lidar_inC=1,
                 lidar_xbound=[-50.0, 50.0, 0.5],
                 lidar_ybound=[-50.0, 50.0, 0.5],

                 bsz=4,
                 nworkers=4,
                 lr=1e-3,
                 weight_decay=1e-7,
                 log_vis_interval=500,  # 每 N 个 batch 记录一次可视化信息
                 ):
    # --- 设置 Autograd Anomaly Detection ---
    torch.autograd.set_detect_anomaly(True)
    # --------------------------------------

    # 训练前验证并确保数据集缓存存在
    check_and_ensure_cache(dataroot, version)

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
        'lidar_xbound': lidar_xbound,
        'lidar_ybound': lidar_ybound,
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

    # --- Data Loading for Fusion ---
    print("Attempting to load data using 'fusiondata' parser...")
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='fusiondata', lidar_inC=lidar_inC)
    print("Data loaded.")
    # --- End Data Loading ---

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # --- Model Compilation for Fusion ---
    print("Compiling FusionNet model...")
    model = compile_model(grid_conf, data_aug_conf,
                          outC=1, lidar_inC=lidar_inC)
    D_depth = model.D  # 从模型获取深度维度 D
    print("Model compiled.")
    # --- End Model Compilation ---

    loss_seg = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight)).to(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)

    if resume != '':
        path = os.path.dirname(resume)
        if not os.path.exists(path):
            print(
                f"Resume directory {path} does not exist. Creating log directory instead.")
            path = save_path(logdir)  # Fallback to default logdir
    else:
        path = save_path(logdir)  # Use the potentially different logdir
    writer = SummaryWriter(logdir=path)
    print(f"Logging to: {path}")

    val_step = 10 if version == 'mini' else 30
    scaler = GradScaler(enabled=amp)
    model.train()
    global_step = 0  # 使用 global_step 记录 TensorBoard
    epoch = 0
    max_iou = 0.0

    if resume != '':
        if os.path.exists(resume):
            print("Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch'] + 1
            if 'max_iou' in checkpoint:
                max_iou = checkpoint['max_iou']
            if amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                try:
                    scaler.load_state_dict(checkpoint['scaler'])
                except Exception as e:
                    print(f"Warning: Could not load scaler state: {e}")
            if 'global_step' in checkpoint:  # 恢复 global_step
                global_step = checkpoint['global_step']
            print(
                f"Resuming training from epoch {epoch}, step {global_step}, best IoU so far: {max_iou:.4f}")
        else:
            print(
                f"Resume checkpoint {resume} not found. Starting from scratch.")

    if load_weight != '':
        if os.path.exists(load_weight):
            print("Loading weight '{}'".format(load_weight))
            checkpoint = torch.load(load_weight, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict_source = checkpoint.get('net', checkpoint)
            pretrained_dict = {k: v for k, v in pretrained_dict_source.items(
            ) if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(
                f"Loaded weights from {load_weight}. {len(pretrained_dict)} matching keys found.")
        else:
            print(
                f"Weight file {load_weight} not found. Skipping weight loading.")

    print(f"Starting training from epoch {epoch}...")
    while epoch < nepochs:
        np.random.seed()
        t0_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5', desc=f"Epoch {epoch+1}/{nepochs}")

        for batchi, batch_data in pbar:
            if len(batch_data) != 8:
                print(
                    f"Error: Expected 8 items from dataloader, got {len(batch_data)}. Check 'fusiondata' parser. Skipping batch.")
                continue
            try:
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev = batch_data
            except ValueError as e:
                print(
                    f"Error unpacking batch data: {e}. Expected 8 items. Skipping batch.")
                continue

            opt.zero_grad()
            context = autocast(enabled=amp)
            with context:
                preds, depth_prob = model(imgs.to(device),
                                          rots.to(device),
                                          trans.to(device),
                                          intrins.to(device),
                                          post_rots.to(device),
                                          post_trans.to(device),
                                          lidar_bev.to(device)
                                          )
                binimgs = binimgs.to(device)
                loss = loss_seg(preds, binimgs)

            if not torch.isfinite(loss):
                print(
                    f"警告: Epoch {epoch+1}, Batch {batchi}, 检测到无效损失值 (nan/inf)。跳过梯度更新。 Loss: {loss.item()}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % log_vis_interval == 0:
                with torch.no_grad():
                    vis_idx = 0
                    input_img_vis = imgs[vis_idx, 1].cpu()
                    writer.add_image('train/input_image_front',
                                     input_img_vis, global_step)

                    lidar_bev_vis = lidar_bev[vis_idx].cpu().sum(
                        0, keepdim=True)
                    lidar_bev_vis = (lidar_bev_vis - lidar_bev_vis.min()) / \
                        (lidar_bev_vis.max() - lidar_bev_vis.min() + 1e-6)
                    writer.add_image('train/input_lidar_bev',
                                     lidar_bev_vis, global_step)

                    gt_bev_vis = binimgs[vis_idx].cpu().float()
                    writer.add_image('train/gt_bev', gt_bev_vis, global_step)

                    pred_bev_vis = torch.sigmoid(preds[vis_idx]).cpu()
                    writer.add_image('train/pred_bev',
                                     pred_bev_vis, global_step)

                    front_cam_depth_prob = depth_prob[vis_idx, 1].cpu()
                    depth_indices = torch.argmax(
                        front_cam_depth_prob, dim=0, keepdim=True)
                    vis_depth_map = depth_indices.float() / (D_depth - 1)
                    vis_depth_map_resized = F.interpolate(vis_depth_map.unsqueeze(
                        0), size=final_dim, mode='nearest').squeeze(0)
                    writer.add_image('train/depth_map_front_maxprob',
                                     vis_depth_map_resized, global_step)

        avg_epoch_loss = epoch_loss / \
            len(trainloader) if len(trainloader) > 0 else 0.0
        t1_epoch = time.time()
        print(
            f"Epoch {epoch + 1} Average Train Loss: {avg_epoch_loss:.4f}, Time: {t1_epoch - t0_epoch:.2f}s")
        writer.add_scalar('train/loss_epoch', avg_epoch_loss, epoch + 1)

        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scaler': scaler.state_dict() if amp else None,
            'epoch': epoch,
            'max_iou': max_iou,
            'global_step': global_step
        }
        last_model_path = os.path.join(path, "last_fusion.pt")
        torch.save(last_checkpoint, last_model_path)

        if (epoch + 1) % val_step == 0:
            print(f"Running validation for epoch {epoch+1}...")
            model.eval()
            val_loss_total = 0.0
            val_intersect_total = 0.0
            val_union_total = 0.0
            num_val_batches = 0
            visualized_val = False

            with torch.no_grad():
                for batchi_val, batch_data_val in tqdm(enumerate(valloader), total=len(valloader), desc="Validation"):
                    if len(batch_data_val) != 8:
                        continue
                    try:
                        imgs_val, rots_val, trans_val, intrins_val, post_rots_val, post_trans_val, binimgs_val, lidar_bev_val = batch_data_val
                    except ValueError:
                        continue

                    context_val = autocast(enabled=amp)
                    with context_val:
                        preds_val, depth_prob_val = model(imgs_val.to(device),
                                                          rots_val.to(device),
                                                          trans_val.to(device),
                                                          intrins_val.to(
                                                              device),
                                                          post_rots_val.to(
                                                              device),
                                                          post_trans_val.to(
                                                              device),
                                                          lidar_bev_val.to(
                                                              device)
                                                          )
                        binimgs_val = binimgs_val.to(device)
                        loss_val = loss_seg(preds_val, binimgs_val)

                    if torch.isfinite(loss_val):
                        val_loss_total += loss_val.item()
                        intersect, union, _ = get_batch_iou(
                            preds_val, binimgs_val)
                        val_intersect_total += intersect.item()
                        val_union_total += union.item()
                        num_val_batches += 1

                        if not visualized_val:
                            vis_idx_val = 0
                            input_img_vis_val = imgs_val[vis_idx_val, 1].cpu()
                            writer.add_image(
                                'val/input_image_front', input_img_vis_val, global_step)

                            lidar_bev_vis_val = lidar_bev_val[vis_idx_val].cpu().sum(
                                0, keepdim=True)
                            lidar_bev_vis_val = (lidar_bev_vis_val - lidar_bev_vis_val.min()) / (
                                lidar_bev_vis_val.max() - lidar_bev_vis_val.min() + 1e-6)
                            writer.add_image(
                                'val/input_lidar_bev', lidar_bev_vis_val, global_step)

                            gt_bev_vis_val = binimgs_val[vis_idx_val].cpu(
                            ).float()
                            writer.add_image(
                                'val/gt_bev', gt_bev_vis_val, global_step)

                            pred_bev_vis_val = torch.sigmoid(
                                preds_val[vis_idx_val]).cpu()
                            writer.add_image(
                                'val/pred_bev', pred_bev_vis_val, global_step)

                            front_cam_depth_prob_val = depth_prob_val[vis_idx_val, 1].cpu(
                            )
                            depth_indices_val = torch.argmax(
                                front_cam_depth_prob_val, dim=0, keepdim=True)
                            vis_depth_map_val = depth_indices_val.float() / (D_depth - 1)
                            vis_depth_map_resized_val = F.interpolate(
                                vis_depth_map_val.unsqueeze(0), size=final_dim, mode='nearest').squeeze(0)
                            writer.add_image(
                                'val/depth_map_front_maxprob', vis_depth_map_resized_val, global_step)
                            visualized_val = True

            avg_val_loss = val_loss_total / num_val_batches if num_val_batches > 0 else 0.0
            current_iou = val_intersect_total / \
                val_union_total if val_union_total > 0 else 0.0

            print(
                f"Validation Results Epoch {epoch + 1}: Avg Loss: {avg_val_loss:.4f}, IoU: {current_iou:.4f}")
            writer.add_scalar('val/loss_epoch', avg_val_loss, epoch + 1)
            writer.add_scalar('val/iou', current_iou, epoch + 1)

            if current_iou > max_iou:
                max_iou = current_iou
                print(
                    f"*** New best IoU: {max_iou:.4f}. Saving best model... ***")
                best_checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict() if amp else None,
                    'epoch': epoch,
                    'max_iou': max_iou,
                    'global_step': global_step
                }
                best_model_path = os.path.join(path, "best_fusion.pt")
                torch.save(best_checkpoint, best_model_path)
                print(f"Saved best model checkpoint to {best_model_path}")

            model.train()

        epoch += 1

    print("Training finished.")
    writer.close()
