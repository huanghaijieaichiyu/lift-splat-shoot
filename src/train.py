import torch
import torch.nn as nn
import torch.nn.functional as F
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
from nuscenes import NuScenes    # Import NuScenes


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
        if not os.path.exists(path):
            print(
                f"Resume directory {path} does not exist. Creating log directory instead.")
            path = save_path(logdir)
    else:
        path = save_path(logdir)

    writer = SummaryWriter(logdir=path)
    val_step = 10 if version == 'mini' else 30
    model.train()
    counter = 0
    epoch = 0
    max_f1 = 0.0  # Track best F1 score

    if resume != '':
        if os.path.exists(resume):
            print("Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume, map_location=device)
            model.load_state_dict(checkpoint['net'])
            opt.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint.get('epoch', 0) + 1
            max_f1 = checkpoint.get('max_f1', 0.0)
            print(
                f"Resuming training from epoch {epoch}, best F1 so far: {max_f1:.4f}")
        else:
            print(
                f"Resume checkpoint {resume} not found. Starting from scratch.")

    if load_weight != '':
        if os.path.exists(load_weight):
            print("Loading weight '{}'".format(load_weight))
            checkpoint = torch.load(load_weight, map_location=device)
            # Handle checkpoints that might be just the state_dict or a dict containing 'net'
            state_dict_to_load = checkpoint.get('net', checkpoint)
            model.load_state_dict(state_dict_to_load, strict=False)
            # Optionally load optimizer state if available and needed
            # if 'optimizer' in checkpoint:
            #     opt.load_state_dict(checkpoint['optimizer'])
        else:
            print(
                f"Weight file {load_weight} not found. Skipping weight loading.")

    while epoch < nepochs:
        np.random.seed()
        # Iou = [0.0]  # Keep track of IoU history if needed
        t0_epoch = time.time()
        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5', desc=f"Epoch {epoch+1}/{nepochs}")

        epoch_loss_total = 0.0
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in pbar:
            t0_batch = time.time()
            opt.zero_grad()
            context = autocast(enabled=amp)
            with context:
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )
            binimgs = binimgs.to(device)

            with context:
                loss = loss_seg(preds, binimgs)  # 使用分割损失变量名

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                opt.step()
                epoch_loss_total += loss.item()
            else:
                print(
                    f"Warning: Epoch {epoch+1}, Batch {batchi}, NaN/Inf loss detected. Skipping update. Loss: {loss.item()}")

            counter += 1
            t1_batch = time.time()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_epoch_loss = epoch_loss_total / \
            len(trainloader) if len(trainloader) > 0 else 0.0
        t1_epoch = time.time()
        print(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}, Time: {t1_epoch-t0_epoch:.2f}s")
        writer.add_scalar('train/loss_epoch', avg_epoch_loss, epoch + 1)

        # Save the last model for resume
        last_checkpoint = {
            'net': model.state_dict(),
            'optimizer': opt.state_dict(),
            'epoch': epoch,
            'max_f1': max_f1  # Save current best f1
        }
        last_model_path = os.path.join(path, "last.pt")
        torch.save(last_checkpoint, last_model_path)

        if (epoch + 1) % val_step == 0:
            # --- Validation Step for Simple Model ---
            val_info = get_val_info(model, valloader, loss_seg, device)
            current_iou = val_info['iou']
            current_f1 = val_info['f1']
            current_precision = val_info['precision']
            current_recall = val_info['recall']
            current_loss = val_info['loss']

            print(
                f"|| Val Metrics Epoch {epoch+1}: Loss: {current_loss:.4f} | IoU: {current_iou:.4f} | Precision: {current_precision:.4f} | Recall: {current_recall:.4f} | F1: {current_f1:.4f} ||")

            writer.add_scalar('val/loss', current_loss, epoch + 1)
            writer.add_scalar('val/iou', current_iou, epoch + 1)
            writer.add_scalar('val/precision', current_precision, epoch + 1)
            writer.add_scalar('val/recall', current_recall, epoch + 1)
            writer.add_scalar('val/f1', current_f1, epoch + 1)

            # Save the best model based on F1-score
            if current_f1 > max_f1:
                max_f1 = current_f1
                print(
                    f"*** New best F1: {max_f1:.4f}. Saving best model... ***")
                best_model_path = os.path.join(path, "best_f1.pt")
                best_checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': epoch,
                    'max_f1': max_f1,
                    'iou_at_best_f1': current_iou  # Store IoU corresponding to best F1
                }
                torch.save(best_checkpoint, best_model_path)
                print(f"Saved best model checkpoint to {best_model_path}")

            # Optional: Debug print from original code
            # print(
            #     '||val/loss: {} ||-----|-----||val/iou: {}||'.format(current_loss, current_iou))
            # print('-----------------|done|--------------------')
            # --- End Validation Step ---

        epoch += 1
        # pbar.close() # Closing inside loop might cause issues if tqdm is outside
    pbar.close()  # Close pbar after the loop finishes
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
                 # Add arg for map layers to evaluate
                 map_layers=['drivable_area'],
                 val_step=10,  # Ensure val_step is defined or passed
                 ):
    # --- Setup Autograd Anomaly Detection & Cache Check ---
    # torch.autograd.set_detect_anomaly(True) # Enable only if debugging NaNs
    check_and_ensure_cache(dataroot, version)

    # --- Configs ---
    grid_conf = {  # Ensure grid_conf includes lidar bounds if used by model compilation
        'xbound': xbound, 'ybound': ybound, 'zbound': zbound, 'dbound': dbound,
        'lidar_xbound': lidar_xbound, 'lidar_ybound': lidar_ybound
    }
    data_aug_conf = {  # Ensure data_aug_conf includes final_dim
        'resize_lim': resize_lim, 'final_dim': final_dim, 'rot_lim': rot_lim,
        'H': H, 'W': W, 'rand_flip': rand_flip, 'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    # Necessary for visualization if used
    final_dim_vis = data_aug_conf['final_dim']

    # --- Data Loading ---
    print("Loading data with 'fusiondata' parser (ensure it yields sample_token as 9th item)...")
    # Pass lidar_inC if required by your compile_data/model
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='fusiondata', lidar_inC=lidar_inC)
    print("Data loaded.")

    # --- Device and Model ---
    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    print("Compiling FusionNet model...")
    # Pass lidar_inC if required by your model compilation
    model = compile_model(grid_conf, data_aug_conf,
                          outC=1, lidar_inC=lidar_inC)
    # Get depth dimension if needed for visualization, handle case where model might not have D
    D_depth = getattr(model, 'D', None)
    print("Model compiled.")
    model.to(device)

    # --- Loss, Optimizer, Scaler ---
    loss_seg = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    scaler = GradScaler(enabled=amp)

    # --- Logging and Checkpointing Setup ---
    if resume != '':
        path = os.path.dirname(resume)
        if not os.path.exists(path):
            print(
                f"Resume directory {path} does not exist. Creating log directory instead.")
            path = save_path(logdir)
    else:
        path = save_path(logdir)
    writer = SummaryWriter(logdir=path)
    print(f"Logging to: {path}")

    # --- Initialize NuScenes API ---
    nusc = None  # Initialize nusc to None
    try:
        print("Initializing NuScenes API...")
        nusc_version_str = f'v1.0-{version}'
        print(
            f"Using NuScenes version: {nusc_version_str}, dataroot: {dataroot}")
        nusc = NuScenes(version=nusc_version_str,
                        dataroot=dataroot, verbose=False)
        print("NuScenes API initialized successfully.")
    except ImportError:
        print("WARNING: nuscenes-devkit not found. Devkit validation will be skipped.")
        print("Install with: pip install nuscenes-devkit")
    except Exception as e:
        print(f"WARNING: Failed to initialize NuScenes API: {e}")
        print("Devkit validation will be skipped. Check dataroot and version.")

    # --- Training State Init ---
    global_step = 0
    epoch = 0
    max_iou = 0.0  # Track best simple IoU
    max_f1 = 0.0   # Track best simple F1
    max_devkit_f1 = 0.0  # Track best devkit F1

    # --- Resume Logic ---
    if resume != '':
        if os.path.exists(resume):
            print(f"Loading checkpoint '{resume}'")
            checkpoint = torch.load(resume, map_location=device)

            # Load model state
            model_state_dict = checkpoint.get(
                'net', checkpoint.get('model', None))
            if model_state_dict:
                model.load_state_dict(model_state_dict)
            else:
                print(
                    "Warning: Could not find model state ('net' or 'model') in checkpoint.")

            # Load optimizer state
            if 'optimizer' in checkpoint:
                opt.load_state_dict(checkpoint['optimizer'])
            else:
                print("Warning: Could not find optimizer state in checkpoint.")

            # Load scaler state
            if amp and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
                try:
                    scaler.load_state_dict(checkpoint['scaler'])
                except Exception as e:
                    print(f"Warning: Could not load scaler state: {e}")
            elif amp and 'scaler' not in checkpoint:
                print(
                    "Warning: AMP is enabled, but no scaler state found in checkpoint.")

            # Load epoch, step, and best metrics
            epoch = checkpoint.get('epoch', 0) + 1
            max_iou = checkpoint.get('max_iou', 0.0)
            max_f1 = checkpoint.get('max_f1', 0.0)
            max_devkit_f1 = checkpoint.get(
                'max_devkit_f1', 0.0)  # Load devkit metric
            global_step = checkpoint.get('global_step', 0)

            print(
                f"Resuming training from epoch {epoch}, step {global_step}, best Simple IoU: {max_iou:.4f}, Simple F1: {max_f1:.4f}, Devkit F1: {max_devkit_f1:.4f}")
        else:
            print(
                f"Resume checkpoint {resume} not found. Starting from scratch.")

    # --- Load Weight Logic ---
    if load_weight != '':
        if os.path.exists(load_weight):
            print("Loading weight '{}'".format(load_weight))
            checkpoint = torch.load(load_weight, map_location=device)
            model_dict = model.state_dict()
            # Handle weights saved directly or within 'net' key
            pretrained_dict_source = checkpoint.get('net', checkpoint)
            # Filter out unnecessary keys and layers with size mismatches
            pretrained_dict = {k: v for k, v in pretrained_dict_source.items()
                               if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(
                f"Loaded {len(pretrained_dict)} matching weights from {load_weight}. Ignored {len(pretrained_dict_source) - len(pretrained_dict)} keys.")
        else:
            print(
                f"Weight file {load_weight} not found. Skipping weight loading.")

    # --- Training Loop ---
    print(f"Starting training from epoch {epoch}...")
    while epoch < nepochs:
        np.random.seed(epoch)  # Seed with epoch for reproducibility if needed
        t0_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader),
                    colour='#8762A5', desc=f"Epoch {epoch+1}/{nepochs}")

        for batchi, batch_data in pbar:
            # 处理batch_data，确保正确处理sample_token
            if len(batch_data) == 9:
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev, sample_tokens = batch_data
            elif len(batch_data) == 8:
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev = batch_data
                sample_tokens = None
            else:
                print(f"错误：期望从训练数据加载器获取8或9个项，但得到了{len(batch_data)}个。跳过该批次。")
                continue

            opt.zero_grad()
            context = autocast(enabled=amp)
            with context:
                # 确保模型调用匹配预期输入
                model_output = model(imgs.to(device),
                                     rots.to(device),
                                     trans.to(device),
                                     intrins.to(device),
                                     post_rots.to(device),
                                     post_trans.to(device),
                                     lidar_bev.to(device)
                                     )
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    preds, depth_prob = model_output
                elif isinstance(model_output, torch.Tensor):
                    preds = model_output
                    depth_prob = None
                else:
                    print(f"错误：意外的模型输出格式：{type(model_output)}。跳过该批次。")
                    continue

                binimgs = binimgs.to(device)
                loss = loss_seg(preds, binimgs)

            if not torch.isfinite(loss):
                print(
                    f"警告：第{epoch+1}轮，第{batchi}批次，检测到NaN/Inf损失。跳过更新。损失：{loss.item()}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # 训练可视化（可选）
            if log_vis_interval > 0 and global_step % log_vis_interval == 0:
                with torch.no_grad():
                    try:
                        vis_idx = 0
                        # 输入图像
                        input_img_vis = imgs[vis_idx, 1].cpu()
                        writer.add_image(
                            'train/input_image_front', input_img_vis, global_step)

                        # LiDAR BEV输入
                        lidar_bev_vis = lidar_bev[vis_idx].cpu().sum(
                            0, keepdim=True)
                        lidar_bev_vis = (lidar_bev_vis - lidar_bev_vis.min()) / \
                            (lidar_bev_vis.max() - lidar_bev_vis.min() + 1e-6)
                        writer.add_image('train/input_lidar_bev',
                                         lidar_bev_vis, global_step)

                        # 地面真值BEV
                        gt_bev_vis = binimgs[vis_idx].cpu().float()
                        writer.add_image(
                            'train/gt_bev', gt_bev_vis, global_step)

                        # 预测BEV
                        pred_bev_vis = torch.sigmoid(preds[vis_idx]).cpu()
                        writer.add_image('train/pred_bev',
                                         pred_bev_vis, global_step)

                        # 深度图可视化（如果可用）
                        if depth_prob is not None and D_depth is not None:
                            front_cam_depth_prob = depth_prob[vis_idx, 1].cpu()
                            depth_indices = torch.argmax(
                                front_cam_depth_prob, dim=0, keepdim=True)
                            vis_depth_map = depth_indices.float() / max(1, D_depth - 1)
                            vis_depth_map_resized = F.interpolate(vis_depth_map.unsqueeze(
                                0), size=final_dim_vis, mode='nearest').squeeze(0)
                            writer.add_image(
                                'train/depth_map_front_maxprob', vis_depth_map_resized, global_step)

                        # 记录sample_token（如果可用）
                        if sample_tokens is not None:
                            writer.add_text('train/sample_token',
                                            sample_tokens[vis_idx], global_step)
                    except Exception as vis_e:
                        print(f"警告：训练可视化期间出错：{vis_e}")
                        pass

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / \
            len(trainloader) if len(trainloader) > 0 else 0.0
        t1_epoch = time.time()
        print(
            f"Epoch {epoch + 1} Avg Train Loss: {avg_epoch_loss:.4f}, Time: {t1_epoch - t0_epoch:.2f}s")
        writer.add_scalar('train/loss_epoch', avg_epoch_loss, epoch + 1)

        # --- Save Last Checkpoint ---
        last_checkpoint = {
            'net': model.state_dict(), 'optimizer': opt.state_dict(),
            'scaler': scaler.state_dict() if amp else None,
            'epoch': epoch,
            # Save all best metrics
            'max_iou': max_iou, 'max_f1': max_f1, 'max_devkit_f1': max_devkit_f1,
            'global_step': global_step
        }
        last_model_path = os.path.join(path, "last_fusion.pt")
        try:
            torch.save(last_checkpoint, last_model_path)
        except Exception as e:
            print(f"Error saving last checkpoint: {e}")

        # --- Validation Step ---
        if val_step > 0 and (epoch + 1) % val_step == 0:
            print(f"--- 运行验证 第{epoch+1}轮 ---")
            val_info = get_val_info_fusion(model=model,
                                           valloader=valloader,
                                           loss_fn=loss_seg,
                                           device=device,
                                           nusc=nusc,
                                           grid_conf=grid_conf,
                                           writer=writer,
                                           global_step=global_step,
                                           map_layers=map_layers,
                                           final_dim_vis=final_dim_vis,
                                           D_depth=D_depth)  # 移除sample_tokens参数

            # Log metrics returned by get_val_info_fusion
            # ... (Log simple and devkit metrics logic remains same) ...
            writer.add_scalar('val/loss_epoch', val_info['loss'], epoch + 1)
            writer.add_scalar('val/simple_iou',
                              val_info['simple_iou'], epoch + 1)
            writer.add_scalar('val/simple_precision',
                              val_info['simple_precision'], epoch + 1)
            writer.add_scalar('val/simple_recall',
                              val_info['simple_recall'], epoch + 1)
            writer.add_scalar(
                'val/simple_f1', val_info['simple_f1'], epoch + 1)
            if 'devkit_f1' in val_info:
                writer.add_scalar('val/devkit_iou',
                                  val_info['devkit_iou'], epoch + 1)
                writer.add_scalar('val/devkit_precision',
                                  val_info['devkit_precision'], epoch + 1)
                writer.add_scalar('val/devkit_recall',
                                  val_info['devkit_recall'], epoch + 1)
                writer.add_scalar(
                    'val/devkit_f1', val_info['devkit_f1'], epoch + 1)

            # --- Save Best Model Checkpoints --- #
            # ... (Save best simple F1 and devkit F1 logic remains same) ...
            current_f1 = val_info['simple_f1']
            if current_f1 > max_f1:
                max_f1 = current_f1
                print(
                    f"*** New best Simple F1: {max_f1:.4f}. Saving best_simple_f1 model... ***")
                best_checkpoint_f1 = {
                    'net': model.state_dict(), 'optimizer': opt.state_dict(),
                    'scaler': scaler.state_dict() if amp else None, 'epoch': epoch,
                    'max_iou': val_info.get('simple_iou', 0.0),
                    'max_f1': max_f1,
                    'max_devkit_f1': val_info.get('devkit_f1', 0.0),
                    'global_step': global_step
                }
                best_model_path_f1 = os.path.join(
                    path, "best_fusion_simple_f1.pt")
                try:
                    torch.save(best_checkpoint_f1, best_model_path_f1)
                    print(
                        f"Saved best simple F1 model checkpoint to {best_model_path_f1}")
                except Exception as e:
                    print(f"Error saving best simple F1 checkpoint: {e}")
            if 'devkit_f1' in val_info:
                current_devkit_f1 = val_info['devkit_f1']
                if current_devkit_f1 > max_devkit_f1:
                    max_devkit_f1 = current_devkit_f1
                    print(
                        f"*** New best Devkit F1: {max_devkit_f1:.4f}. Saving best_devkit_f1 model... ***")
                    best_checkpoint_devkit_f1 = {
                        'net': model.state_dict(), 'optimizer': opt.state_dict(),
                        'scaler': scaler.state_dict() if amp else None, 'epoch': epoch,
                        'max_iou': val_info.get('simple_iou', 0.0),
                        'max_f1': val_info.get('simple_f1', 0.0),
                        'max_devkit_f1': max_devkit_f1,
                        'global_step': global_step
                    }
                    best_model_path_devkit_f1 = os.path.join(
                        path, "best_fusion_devkit_f1.pt")
                    try:
                        torch.save(best_checkpoint_devkit_f1,
                                   best_model_path_devkit_f1)
                        print(
                            f"Saved best devkit F1 model checkpoint to {best_model_path_devkit_f1}")
                    except Exception as e:
                        print(f"Error saving best devkit F1 checkpoint: {e}")
            current_iou = val_info['simple_iou']
            if current_iou > max_iou:
                max_iou = current_iou
                # Optional: save best IoU model

            # --- Validation Visualization Block Removed ---
            # Visualization is now handled inside get_val_info_fusion in tools.py

            print(f"--- Finished Validation Epoch {epoch+1} ---")
            model.train()  # Ensure model is back in train mode

        # --- Increment Epoch --- #
        epoch += 1

    # --- End Training Loop --- #
    print("Training finished.")
    writer.close()

# Note: Ensure that the functions called within (like compile_data, compile_model, get_val_info_fusion, etc.)
# are correctly defined and imported elsewhere in your project.
