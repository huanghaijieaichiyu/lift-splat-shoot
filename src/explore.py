"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from .models import compile_model
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map, get_local_map,
                    get_val_info_fusion)
from .data import compile_data
from .nuscenes_info import load_nuscenes_infos  # 导入数据集缓存加载函数
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # 正确导入matplotlib的colormap模块
from matplotlib import gridspec
import os.path

import torch
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import torch.nn as nn

from .tools import save_path

# 在PIL.Image中，常量值0表示水平翻转
FLIP_LEFT_RIGHT = 0  # PIL标准值

plt.switch_backend('Agg')


def check_and_load_cache(dataroot, version):
    """
    检查并加载数据集缓存

    Args:
        dataroot: 数据集根目录
        version: 数据集版本，例如'mini'

    Returns:
        loaded: 是否成功加载缓存
        info: 缓存信息（如果加载成功）或None
    """
    # 配置NuScenes版本
    version_str = f'v1.0-{version}'
    max_sweeps = 10  # 设置最大扫描帧数

    try:
        from .nuscenes_info import load_nuscenes_infos
        print(f"检查NuScenes缓存信息...")

        # 这里会自动创建缓存（如果不存在）
        nusc_infos = load_nuscenes_infos(
            dataroot, version=version_str, max_sweeps=max_sweeps)

        # 验证缓存数据的完整性
        if 'infos' in nusc_infos and 'train' in nusc_infos['infos'] and 'val' in nusc_infos['infos']:
            train_samples = len(nusc_infos['infos']['train'])
            val_samples = len(nusc_infos['infos']['val'])
            print(f"✓ 缓存加载成功！包含{train_samples}个训练样本和{val_samples}个验证样本")
            return True, nusc_infos
        else:
            print("✗ 缓存数据结构不完整")
            return False, None

    except FileNotFoundError as e:
        print(f"✗ 缓存文件不存在或无法访问: {e}")
        print("  请确保数据集路径正确，并且有写入权限")
        return False, None

    except Exception as e:
        print(f"✗ 缓存处理出错: {e}")
        print("  将尝试从原始数据加载")
        return False, None


def lidar_check(version,
                dataroot='/data/nuscenes',
                show_lidar=True,
                viz_train=False,
                nepochs=1,

                H=900, W=1600,
                resize_lim=(0.193, 0.225),
                final_dim=(128, 352),
                bot_pct_lim=(0.0, 0.22),
                rot_lim=(-5.4, 5.4),
                rand_flip=True,

                xbound=[-50.0, 50.0, 0.5],
                ybound=[-50.0, 50.0, 0.5],
                zbound=[-10.0, 10.0, 20.0],
                dbound=[4.0, 45.0, 1.0],

                bsz=1,
                nworkers=10,
                ):
    # 首先检查是否有缓存数据
    cache_loaded, nusc_infos = check_and_load_cache(dataroot, version)

    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': cams,
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='vizdata')

    loader = trainloader if viz_train else valloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)

    rat = H / W
    val = 10.1
    fig = plt.figure(figsize=(val + val / 3 * 2 * rat * 3, val / 3 * 2 * rat))
    gs = gridspec.GridSpec(2, 6, width_ratios=(
        1, 1, 1, 2 * rat, 2 * rat, 2 * rat))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    for epoch in range(nepochs):
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, pts, binimgs) in enumerate(loader):

            img_pts = model.get_geometry(
                rots, trans, intrins, post_rots, post_trans)

            for si in range(imgs.shape[0]):
                plt.clf()
                final_ax = plt.subplot(gs[:, 5:6])
                for imgi, img in enumerate(imgs[si]):
                    ego_pts = ego_to_cam(
                        pts[si], rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                    mask = get_only_in_img_mask(ego_pts, H, W)
                    plot_pts = post_rots[si, imgi].matmul(
                        ego_pts) + post_trans[si, imgi].unsqueeze(1)

                    ax = plt.subplot(gs[imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    plt.imshow(showimg)
                    if show_lidar:
                        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=ego_pts[2, mask],
                                    s=5, alpha=0.1, cmap='jet')
                    # plot_pts = post_rots[si, imgi].matmul(img_pts[si, imgi].view(-1, 3).t()) + post_trans[si, imgi].unsqueeze(1)
                    # plt.scatter(img_pts[:, :, :, 0].view(-1), img_pts[:, :, :, 1].view(-1), s=1)
                    plt.axis('off')

                    plt.sca(final_ax)
                    plt.plot(img_pts[si, imgi, :, :, :, 0].view(-1), img_pts[si, imgi, :, :, :, 1].view(-1), '.',
                             label=cams[imgi].replace('_', ' '))

                plt.legend(loc='upper right')
                final_ax.set_aspect('equal')
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))

                ax = plt.subplot(gs[:, 3:4])
                plt.scatter(pts[si, 0], pts[si, 1],
                            c=pts[si, 2], vmin=-5, vmax=5, s=5)
                plt.xlim((-50, 50))
                plt.ylim((-50, 50))
                ax.set_aspect('equal')

                ax = plt.subplot(gs[:, 4:5])
                plt.imshow(binimgs[si].squeeze(
                    0).T, origin='lower', cmap='Greys', vmin=0, vmax=1)

                imname = f'lcheck{epoch:03}_{batchi:05}_{si:02}.jpg'
                print('saving', imname)
                plt.savefig(imname)


def cumsum_check(version,
                 dataroot='/data/nuscenes',
                 gpuid=1,

                 H=900, W=1600,
                 resize_lim=(0.193, 0.225),
                 final_dim=(128, 352),
                 bot_pct_lim=(0.0, 0.22),
                 rot_lim=(-5.4, 5.4),
                 rand_flip=True,

                 xbound=[-50.0, 50.0, 0.5],
                 ybound=[-50.0, 50.0, 0.5],
                 zbound=[-10.0, 10.0, 20.0],
                 dbound=[4.0, 45.0, 1.0],

                 bsz=4,
                 nworkers=10,
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
        'Ncams': 5,
    }
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
    loader = trainloader

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model.eval()
    for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
        model.use_quickcumsum = False
        model.zero_grad()
        out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
        out.mean().backward()

        # 安全访问梯度
        try:
            grad_value = model.camencode.depthnet.weight.grad.mean().item() \
                if hasattr(model, 'camencode') and hasattr(model.camencode, 'depthnet') else "N/A"
            print('autograd:    ', out.mean().detach().item(), grad_value)
        except (AttributeError, ValueError) as e:
            print(f"访问模型属性时出错: {e}")
            print('autograd:    ', out.mean().detach().item(), "N/A")

        model.use_quickcumsum = True
        model.zero_grad()
        out = model(imgs.to(device),
                    rots.to(device),
                    trans.to(device),
                    intrins.to(device),
                    post_rots.to(device),
                    post_trans.to(device),
                    )
        out.mean().backward()

        # 安全访问梯度
        try:
            grad_value = model.camencode.depthnet.weight.grad.mean().item() \
                if hasattr(model, 'camencode') and hasattr(model.camencode, 'depthnet') else "N/A"
            print('quick cumsum:', out.mean().detach().item(), grad_value)
        except (AttributeError, ValueError) as e:
            print(f"访问模型属性时出错: {e}")
            print('quick cumsum:', out.mean().detach().item(), "N/A")
        print()


def eval_model_iou(version,
                   modelf,
                   dataroot='/data/nuscenes',
                   gpuid=0,
                   model_type='standard',
                   map_layers=['drivable_area'],

                   H=900, W=1600,
                   resize_lim=(0.193, 0.225),
                   final_dim=(128, 352),
                   bot_pct_lim=(0.0, 0.22),
                   rot_lim=(-5.4, 5.4),
                   rand_flip=True,

                   xbound=[-50.0, 50.0, 0.5],
                   ybound=[-50.0, 50.0, 0.5],
                   zbound=[-10.0, 10.0, 20.0],
                   dbound=[4.0, 45.0, 1.0],

                   lidar_xbound=[-50.0, 50.0, 0.5],
                   lidar_ybound=[-50.0, 50.0, 0.5],
                   lidar_inC=1,

                   bsz=4,
                   nworkers=10,
                   ):
    """
    评估模型性能，包括简单指标和（可选的）基于NuScenes Devkit的指标。
    """
    # --- 检查缓存 ---
    check_and_load_cache(dataroot, version)

    # --- 配置 --- #
    grid_conf = {
        'xbound': xbound, 'ybound': ybound, 'zbound': zbound, 'dbound': dbound,
        'lidar_xbound': lidar_xbound, 'lidar_ybound': lidar_ybound  # Fusion需要
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim, 'final_dim': final_dim, 'rot_lim': rot_lim,
        'H': H, 'W': W, 'rand_flip': rand_flip, 'bot_pct_lim': bot_pct_lim,
        'cams': cams, 'Ncams': len(cams),  # 使用cams列表的长度
    }

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # --- 初始化 NuScenes API (如果需要) --- #
    nusc = None
    if model_type == 'fusion':  # 只有 fusion 模型评估需要 nusc API
        try:
            from nuscenes import NuScenes
            print("Initializing NuScenes API for devkit evaluation...")
            nusc_version_str = f'v1.0-{version}'
            nusc = NuScenes(version=nusc_version_str,
                            dataroot=dataroot, verbose=False)
            print("NuScenes API initialized.")
        except ImportError:
            print(
                "WARNING: nuscenes-devkit not found. Devkit evaluation will be skipped.")
        except Exception as e:
            print(
                f"WARNING: Failed to initialize NuScenes API: {e}. Devkit evaluation will be skipped.")

    # --- 加载模型和数据加载器 --- #
    print('loading model checkpoint:', modelf)
    model: nn.Module | None = None
    try:
        checkpoint = torch.load(modelf, map_location='cpu')
        state_dict = checkpoint.get('net', checkpoint)

        if model_type == 'fusion':
            print("Model type: 'fusion'. Loading FusionData parser...")
            parser = 'fusiondata'
            model = compile_model(grid_conf, data_aug_conf,
                                  outC=1, lidar_inC=lidar_inC)
        elif model_type == 'standard':
            print("Model type: 'standard'. Loading SegmentationData parser...")
            parser = 'segmentationdata'
            model = compile_model(grid_conf, data_aug_conf, outC=1)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Choose 'fusion' or 'standard'.")

        if model is not None and isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
            print("Model state loaded successfully.")
        else:
            raise ValueError("Model compilation or state_dict loading failed.")

        model.to(device)
        model.eval()

        # 加载验证数据加载器
        _, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                    grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                    parser_name=parser, lidar_inC=lidar_inC)  # 传递 lidar_inC

    except Exception as e:
        print(f"Error during model/data loading: {e}")
        return

    # --- 执行评估 --- #
    # 创建一个虚拟/简单的损失函数，因为get_val_info*需要它
    # 注意：如果只关心指标，损失值本身可以忽略
    dummy_loss_fn = nn.BCEWithLogitsLoss().to(device)

    print("\nStarting evaluation...")
    if model_type == 'fusion':
        print(f"Evaluating Fusion model using devkit layers: {map_layers}")
        # 调用 fusion 版本的评估函数
        eval_results = get_val_info_fusion(
            model=model,
            valloader=valloader,
            loss_fn=dummy_loss_fn,  # 传递虚拟损失
            device=device,
            nusc=nusc,  # 传递初始化的 nusc 对象
            grid_conf=grid_conf,
            writer=None,  # 不需要 writer
            global_step=0,  # 不需要 global_step
            map_layers=map_layers,
            final_dim_vis=None,  # 不需要可视化参数
            D_depth=None  # 不需要可视化参数
        )
    else:  # model_type == 'standard'
        print("Evaluating Standard model (simple metrics only)...")
        # 调用标准版本的评估函数
        eval_results = get_val_info(
            model=model,
            valloader=valloader,
            loss_fn=dummy_loss_fn,
            device=device
        )

    # --- 打印结果 --- #
    print("\n--- Evaluation Results ---")
    if eval_results:
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("Evaluation did not return any results.")
    print("-------------------------")


def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=0,
                    viz_train=False,
                    model_type='standard',

                    H=900, W=1600,
                    resize_lim=(0.193, 0.225),
                    final_dim=(128, 352),
                    bot_pct_lim=(0.0, 0.22),
                    rot_lim=(-5.4, 5.4),
                    rand_flip=True,

                    xbound=[-50.0, 50.0, 0.5],
                    ybound=[-50.0, 50.0, 0.5],
                    zbound=[-10.0, 10.0, 20.0],
                    dbound=[4.0, 45.0, 1.0],

                    # Add lidar specific bounds if needed for FusionNet
                    lidar_xbound=[-50.0, 50.0, 0.5],
                    lidar_ybound=[-50.0, 50.0, 0.5],
                    lidar_inC=1,  # Default, adjust if FusionNet uses different lidar_inC

                    bsz=1,
                    nworkers=1,
                    ):
    # 首先检查是否有缓存数据
    check_and_load_cache(dataroot, version)

    grid_conf = {
        'xbound': xbound, 'ybound': ybound, 'zbound': zbound, 'dbound': dbound,
        # Include lidar bounds needed by FusionData and potentially FusionNet compilation
        'lidar_xbound': lidar_xbound, 'lidar_ybound': lidar_ybound
    }
    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    data_aug_conf = {
        'resize_lim': resize_lim, 'final_dim': final_dim, 'rot_lim': rot_lim,
        'H': H, 'W': W, 'rand_flip': rand_flip, 'bot_pct_lim': bot_pct_lim,
        'cams': cams, 'Ncams': 6,
    }

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # --- Model Loading and Parser Selection ---
    print('loading model checkpoint:', modelf)
    model: nn.Module | None = None
    try:
        checkpoint = torch.load(modelf, map_location='cpu')
        state_dict = checkpoint.get('net', checkpoint)

        if model_type == 'fusion':
            print("Model type specified as 'fusion'. Using FusionData parser.")
            is_fusion_model = True
            parser = 'fusiondata'
            model = compile_model(grid_conf, data_aug_conf,
                                  outC=1, lidar_inC=lidar_inC)
        elif model_type == 'standard':
            print("Model type specified as 'standard'. Using SegmentationData parser.")
            is_fusion_model = False
            parser = 'segmentationdata'
            model = compile_model(grid_conf, data_aug_conf, outC=1)
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Choose 'fusion' or 'standard'.")

        if model is not None and isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
            print("Model state loaded successfully.")
        else:
            raise ValueError("Model was not compiled correctly.")

    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    model.to(device)
    model.eval()
    # --- End Model Loading ---

    # --- Data Loading (using selected parser) ---
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name=parser, lidar_inC=lidar_inC)
    loader: torch.utils.data.DataLoader = trainloader if viz_train else valloader
    # --- End Data Loading ---

    nusc_maps = get_nusc_maps(map_folder)
    dx, bx, _ = gen_dx_bx(grid_conf['xbound'],
                          grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    # Scene to map mapping (remains the same)
    scene2map = {}
    nusc_api = None  # Initialize nusc_api
    try:
        # Check if dataset exists and has 'nusc' attribute
        # Use type assertion if confident about the dataset type
        from .data import NuscData  # Import base class for type check
        if loader is not None and isinstance(loader.dataset, NuscData) and loader.dataset.nusc is not None:
            nusc_api = loader.dataset.nusc  # Store nusc api for later use
            for rec in nusc_api.scene:
                try:
                    log = nusc_api.get('log', rec['log_token'])
                    scene2map[rec['name']] = log['location']
                except (KeyError, AttributeError) as e:
                    print(f"处理场景数据时出错: {e}")
        else:
            print("Dataset or nusc attribute not available, skipping map loading.")
    except AttributeError:
        print("数据集没有nusc属性，跳过地图加载")

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3*fW*val, (1.5*fW + 2*fH)*val))
    gs = gridspec.GridSpec(3, 3, height_ratios=(1.5*fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    # Ensure logdir exists before saving path
    base_logdir = 'runs/prediction'
    os.makedirs(base_logdir, exist_ok=True)  # Create base dir if needed
    path = save_path(base_logdir)  # Get unique subfolder
    os.makedirs(path, exist_ok=True)  # Create unique subfolder
    print(f"Saving visualizations to: {path}")

    counter = 0
    with torch.no_grad():
        # --- Modified Data Iteration and Model Call ---
        data_iter = tqdm(enumerate(loader), total=len(
            loader), desc="Visualizing Predictions")
        for batchi, batch_data in data_iter:
            # Unpack data based on the parser used
            if is_fusion_model:
                # Expect 9 items: imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev, sample_tokens
                if len(batch_data) != 9:
                    print(
                        f"Warning: FusionData expected 9 items, got {len(batch_data)}. Skipping batch {batchi}.")
                    continue
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs, lidar_bev, sample_tokens = batch_data
                # Move lidar_bev to device
                lidar_bev_dev = lidar_bev.to(device)
            else:
                # Expect 7 items: imgs, rots, trans, intrins, post_rots, post_trans, binimgs
                if len(batch_data) != 7:
                    print(
                        f"Warning: SegmentationData expected 7 items, got {len(batch_data)}. Skipping batch {batchi}.")
                    continue
                imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch_data
                lidar_bev_dev = None  # No lidar input for non-fusion model

            # Move common tensors to device
            imgs_dev = imgs.to(device)
            rots_dev = rots.to(device)
            trans_dev = trans.to(device)
            intrins_dev = intrins.to(device)
            post_rots_dev = post_rots.to(device)
            post_trans_dev = post_trans.to(device)

            # Model forward pass
            if is_fusion_model:
                model_output = model(imgs_dev, rots_dev, trans_dev, intrins_dev,
                                     post_rots_dev, post_trans_dev, lidar_bev_dev)
            else:
                model_output = model(imgs_dev, rots_dev, trans_dev, intrins_dev,
                                     post_rots_dev, post_trans_dev)

            # Handle potential tuple output (e.g., if model returns depth)
            if isinstance(model_output, tuple):
                out = model_output[0]  # Assume prediction is the first element
            else:
                out = model_output

            out = out.sigmoid().cpu()  # Process predictions

            # --- Visualization Logic (largely unchanged) ---
            for si in range(imgs.shape[0]):
                plt.clf()
                # Plot camera images
                # Use img_tensor to avoid confusion
                for imgi, img_tensor in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    # showimg = denormalize_img(img) # <-- 移除过早的反归一化

                    # 直接在张量上操作
                    img_to_show = img_tensor.cpu()  # Work with CPU tensor

                    # Check if it's one of the bottom row images (indices 3, 4, 5)
                    if imgi >= 3:
                        # Permute to HWC for numpy, flip, permute back to CHW
                        # Ensure img_to_show is a tensor before permute
                        if isinstance(img_to_show, torch.Tensor):
                            showimg_np = img_to_show.permute(1, 2, 0).numpy()
                            showimg_flipped_np = np.ascontiguousarray(
                                np.fliplr(showimg_np))
                            img_to_show = torch.from_numpy(showimg_flipped_np).permute(
                                2, 0, 1)  # Update img_to_show with flipped tensor
                        else:
                            # This case should ideally not happen if input 'imgs' is correct
                            print(
                                f"Warning: Input image is not a tensor (type: {type(img_to_show)}), skipping flip.")

                    # Denormalize final tensor just before plotting
                    # Pass the final tensor (original or flipped)
                    showimg_denorm = denormalize_img(img_to_show)
                    plt.imshow(showimg_denorm)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '),
                                 (0.01, 0.92), xycoords='axes fraction')

                # Plot BEV prediction and map
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine_name, spine in ax.spines.items():  # Corrected spine iteration
                    plt.setp(spine, color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0),
                                   label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31),
                                   label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # Plot static map
                try:
                    # Access sample token differently depending on parser
                    current_sample_token = None
                    if is_fusion_model and 'sample_tokens' in locals() and sample_tokens is not None and si < len(sample_tokens):
                        current_sample_token = sample_tokens[si]
                    elif hasattr(loader.dataset, 'ixes'):
                        # Fallback for SegmentationData or if sample_tokens missing
                        # Ensure batch_size is not None before multiplication
                        current_batch_size = loader.batch_size
                        if current_batch_size is not None:
                            rec_idx = batchi * current_batch_size + si
                            # Check dataset and ixes existence and bounds
                            # Check type of dataset again if necessary
                            if isinstance(loader.dataset, NuscData) and hasattr(loader.dataset, 'ixes') and loader.dataset.ixes is not None and rec_idx < len(loader.dataset.ixes):
                                current_sample_token = loader.dataset.ixes[rec_idx]['token']
                        else:
                            print(
                                f"Warning: loader.batch_size is None. Cannot calculate rec_idx for batch {batchi}.")

                    # Use cached nusc_api if available
                    if current_sample_token and nusc_api is not None:
                        rec = nusc_api.get('sample', current_sample_token)
                        plot_nusc_map(rec, nusc_maps, nusc_api,
                                      scene2map, dx, bx)
                    # else: # Optional: Print warning if map cannot be plotted
                    #    print(f"Warning: Could not plot map for batch {batchi}, sample {si}")

                # Added NameError for sample_tokens
                except (AttributeError, IndexError, KeyError, NameError) as e:
                    print(f"访问数据集属性以绘制地图时出错: {e}")

                plt.xlim((out.shape[-1], 0))  # Use last dim for width
                # Use second to last dim for height
                plt.ylim((0, out.shape[-2]))
                add_ego(bx, dx)

                imname = f'eval{batchi:06}_{si:03}.jpg'
                # print('saving', imname) # Reduce print frequency
                plt.savefig(os.path.join(path, imname))
            # --- End Visualization Logic ---

            # Increment counter by batch size processed
            counter += imgs.shape[0]
        data_iter.close()  # Close tqdm iterator
