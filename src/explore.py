"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from .models import compile_model
from .tools import (ego_to_cam, get_only_in_img_mask, denormalize_img,
                    SimpleLoss, get_val_info, add_ego, gen_dx_bx,
                    get_nusc_maps, plot_nusc_map)
from .data import compile_data
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os.path

import torch
import numpy as np

from .tools import save_path

plt.switch_backend('Agg')


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
        print('autograd:    ', out.mean().detach().item(),
              model.camencode.depthnet.weight.grad.mean().item())

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
        print('quick cumsum:', out.mean().detach().item(),
              model.camencode.depthnet.weight.grad.mean().item())
        print()


def eval_model_iou(version,
                   modelf,
                   dataroot='/data/nuscenes',
                   gpuid=0,

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

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading', modelf)
    if torch.load(modelf)["net"] is not None:
        model.load_state_dict(torch.load(modelf)['net'])
    else:
        model.load_state_dict(torch.load(modelf))
    model.to(device)

    loss_fn = SimpleLoss(1.0).cuda(gpuid)

    model.eval()
    val_info = get_val_info(model, valloader, loss_fn, device)
    print(val_info)


def viz_model_preds(version,
                    modelf,
                    dataroot='/data/nuscenes',
                    map_folder='/data/nuscenes/mini',
                    gpuid=0,
                    viz_train=False,  # lidar

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
                    nworkers=1,
                    ):
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
                                          parser_name='segmentationdata')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1, model='lss')
    print('loading', modelf)
    try:
        model.load_state_dict(torch.load(modelf)['net'])
    except:
        model.load_state_dict(torch.load(modelf))
    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'],
                          grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (1.5 * fW + 2 * fH) * val))
    gs = gridspec.GridSpec(3, 3, height_ratios=(1.5 * fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
    logdir = 'runs/prediction'
    path = save_path(logdir)
    os.makedirs(path)
    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
            out = model(imgs.to(device),
                        rots.to(device),
                        trans.to(device),
                        intrins.to(device),
                        post_rots.to(device),
                        post_trans.to(device),
                        )
            out = out.sigmoid().cpu()

            for si in range(imgs.shape[0]):
                plt.clf()
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[1 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # 修复PIL翻转常量问题
                    if imgi > 2:
                        # 使用 PIL.Image.FLIP_LEFT_RIGHT 常量
                        showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '),
                                 (0.01, 0.92), xycoords='axes fraction')

                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.setp(ax.spines.values(), color='b', linewidth=2)
                plt.legend(handles=[
                    mpatches.Patch(color=(0.0, 0.0, 1.0, 1.0),
                                   label='Output Vehicle Segmentation'),
                    mpatches.Patch(color='#76b900', label='Ego Vehicle'),
                    mpatches.Patch(color=(1.00, 0.50, 0.31, 0.8),
                                   label='Map (for visualization purposes only)')
                ], loc=(0.01, 0.86))
                plt.imshow(out[si].squeeze(0), vmin=0, vmax=1, cmap='Blues')

                # plot static map (improves visualization)
                rec = loader.dataset.ixes[counter]
                plot_nusc_map(rec, nusc_maps, loader.dataset.nusc,
                              scene2map, dx, bx)
                plt.xlim((out.shape[3], 0))
                plt.ylim((0, out.shape[3]))
                add_ego(bx, dx)

                imname = f'eval{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(os.path.join(path, imname))
                counter += 1


def viz_3d_detection(version,
                     modelf,
                     dataroot='/data/nuscenes',
                     map_folder='/data/nuscenes/mini',
                     gpuid=0,
                     viz_train=False,

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
                     nworkers=1,
                     num_classes=10,
                     colormap='tab10',  # 类别颜色图
                     conf_thresh=0.5,   # 置信度阈值
                     ):
    """
    可视化3D目标检测模型预测结果
    """
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

    # 使用detection3d数据解析器
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='detection3d')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 加载模型，使用beve模型进行3D检测
    model = compile_model(grid_conf, data_aug_conf,
                          outC=num_classes, model='beve', num_classes=num_classes)
    print('loading', modelf)
    try:
        checkpoint = torch.load(modelf)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'],
                          grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    # 设置颜色映射，用于不同类别的可视化
    cmap = plt.cm.get_cmap(colormap, num_classes)

    # 设置可视化图表大小和布局
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (2.5 * fW + 2 * fH) * val))
    gs = gridspec.GridSpec(4, 3, height_ratios=(1.5 * fW, fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    # 创建保存目录
    logdir = 'runs/3d_detection_visualization'
    path = save_path(logdir)
    os.makedirs(path, exist_ok=True)

    # 类别名称（示例）
    class_names = [f'Class {i}' for i in range(num_classes)]

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, targets_list) in enumerate(loader):
            # 前向传播获取预测结果
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              )

            # 获取预测的类别、边界框等信息
            pred_cls = preds['cls_preds'].sigmoid().cpu() if isinstance(
                preds, dict) else preds[:, :num_classes].sigmoid().cpu()
            pred_boxes = preds['reg_preds'].cpu() if isinstance(
                preds, dict) else preds[:, num_classes:].cpu()

            for si in range(imgs.shape[0]):
                plt.clf()

                # 绘制相机图像
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[2 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # 翻转底部图像
                    if imgi > 2:
                        showimg = showimg.transpose(
                            Image.TRANSPOSE_FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '),
                                 (0.01, 0.92), xycoords='axes fraction')

                # 绘制BEV检测结果
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine in ax.spines.values():
                    spine.set_color('b')
                    spine.set_linewidth(2)

                # 绘制地图
                rec = loader.dataset.ixes[counter] if hasattr(
                    loader.dataset, 'ixes') else None
                if rec:
                    plot_nusc_map(rec, nusc_maps,
                                  loader.dataset.nusc, scene2map, dx, bx)

                # 可视化预测结果
                # 创建一个热力图来显示检测置信度
                detection_map = np.zeros(
                    (pred_cls.shape[2], pred_cls.shape[3], 4))

                # 为每个类别创建图例项
                legend_handles = []

                # 对每个类别进行可视化
                for cls_idx in range(num_classes):
                    cls_mask = pred_cls[si, cls_idx] > conf_thresh
                    if cls_mask.sum() > 0:
                        # 使用颜色映射为每个类别分配颜色
                        color = cmap(cls_idx)
                        # 在检测图上用对应颜色标记
                        detection_map[cls_mask] = color
                        # 添加到图例
                        legend_handles.append(mpatches.Patch(
                            color=color, label=class_names[cls_idx]))

                # 显示检测结果
                plt.imshow(detection_map, origin='lower')
                plt.xlim((detection_map.shape[1], 0))
                plt.ylim((0, detection_map.shape[0]))

                # 添加自车位置
                add_ego(bx, dx)

                # 添加图例
                if legend_handles:
                    plt.legend(handles=legend_handles, loc=(
                        0.01, 0.86), fontsize='small')

                # 显示置信度热力图
                ax = plt.subplot(gs[1, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine in ax.spines.values():
                    spine.set_color('g')
                    spine.set_linewidth(2)
                plt.title("Confidence Heatmap")

                # 计算所有类别的最大置信度
                conf_heatmap = pred_cls[si].max(dim=0)[0]
                plt.imshow(conf_heatmap, cmap='hot', vmin=0, vmax=1)
                plt.colorbar(label='Confidence')

                plt.xlim((conf_heatmap.shape[1], 0))
                plt.ylim((0, conf_heatmap.shape[0]))

                # 保存图像
                imname = f'3d_detection_{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(os.path.join(path, imname))
                counter += 1

                # 限制处理的批次数，避免生成过多图像
                if counter >= 20:
                    print(f"已生成{counter}张可视化图像，停止处理更多批次。")
                    return


def viz_fusion_detection(version,
                         modelf,
                         dataroot='/data/nuscenes',
                         map_folder='/data/nuscenes/mini',
                         gpuid=0,
                         viz_train=False,

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
                         nworkers=1,
                         num_classes=10,
                         lidar_channels=18,
                         colormap='tab10',
                         conf_thresh=0.5,
                         ):
    """
    可视化多模态融合模型（相机+LiDAR）的预测结果
    """
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

    # 使用多模态数据加载器
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='multimodal_detection')
    loader = trainloader if viz_train else valloader
    nusc_maps = get_nusc_maps(map_folder)

    device = torch.device(
        'cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    # 加载融合模型
    model = compile_model(grid_conf, data_aug_conf, outC=num_classes*9,
                          model='fusion', num_classes=num_classes,
                          lidar_channels=lidar_channels)
    print('loading', modelf)
    try:
        checkpoint = torch.load(modelf)
        if 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    model.to(device)

    dx, bx, _ = gen_dx_bx(grid_conf['xbound'],
                          grid_conf['ybound'], grid_conf['zbound'])
    dx, bx = dx[:2].numpy(), bx[:2].numpy()

    scene2map = {}
    for rec in loader.dataset.nusc.scene:
        log = loader.dataset.nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    # 设置颜色映射，用于不同类别的可视化
    cmap = plt.cm.get_cmap(colormap, num_classes)

    # 设置可视化图表大小和布局 - 增加一行用于LiDAR可视化
    val = 0.01
    fH, fW = final_dim
    fig = plt.figure(figsize=(3 * fW * val, (3.5 * fW + 2 * fH) * val))
    gs = gridspec.GridSpec(5, 3, height_ratios=(1.5 * fW, fW, fW, fH, fH))
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    # 创建保存目录
    logdir = 'runs/fusion_detection_visualization'
    path = save_path(logdir)
    os.makedirs(path, exist_ok=True)

    # 类别名称（示例）
    class_names = [f'Class {i}' for i in range(num_classes)]

    model.eval()
    counter = 0
    with torch.no_grad():
        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_list) in enumerate(loader):
            # 前向传播获取预测结果
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(imgs.to(device),
                              rots.to(device),
                              trans.to(device),
                              intrins.to(device),
                              post_rots.to(device),
                              post_trans.to(device),
                              lidar_bev.to(device)
                              )

            # 获取预测的类别、边界框等信息
            pred_cls = preds['cls_preds'].sigmoid().cpu() if isinstance(
                preds, dict) else preds[:, :num_classes].sigmoid().cpu()
            pred_boxes = preds['reg_preds'].cpu() if isinstance(
                preds, dict) else preds[:, num_classes:].cpu()

            for si in range(imgs.shape[0]):
                plt.clf()

                # 绘制相机图像
                for imgi, img in enumerate(imgs[si]):
                    ax = plt.subplot(gs[3 + imgi // 3, imgi % 3])
                    showimg = denormalize_img(img)
                    # 翻转底部图像
                    if imgi > 2:
                        showimg = showimg.transpose(
                            Image.TRANSPOSE_FLIP_LEFT_RIGHT)
                    plt.imshow(showimg)
                    plt.axis('off')
                    plt.annotate(cams[imgi].replace('_', ' '),
                                 (0.01, 0.92), xycoords='axes fraction')

                # 绘制BEV检测结果
                ax = plt.subplot(gs[0, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine in ax.spines.values():
                    spine.set_color('b')
                    spine.set_linewidth(2)

                # 绘制地图
                rec = loader.dataset.ixes[counter] if hasattr(
                    loader.dataset, 'ixes') else None
                if rec:
                    plot_nusc_map(rec, nusc_maps,
                                  loader.dataset.nusc, scene2map, dx, bx)

                # 可视化预测结果
                # 创建一个热力图来显示检测置信度
                detection_map = np.zeros(
                    (pred_cls.shape[2], pred_cls.shape[3], 4))

                # 为每个类别创建图例项
                legend_handles = []

                # 对每个类别进行可视化
                for cls_idx in range(num_classes):
                    cls_mask = pred_cls[si, cls_idx] > conf_thresh
                    if cls_mask.sum() > 0:
                        # 使用颜色映射为每个类别分配颜色
                        color = cmap(cls_idx)
                        # 在检测图上用对应颜色标记
                        detection_map[cls_mask] = color
                        # 添加到图例
                        legend_handles.append(mpatches.Patch(
                            color=color, label=class_names[cls_idx]))

                # 显示检测结果
                plt.imshow(detection_map, origin='lower')
                plt.xlim((detection_map.shape[1], 0))
                plt.ylim((0, detection_map.shape[0]))

                # 添加自车位置
                add_ego(bx, dx)

                # 添加图例
                if legend_handles:
                    plt.legend(handles=legend_handles, loc=(
                        0.01, 0.86), fontsize='small')

                # 显示置信度热力图
                ax = plt.subplot(gs[1, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine in ax.spines.values():
                    spine.set_color('g')
                    spine.set_linewidth(2)
                plt.title("Confidence Heatmap")

                # 计算所有类别的最大置信度
                conf_heatmap = pred_cls[si].max(dim=0)[0]
                plt.imshow(conf_heatmap, cmap='hot', vmin=0, vmax=1)
                plt.colorbar(label='Confidence')

                plt.xlim((conf_heatmap.shape[1], 0))
                plt.ylim((0, conf_heatmap.shape[0]))

                # 显示LiDAR点云BEV
                ax = plt.subplot(gs[2, :])
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                for spine in ax.spines.values():
                    spine.set_color('r')
                    spine.set_linewidth(2)
                plt.title("LiDAR BEV")

                # 可视化LiDAR BEV图
                # 如果lidar_bev有多个通道，取前3个或RGB组合
                lidar_vis = lidar_bev[si].cpu().numpy()
                if lidar_vis.shape[0] >= 3:
                    # 使用前三个通道作为RGB
                    lidar_rgb = lidar_vis[:3].transpose(1, 2, 0)
                    # 归一化到[0,1]
                    lidar_rgb = (lidar_rgb - lidar_rgb.min()) / \
                        (lidar_rgb.max() - lidar_rgb.min() + 1e-6)
                    plt.imshow(lidar_rgb)
                else:
                    # 单通道显示
                    plt.imshow(lidar_vis[0], cmap='viridis')

                plt.xlim((lidar_vis.shape[2], 0))
                plt.ylim((0, lidar_vis.shape[1]))

                # 保存图像
                imname = f'fusion_detection_{batchi:06}_{si:03}.jpg'
                print('saving', imname)
                plt.savefig(os.path.join(path, imname))
                counter += 1

                # 限制处理的批次数，避免生成过多图像
                if counter >= 20:
                    print(f"已生成{counter}张可视化图像，停止处理更多批次。")
                    return
