"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm
from torch.cuda.amp.autocast_mode import autocast
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import save_path
from .tools import SimpleLoss, get_batch_iou, get_val_info


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
          nworkers=10,
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
    val_step = 10 if version == 'mini' else 100
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
        Iou = [0]
        pbar = tqdm(enumerate(trainloader), total=len(
            trainloader), colour='#8762A5', ncols=200)

        for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in pbar:
            t0 = time()
            opt.zero_grad()
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
            t1 = time()

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
            if iou > max(Iou):
                best_model = os.path.join(path, "best.pt")
                torch.save(model, best_model)

            Iou.append(iou)
            writer.add_scalar('train/iou', iou, epoch + 1)
            writer.add_scalar('train/step_time', t1 - t0, epoch + 1)
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
