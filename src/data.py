"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot,
                      'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot,
                       'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot,
                      'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (
                        rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            sens = self.nusc.get('calibrated_sensor',
                                 samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                       resize=resize,
                                                       resize_dims=resize_dims,
                                                       crop=crop,
                                                       flip=flip,
                                                       rotate=rotate,
                                                       )

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            # 在NuScenes数据集中，有些标注没有velocity字段
            # 构建Box对象时需传入默认的velocity元组 (np.nan, np.nan, np.nan)
            velocity = (np.nan, np.nan, np.nan)  # 默认值
            if 'velocity' in inst:
                velocity = tuple(inst['velocity'])
            box = Box(inst['translation'], inst['size'], Quaternion(
                inst['rotation']), velocity=velocity)
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], (1.0,))

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)
        binimg = self.get_binimg(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


class Detection3DData(NuscData):
    """
    3D目标检测数据集，生成检测任务所需的目标
    """

    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        super(Detection3DData, self).__init__(
            nusc, is_train, data_aug_conf, grid_conf)
        self.is_train = is_train
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'],
                               grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx, bx, nx

        # 设置类别解析
        self.DETECTION_CLASSES = ['car', 'truck', 'bus', 'trailer',
                                  'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
                                  'traffic_cone', 'barrier']
        # 添加类别映射
        self.category_map = {
            'vehicle.car': 1,
            'vehicle.truck': 2,
            'vehicle.bus': 3,
            'vehicle.trailer': 4,
            'vehicle.construction': 5,
            'human.pedestrian': 6,
            'vehicle.motorcycle': 7,
            'vehicle.bicycle': 8,
            'movable_object.traffic_cone': 9,
            'movable_object.barrier': 10
        }
        self.cls_mean_size = {
            'car': np.array([4.63, 1.97, 1.74]),
            'truck': np.array([6.52, 2.51, 2.80]),
            'bus': np.array([10.5, 2.94, 3.47]),
            'trailer': np.array([8.17, 2.85, 3.23]),
            'construction_vehicle': np.array([6.82, 3.24, 3.41]),
            'pedestrian': np.array([0.73, 0.67, 1.72]),
            'motorcycle': np.array([2.11, 0.77, 1.47]),
            'bicycle': np.array([1.70, 0.6, 1.28]),
            'traffic_cone': np.array([0.41, 0.41, 1.07]),
            'barrier': np.array([2.31, 0.34, 1.27]),
        }

    def __getitem__(self, index):
        rec = self.ixes[index]

        # 获取图像数据
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)

        # 获取检测目标
        targets = self.get_detection_targets(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, targets

    def get_detection_targets(self, rec):
        """
        生成3D目标检测任务的标签
        """
        # 创建空标签地图
        H, W = int(self.nx[1].item()), int(self.nx[0].item())
        cls_map = torch.zeros(H, W, dtype=torch.long)
        # x, y, z, w, l, h, sin(θ), cos(θ), vel
        reg_map = torch.zeros(9, H, W, dtype=torch.float32)
        reg_weight = torch.zeros(1, H, W, dtype=torch.float32)
        iou_map = torch.zeros(1, H, W, dtype=torch.float32)

        # 获取自车姿态
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse

        # 遍历每个标注
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            category = inst['category_name']

            # 检查是否为我们关注的类别
            main_category = category.split('.')[0]
            full_category = '.'.join(category.split('.')[:2])

            if full_category in self.category_map:
                class_id = self.category_map[full_category]
            elif main_category == 'vehicle':
                class_id = 1  # 默认为车辆类
            else:
                continue  # 跳过不关注的类别

            # 获取3D边界框
            # 在NuScenes数据集中，有些标注没有velocity字段
            # 构建Box对象时需传入默认的velocity元组 (np.nan, np.nan, np.nan)
            velocity = (np.nan, np.nan, np.nan)  # 默认值
            if 'velocity' in inst:
                velocity = tuple(inst['velocity'])
            box = Box(inst['translation'], inst['size'], Quaternion(
                inst['rotation']), velocity=velocity)

            # 转换到自车坐标系
            box.translate(trans)
            box.rotate(rot)

            # 边界框中心在BEV图中的位置
            center = box.center
            center_x, center_y = center[0], center[1]

            # 将中心点坐标转换为网格索引
            grid_x = int(
                (center_x - self.bx[0] + self.dx[0] / 2.) / self.dx[0])
            grid_y = int(
                (center_y - self.bx[1] + self.dx[1] / 2.) / self.dx[1])

            # 检查是否在范围内
            if 0 <= grid_x < W and 0 <= grid_y < H:
                # 更新类别地图
                cls_map[grid_y, grid_x] = class_id

                # 更新回归地图 - 回归目标为归一化的边界框参数
                # 中心点坐标 (x, y, z)
                reg_map[0, grid_y, grid_x] = center[0]
                reg_map[1, grid_y, grid_x] = center[1]
                reg_map[2, grid_y, grid_x] = center[2]

                # 尺寸 (w, l, h)
                reg_map[3, grid_y, grid_x] = box.wlh[0]  # width
                reg_map[4, grid_y, grid_x] = box.wlh[1]  # length
                reg_map[5, grid_y, grid_x] = box.wlh[2]  # height

                # 旋转角 - 使用sin和cos表示，避免角度的周期性问题
                yaw = box.orientation.yaw_pitch_roll[0]
                reg_map[6, grid_y, grid_x] = float(np.sin(yaw))
                reg_map[7, grid_y, grid_x] = float(np.cos(yaw))

                # 速度 - 只有在velocity不是NaN时才设置
                if box.velocity is not None and not np.isnan(box.velocity[0]):
                    # 使用float()显式转换numpy.floating类型为Python float
                    velocity = float(np.linalg.norm(box.velocity[:2]))
                    reg_map[8, grid_y, grid_x] = velocity

                # 更新权重和IoU地图
                reg_weight[0, grid_y, grid_x] = 1.0
                iou_map[0, grid_y, grid_x] = 1.0

                # 为周围网格设置较小的权重，形成高斯分布
                radius = 1
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        ny, nx = grid_y + dy, grid_x + dx
                        if 0 <= nx < W and 0 <= ny < H and (dx != 0 or dy != 0):
                            distance = np.sqrt(dx ** 2 + dy ** 2)
                            weight = np.exp(-distance ** 2 / 2)

                            # 如果当前位置没有更高优先级的目标
                            if cls_map[ny, nx] == 0:
                                cls_map[ny, nx] = class_id

                                # 复制中心目标的回归目标
                                for i in range(9):
                                    reg_map[i, ny, nx] = reg_map[i,
                                                                 grid_y, grid_x]

                                # 设置权重
                                reg_weight[0, ny, nx] = weight * 0.5
                                iou_map[0, ny, nx] = weight * 0.5

        return cls_map, reg_map, reg_weight, iou_map


class MultiModalDetectionData(Detection3DData):
    """
    多模态检测数据集，同时加载相机和LiDAR数据
    """

    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        super(MultiModalDetectionData, self).__init__(
            nusc, is_train, data_aug_conf, grid_conf)

        # 新增LiDAR处理配置
        self.nsweeps = 1  # 仅使用当前帧点云
        self.point_cloud_range = [
            grid_conf['xbound'][0], grid_conf['ybound'][0], grid_conf['zbound'][0],
            grid_conf['xbound'][1], grid_conf['ybound'][1], grid_conf['zbound'][1]
        ]

    def __getitem__(self, index):
        rec = self.ixes[index]

        # 获取图像数据
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)

        # 获取LiDAR BEV特征
        lidar_bev = self.get_lidar_bev(rec)

        # 获取检测目标
        targets = self.get_detection_targets(rec)

        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets

    def get_lidar_bev(self, rec):
        """
        将LiDAR点云转换为BEV特征图
        """
        # 获取点云数据
        pc = self.get_lidar_data(rec, self.nsweeps)

        # 过滤点云 - 只保留指定范围内的点
        pc_range = self.point_cloud_range
        mask = (pc[:, 0] >= pc_range[0]) & (pc[:, 0] < pc_range[3]) & \
               (pc[:, 1] >= pc_range[1]) & (pc[:, 1] < pc_range[4]) & \
               (pc[:, 2] >= pc_range[2]) & (pc[:, 2] < pc_range[5])
        pc = pc[mask]

        # 如果没有点，返回零张量
        if len(pc) == 0:
            return torch.zeros((18, int(self.nx[0]), int(self.nx[1])), dtype=torch.float32)

        # 将点云投影到BEV网格
        pc_bev = self.points_to_bev(pc)

        return pc_bev

    def points_to_bev(self, points):
        """
        将点云转换为BEV特征图
        """
        # 定义BEV网格参数
        xbound = self.grid_conf['xbound']
        ybound = self.grid_conf['ybound']
        zbound = self.grid_conf['zbound']
        nx = int(self.nx[0].item())
        ny = int(self.nx[1].item())

        # 初始化特征通道
        height_map = torch.zeros((1, nx, ny), dtype=torch.float32)  # 高度图
        intensity_map = torch.zeros((1, nx, ny), dtype=torch.float32)  # 强度图
        density_map = torch.zeros((1, nx, ny), dtype=torch.float32)  # 密度图

        # 计算点云在网格中的索引
        x_indices = ((points[:, 0] - xbound[0]) /
                     (xbound[1] - xbound[0]) * nx).type(torch.int32)
        y_indices = ((points[:, 1] - ybound[0]) /
                     (ybound[1] - ybound[0]) * ny).type(torch.int32)
        z = points[:, 2]

        # 过滤出有效索引
        mask = (x_indices >= 0) & (x_indices < nx) & (
            y_indices >= 0) & (y_indices < ny)
        x_indices = x_indices[mask]
        y_indices = y_indices[mask]
        z = z[mask]
        if points.shape[1] > 3:
            intensity = points[mask, 3]
        else:
            intensity = torch.ones_like(z)

        # 更新BEV特征图
        for i in range(len(x_indices)):
            x_idx, y_idx = x_indices[i], y_indices[i]
            # 使用torch.max替代Python内置的max
            height_map[0, x_idx, y_idx] = torch.max(
                torch.tensor([height_map[0, x_idx, y_idx], z[i]]))
            intensity_map[0, x_idx, y_idx] = torch.max(
                torch.tensor([intensity_map[0, x_idx, y_idx], intensity[i]]))
            density_map[0, x_idx, y_idx] += 1

        # 归一化密度图
        if density_map.max() > 0:
            density_map = density_map / density_map.max()

        # 创建多通道BEV特征
        bev_feats = []
        # 添加高度特征 - 减少高度区间数量到16个
        for h in torch.linspace(zbound[0], zbound[1], 16):
            height_slice = ((height_map > h - 0.2) &
                            (height_map <= h + 0.2)).float()
            bev_feats.append(height_slice)

        # 添加强度和密度特征
        bev_feats.append(intensity_map)
        bev_feats.append(density_map)

        # 合并所有特征
        bev_feats = torch.cat(bev_feats, dim=0)

        return bev_feats


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=True)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
        'detection3d': Detection3DData,
        'multimodal_detection': MultiModalDetectionData
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                     grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
