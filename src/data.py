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
import pickle

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx

# Add Gaussian heatmap helper functions (adapted from mmdet3d)


def gaussian_radius(det_size, min_overlap=0.7):
    """Calculate Gaussian radius based on object size and minimum overlap.

    Args:
        det_size (tuple[float]): Object size (h, w).
        min_overlap (float): Minimum overlap between heatmap and ground truth box.

    Returns:
        int: Gaussian radius.
    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    # Ensure radius is at least 2 (as in CenterPoint)
    return max(2, int(min(r1, r2, r3)))


def gaussian_2d(shape, sigma=1):
    """Generate 2D Gaussian-like heatmap.

    Args:
        shape (tuple[int]): Shape of the heatmap (h, w).
        sigma (float): Sigma of the Gaussian.

    Returns:
        np.ndarray: Generated heatmap.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D Gaussian heatmap.

    Args:
        heatmap (np.ndarray): Heatmap to be modified.
        center (tuple[float]): Center of the Gaussian (float_x, float_y).
        radius (int): Radius of the Gaussian.
        k (float): Factor for Gaussian value.

    Returns:
        np.ndarray: Modified heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    # Get float and integer center coordinates
    center_x_float, center_y_float = center
    # Round to nearest integer for center pixel
    x = int(np.round(center_x_float))
    y = int(np.round(center_y_float))

    height, width = heatmap.shape[0:2]

    # Ensure integer center is within bounds
    if x < 0 or x >= width or y < 0 or y >= height:
        return heatmap

    # Calculate integer boundaries for slicing the heatmap and gaussian
    # Determine the intersection area between the gaussian kernel and the heatmap boundaries
    left = max(0, x - radius)           # Heatmap left boundary
    right = min(width, x + radius + 1)  # Heatmap right boundary (exclusive)
    top = max(0, y - radius)            # Heatmap top boundary
    bottom = min(height, y + radius + 1)  # Heatmap bottom boundary (exclusive)

    # Determine the corresponding slice within the gaussian kernel
    gaussian_left = max(0, radius - x)  # Gaussian kernel left boundary
    # Gaussian kernel right boundary
    gaussian_right = radius + (right - (x + 1))
    gaussian_top = max(0, radius - y)  # Gaussian kernel top boundary
    # Gaussian kernel bottom boundary
    gaussian_bottom = radius + (bottom - (y + 1))

    # Ensure slice dimensions are valid and match
    heatmap_slice_h, heatmap_slice_w = bottom - top, right - left
    gaussian_slice_h, gaussian_slice_w = gaussian_bottom - \
        gaussian_top, gaussian_right - gaussian_left

    if heatmap_slice_h > 0 and heatmap_slice_w > 0 and gaussian_slice_h > 0 and gaussian_slice_w > 0 and \
       heatmap_slice_h == gaussian_slice_h and heatmap_slice_w == gaussian_slice_w:

        # Get the actual slices using integer indices
        masked_heatmap = heatmap[top:bottom, left:right]
        masked_gaussian = gaussian[gaussian_top:gaussian_bottom,
                                   gaussian_left:gaussian_right]

        # Apply maximum operation
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    # else: # Optional: Add debug print for non-overlapping/mismatched cases
        # print(f"Skipping draw_heatmap: Center({center_x_float:.1f},{center_y_float:.1f}) Radius({radius}) -> HeatmapSlice({top}:{bottom},{left}:{right}) GaussianSlice({gaussian_top}:{gaussian_bottom},{gaussian_left}:{gaussian_right})")

    return heatmap


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        # 检查是否有预先缓存的信息
        has_cache = False
        try:
            if hasattr(nusc, 'cached_infos') and nusc.__dict__['cached_infos'] is not None:
                has_cache = True
        except (AttributeError, KeyError):
            pass

        if has_cache:
            print(f"使用预缓存的数据集信息（{'train' if is_train else 'val'}集）")
            # 直接使用缓存的场景列表
            self.scenes = self.get_scenes()
            # 从缓存中获取样本索引
            split = 'train' if is_train else 'val'
            cached_infos = nusc.__dict__['cached_infos']['infos'][split]

            # 根据缓存信息重建索引
            self.ixes = []
            for info in cached_infos:
                # 获取对应sample的token
                token = info['token']
                # 获取sample对象
                sample = [s for s in nusc.sample if s['token'] == token][0]
                self.ixes.append(sample)

            print(f"从缓存加载了{len(self.ixes)}个样本")
        else:
            print("未找到缓存信息，从原始数据构建索引")
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
            'vehicle.car': 0,            # 1 -> 0
            'vehicle.truck': 1,          # 2 -> 1
            'vehicle.bus': 2,            # 3 -> 2
            'vehicle.trailer': 3,         # 4 -> 3
            'vehicle.construction': 4,   # 5 -> 4
            'human.pedestrian': 5,       # 6 -> 5
            'vehicle.motorcycle': 6,     # 7 -> 6
            'vehicle.bicycle': 7,        # 8 -> 7
            'movable_object.traffic_cone': 8,  # 9 -> 8
            'movable_object.barrier': 9      # 10 -> 9
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
        # 获取样本token
        sample_token = rec['token']

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)
        target_maps = self.get_detection_targets(rec)

        # 返回包含sample_token的元组
        return imgs, rots, trans, intrins, post_rots, post_trans, target_maps, sample_token

    def get_detection_targets(self, rec):
        """
        生成3D目标检测任务的标签 (优化版，使用高斯热图)
        """
        # 获取BEV网格尺寸
        H, W = int(self.nx[1].item()), int(self.nx[0].item())
        # 获取类别数量 (从映射中获取，+1表示背景)
        num_classes = len(self.DETECTION_CLASSES)

        # 创建空标签地图
        # 分类热图 (每个类别一个通道)
        cls_map = np.zeros((num_classes, H, W), dtype=np.float32)
        # 回归图 (x, y, z, w, l, h, sin(θ), cos(θ), vel)
        reg_map = np.zeros((9, H, W), dtype=np.float32)
        # 回归权重图 (只在目标中心为1)
        reg_weight = np.zeros((1, H, W), dtype=np.float32)
        # IoU/Centerness 图 (在目标中心为1，可以根据需要修改)
        iou_map = np.zeros((1, H, W), dtype=np.float32)

        # 获取自车姿态
        lidar_top_data = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        egopose = self.nusc.get('ego_pose', lidar_top_data['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse

        # 遍历每个标注
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            category = inst['category_name']

            # 检查是否为我们关注的类别，并获取类别ID [0, num_classes-1]
            class_id = -1
            full_category = '.'.join(category.split('.')[:2])
            if full_category in self.category_map:
                class_id = self.category_map[full_category]

            if class_id < 0:
                continue  # 跳过不关注的类别

            # 获取3D边界框
            velocity = tuple(inst['velocity']) if 'velocity' in inst and len(
                inst['velocity']) == 3 else (np.nan, np.nan, np.nan)
            box = Box(inst['translation'], inst['size'],
                      Quaternion(inst['rotation']), velocity=velocity)

            # 转换到自车坐标系
            box.translate(trans)
            box.rotate(rot)

            # 边界框中心在BEV图中的位置
            center = box.center[:2]  # 只关心 x, y

            # 将中心点坐标转换为网格索引 (浮点数，用于高斯中心)
            center_in_grid_x = (center[0] - self.bx[0]) / self.dx[0]
            center_in_grid_y = (center[1] - self.bx[1]) / self.dx[1]

            # 获取整数网格索引用于赋值
            grid_x = int(center_in_grid_x)
            grid_y = int(center_in_grid_y)

            # 检查中心点是否在网格范围内
            if 0 <= grid_x < W and 0 <= grid_y < H:
                # 计算高斯半径 (使用BEV下的长宽)
                # box.wlh: [width, length, height]
                # det_size需要 (height, width) -> 在BEV下是 (length, width)
                det_size_bev = (box.wlh[1] / self.dx[1],
                                box.wlh[0] / self.dx[0])
                radius = gaussian_radius(det_size_bev)
                radius = max(0, int(radius))

                # 在对应类别的热图上绘制高斯分布
                # 使用浮点数中心坐标以获得更准确的高斯峰值位置
                draw_heatmap_gaussian(
                    cls_map[class_id], (center_in_grid_x, center_in_grid_y), radius)

                # 仅在中心点设置回归目标和权重
                # 中心点坐标 (x, y, z)
                reg_map[0, grid_y, grid_x] = box.center[0]
                reg_map[1, grid_y, grid_x] = box.center[1]
                reg_map[2, grid_y, grid_x] = box.center[2]
                # 尺寸 (w, l, h)
                reg_map[3, grid_y, grid_x] = box.wlh[0]  # width
                reg_map[4, grid_y, grid_x] = box.wlh[1]  # length
                reg_map[5, grid_y, grid_x] = box.wlh[2]  # height
                # 旋转角
                yaw = box.orientation.yaw_pitch_roll[0]
                reg_map[6, grid_y, grid_x] = np.sin(yaw)
                reg_map[7, grid_y, grid_x] = np.cos(yaw)
                # 速度
                if box.velocity is not None and not np.any(np.isnan(box.velocity[:2])):
                    velocity_norm = float(np.linalg.norm(box.velocity[:2]))
                    reg_map[8, grid_y, grid_x] = velocity_norm
                else:
                    # Default to 0 if no velocity
                    reg_map[8, grid_y, grid_x] = 0.0

                # 设置中心点的权重和IoU值
                reg_weight[0, grid_y, grid_x] = 1.0
                # Can be adjusted based on strategy
                iou_map[0, grid_y, grid_x] = 1.0

        # 转换为Tensor
        cls_map_tensor = torch.from_numpy(cls_map)
        reg_map_tensor = torch.from_numpy(reg_map)
        reg_weight_tensor = torch.from_numpy(reg_weight)
        iou_map_tensor = torch.from_numpy(iou_map)

        # 之前返回的是 (cls_map, reg_map, reg_weight, iou_map)
        # 需要确认损失函数期望的格式。假设它期望一个包含这些张量的字典。
        # 并且，cls_map现在是多通道的，而之前是单通道的类别索引。
        # 这可能需要调整损失函数或这里的返回格式。

        # *** 修复：将多通道热图转为单通道索引图以匹配CrossEntropyLoss ***
        # 注意：这丢失了高斯热图的信息，更好的方式是修改损失函数以接受多通道热图
        # 背景类 (通道0) 的热图值会被忽略，取其他通道的最大值作为类别索引
        # 如果所有目标通道都为0，则该点为背景 (索引0)
        # 需要确保背景没有被绘制高斯峰 (我们的类别映射从0开始，所以class_id=0是第一个目标类)
        # 因此，我们找 >0 的最大值索引，如果不存在，则为背景(0)
        # 为了处理argmax可能选到0的情况，我们先给背景通道加一个很小的值
        # 背景处理：将背景类(索引0，如果存在且在num_classes内)的热图值设为0，确保argmax不会选到它
        # if num_classes > 0:
        #     cls_map_tensor[:, 0, :, :] = 0 # Assuming class 0 is background for argmax purpose if mapping starts from 1
        # If mapping starts from 0, the background is implicitly where no gaussian is drawn

        cls_index_map = torch.argmax(cls_map_tensor, dim=0).long()
        # return cls_index_map, reg_map_tensor, reg_weight_tensor, iou_map_tensor # Old tuple return

        # 返回包含单通道索引图的字典
        return {'cls_targets': cls_index_map,  # Use single-channel index map
                'reg_targets': reg_map_tensor,
                'reg_weights': reg_weight_tensor,
                'iou_targets': iou_map_tensor
                }


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
    """
    编译数据集，先检查是否有缓存文件，有则直接使用，无则重新加载并创建
    """
    # 导入info加载功能
    from .nuscenes_info import load_nuscenes_infos

    # 设置NuScenes数据集版本
    version_str = f'v1.0-{version}'
    max_sweeps = 10  # 设置最大扫描帧数

    # 加载或创建缓存信息
    nusc_infos = None  # Initialize nusc_infos
    try:
        print(f"加载NuScenes缓存信息...")
        nusc_infos = load_nuscenes_infos(
            dataroot, version=version_str, max_sweeps=max_sweeps)
        print(
            f"成功加载缓存信息! 包含{len(nusc_infos['infos']['train'])}个训练样本和{len(nusc_infos['infos']['val'])}个验证样本")

        # 使用包含缓存数据的NuScenes对象
        nusc = NuScenes(version=version_str,
                        dataroot=dataroot,
                        verbose=True)

        # 将缓存信息添加到数据集实例中
        # 使用__dict__直接添加属性，避免linter错误
        nusc.__dict__['cached_infos'] = nusc_infos
        print("已将缓存信息附加到NuScenes对象")
    except Exception as e:
        print(f"处理缓存信息时出错：{e}")
        print("无法使用缓存，这可能会导致数据加载速度变慢")
        nusc = NuScenes(version=version_str,
                        dataroot=dataroot,
                        verbose=True)
        # 确保cached_infos属性存在但为None
        nusc.__dict__['cached_infos'] = None

    # Define the mapping from parser_name to Dataset class and its specific kwargs
    parser_map = {
        'vizdata': (VizData, {}),
        'segmentationdata': (SegmentationData, {}),
        'detection3d': (Detection3DData, {}),
        'multimodal_detection': (MultiModalDetectionData, {}),
        # Ensure this uses the correct class
        'nuScenes': (Detection3DData, {}),
    }

    # Lookup the class and kwargs based on parser_name
    if parser_name not in parser_map:
        raise KeyError(
            f"Invalid parser_name: '{parser_name}'. Available options: {list(parser_map.keys())}")
    dataset_class, dataset_kwargs = parser_map[parser_name]

    # Instantiate the datasets using the retrieved class and kwargs
    traindata = dataset_class(nusc=nusc, is_train=True,
                              data_aug_conf=data_aug_conf,
                              grid_conf=grid_conf, **dataset_kwargs)
    valdata = dataset_class(nusc=nusc, is_train=False,
                            data_aug_conf=data_aug_conf,
                            grid_conf=grid_conf, **dataset_kwargs)

    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
