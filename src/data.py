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

# --- Define custom_collate at the top level --- #


def custom_collate(batch):
    # This basic collate handles the structure change where the last element is sample_token
    # and the second last is sparse_gt_list (list of lists of dicts)
    # It relies on default_collate for most elements but handles the sparse GT list specifically.

    # Check if batch is empty
    if not batch:
        return None  # Or raise an error, depending on desired behavior

    num_items = len(batch[0])
    elem_0 = batch[0]

    # Check consistency of item count in batch
    if not all(len(item) == num_items for item in batch):
        raise ValueError("Batch items have inconsistent lengths")

    # Separate potentially non-collated items (sparse GTs and tokens)
    # Assuming sparse GTs are second to last, tokens are last
    sparse_gts_batch = [item[num_items - 2] for item in batch]
    sample_tokens = [item[num_items - 1] for item in batch]

    # Collate the rest using default collate
    # Ensure there are items to collate
    if num_items > 2:
        other_items_batch = [item[:num_items - 2] for item in batch]
        collated_others = torch.utils.data.default_collate(other_items_batch)
        # Return collated items + the uncollated sparse GT list + uncollated tokens
        return (*collated_others, sparse_gts_batch, sample_tokens)
    elif num_items == 2:  # Only sparse GTs and tokens
        return (sparse_gts_batch, sample_tokens)
    else:  # Should not happen if __getitem__ returns at least 2 items
        return (sample_tokens,)

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
    # --- FIX: Ensure m, n are floats for calculations ---
    m, n = [(float(ss) - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    # Ensure x and y are floats before exponentiation
    x = x.astype(float)
    y = y.astype(float)
    # ---

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
        dense_target_maps, sparse_gt_list = self.get_detection_targets(rec)

        # 返回包含sample_token的元组
        return imgs, rots, trans, intrins, post_rots, post_trans, dense_target_maps, sparse_gt_list, sample_token

    def get_detection_targets(self, rec):
        """
        生成3D目标检测任务的标签 (优化版，使用高斯热图)
        同时返回用于评估的稀疏GT标注列表。
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

        # --- 新增：用于存储稀疏GT标注 ---
        sparse_gt_boxes = []
        # ---

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
            # --- 修改：处理不同nuScenes版本或自定义类别的可能格式 ---
            matched_cat = None
            for map_cat, cat_id in self.category_map.items():
                # 允许部分匹配 (例如 'vehicle.car' 匹配 'car') 或完全匹配
                if category.startswith(map_cat.split('.')[0]) or category == map_cat:
                    class_id = cat_id
                    matched_cat = map_cat  # 记录匹配的键，用于查找平均尺寸
                    break  # 找到第一个匹配即停止

            # --- 原始逻辑 (可能对某些category name格式不够鲁棒) ---
            # full_category = '.'.join(category.split('.')[:2])
            # if full_category in self.category_map:
            #     class_id = self.category_map[full_category]
            # ---

            if class_id < 0:
                continue  # 跳过不关注的类别

            # 获取3D边界框
            velocity = tuple(inst['velocity']) if 'velocity' in inst and len(
                inst['velocity']) == 3 else (np.nan, np.nan, np.nan)
            # --- 修复：确保传递 velocity 元组给 Box 构造函数 ---
            box = Box(inst['translation'], inst['size'],
                      Quaternion(inst['rotation']), velocity=velocity)
            # ---

            # 转换到自车坐标系
            box.translate(trans)
            box.rotate(rot)

            # --- 生成Dense Target Map ---
            # 边界框中心在BEV图中的位置
            center = box.center[:2]  # 只关心 x, y

            # 将中心点坐标转换为网格索引 (浮点数，用于高斯中心)
            # --- 修复: 使用 self.bx 和 self.dx (tensor) 进行计算 ---
            center_in_grid_x = (
                center[0] - self.bx[0].item()) / self.dx[0].item()
            center_in_grid_y = (
                center[1] - self.bx[1].item()) / self.dx[1].item()
            # ---

            # 获取整数网格索引用于赋值
            grid_x = int(center_in_grid_x)
            grid_y = int(center_in_grid_y)

            # 检查中心点是否在网格范围内
            if 0 <= grid_x < W and 0 <= grid_y < H:
                # 计算高斯半径 (使用BEV下的长宽)
                # box.wlh: [width, length, height]
                # det_size需要 (height, width) -> 在BEV下是 (length, width)
                # --- 修复: 使用 self.dx (tensor) 进行计算 ---
                det_size_bev = (box.wlh[1] / self.dx[1].item(),
                                box.wlh[0] / self.dx[0].item())
                # ---
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
                # 速度 (x, y 分量)
                # --- 修复: 提取x,y速度, 处理NaN ---
                vel_xy = box.velocity[:2]
                if vel_xy is None or np.any(np.isnan(vel_xy)):
                    vel_xy = np.array([0.0, 0.0])
                # 确保是2维
                if len(vel_xy) == 1:
                    vel_xy = np.append(vel_xy, 0.0)
                elif len(vel_xy) > 2:
                    vel_xy = vel_xy[:2]
                # 只存储x,y速度，假设回归图第9个通道是x速度，第10个是y速度 (需要调整reg_map大小)
                # 或者只存储速度大小 (如原代码)
                # --- 保持原逻辑：存储速度大小 ---
                velocity_norm = float(np.linalg.norm(vel_xy))
                reg_map[8, grid_y, grid_x] = velocity_norm
                # ---

                # 设置中心点的权重和IoU值
                reg_weight[0, grid_y, grid_x] = 1.0
                # Can be adjusted based on strategy
                iou_map[0, grid_y, grid_x] = 1.0
            # --- Dense Target 生成结束 ---

            # --- 生成Sparse Target ---
            # 使用转换到自车坐标系后的box信息
            vel_xy_sparse = box.velocity[:2]
            if vel_xy_sparse is None or np.any(np.isnan(vel_xy_sparse)):
                vel_xy_sparse = np.array([0.0, 0.0])
            if len(vel_xy_sparse) == 1:
                vel_xy_sparse = np.append(vel_xy_sparse, 0.0)
            elif len(vel_xy_sparse) > 2:
                vel_xy_sparse = vel_xy_sparse[:2]

            gt_dict = {
                'box_cls': torch.tensor(class_id, dtype=torch.long),
                'box_xyz': torch.from_numpy(box.center.astype(np.float32)),
                'box_wlh': torch.from_numpy(box.wlh.astype(np.float32)),
                'box_rot_sincos': torch.tensor([np.sin(box.orientation.yaw_pitch_roll[0]),
                                                np.cos(box.orientation.yaw_pitch_roll[0])], dtype=torch.float32),
                # Store [vx, vy]
                'box_vel': torch.from_numpy(vel_xy_sparse.astype(np.float32))
            }
            sparse_gt_boxes.append(gt_dict)
            # --- Sparse Target 生成结束 ---

        # 转换为Tensor
        cls_map_tensor = torch.from_numpy(cls_map)
        reg_map_tensor = torch.from_numpy(reg_map)
        reg_weight_tensor = torch.from_numpy(reg_weight)
        iou_map_tensor = torch.from_numpy(iou_map)

        # *** 重要：确定损失函数期望的分类目标格式 ***
        # 选项 A: 使用原始的多通道高斯热图 (如果损失函数支持，如Focal Loss)
        # cls_target_for_loss = cls_map_tensor
        # 选项 B: 使用单通道类别索引图 (如果损失函数是标准的CrossEntropyLoss)
        cls_index_map = torch.argmax(cls_map_tensor, dim=0).long()
        cls_target_for_loss = cls_index_map

        # 构建用于训练(损失计算)的稠密目标字典
        dense_maps = {'cls_targets': cls_target_for_loss,  # 使用选定的格式
                      'reg_targets': reg_map_tensor,
                      'reg_weights': reg_weight_tensor,
                      'iou_targets': iou_map_tensor
                      }

        # 返回稠密图和稀疏列表
        return dense_maps, sparse_gt_boxes


class MultiModalDetectionData(Detection3DData):
    """
    多模态检测数据集，同时加载相机和LiDAR数据
    """

    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        super(MultiModalDetectionData, self).__init__(
            nusc, is_train, data_aug_conf, grid_conf)

        # 新增LiDAR处理配置
        self.nsweeps = 1  # 仅使用当前帧点云
        # --- 修复: 确保 point_cloud_range 是列表 ---
        self.point_cloud_range = list([
            grid_conf['xbound'][0], grid_conf['ybound'][0], grid_conf['zbound'][0],
            grid_conf['xbound'][1], grid_conf['ybound'][1], grid_conf['zbound'][1]
        ])
        # ---

    def __getitem__(self, index):
        rec = self.ixes[index]
        # 获取样本token
        sample_token = rec['token']  # 新增获取sample_token

        # 获取图像数据
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)

        # 获取LiDAR BEV特征
        lidar_bev = self.get_lidar_bev(rec)

        # --- 修改：接收两个返回值 ---
        targets, sparse_gt_list = self.get_detection_targets(rec)
        # ---

        # --- 修改：返回稀疏GT列表和sample_token ---
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets, sparse_gt_list, sample_token
        # ---

    def get_lidar_bev(self, rec):
        """
        将LiDAR点云转换为BEV特征图
        """
        # 获取点云数据
        # --- 修改: 使用父类的 get_lidar_data ---
        # Note: get_lidar_data returns only x,y,z. Need intensity if available.
        # Let's reload lidar data here to potentially get intensity.
        lidar_data_full = get_lidar_data(
            self.nusc, rec, nsweeps=self.nsweeps, min_distance=2.2)  # Get (x,y,z,intensity,...)
        pc = torch.Tensor(lidar_data_full)  # Convert to tensor
        # ---

        # 过滤点云 - 只保留指定范围内的点
        pc_range = self.point_cloud_range
        mask = (pc[:, 0] >= pc_range[0]) & (pc[:, 0] < pc_range[3]) & \
               (pc[:, 1] >= pc_range[1]) & (pc[:, 1] < pc_range[4]) & \
               (pc[:, 2] >= pc_range[2]) & (pc[:, 2] < pc_range[5])
        pc = pc[mask]

        # 如果没有点，返回零张量
        if len(pc) == 0:
            # --- 修复: 使用 self.nx (tensor) 获取尺寸 ---
            return torch.zeros((18, int(self.nx[1].item()), int(self.nx[0].item())), dtype=torch.float32)
            # ---

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
        # Corrected shape order (H, W) -> (ny, nx)
        height_map = torch.zeros((1, ny, nx), dtype=torch.float32)
        intensity_map = torch.zeros(
            (1, ny, nx), dtype=torch.float32)  # Corrected shape order
        # Corrected shape order
        density_map = torch.zeros((1, ny, nx), dtype=torch.float32)

        # 计算点云在网格中的索引
        # Note: Original LSS uses different indexing convention (x for height, y for width)
        # Assuming standard image convention (H=y, W=x) for consistency with heatmap generation
        # Adjust indexing if model expects LSS convention
        x_indices = ((points[:, 0] - xbound[0]) /
                     self.dx[0].item()).type(torch.long)
        y_indices = ((points[:, 1] - ybound[0]) /
                     self.dx[1].item()).type(torch.long)
        z = points[:, 2]

        # 过滤出有效索引
        mask = (x_indices >= 0) & (x_indices < nx) & \
               (y_indices >= 0) & (y_indices < ny)
        x_indices = x_indices[mask]
        y_indices = y_indices[mask]
        z = z[mask]
        if points.shape[1] > 3:
            intensity = points[mask, 3]
        else:
            intensity = torch.ones_like(z)  # Use 1 if intensity missing

        # 更新BEV特征图 (Use y_indices for height dim, x_indices for width dim)
        # Use scatter_max_ for efficiency if possible, but simple loop is clearer
        for i in range(len(x_indices)):
            x_idx, y_idx = x_indices[i], y_indices[i]
            # Max height in the cell
            height_map[0, y_idx, x_idx] = torch.max(
                height_map[0, y_idx, x_idx], z[i])
            # Max intensity in the cell
            intensity_map[0, y_idx, x_idx] = torch.max(
                intensity_map[0, y_idx, x_idx], intensity[i])
            # Count points per cell
            density_map[0, y_idx, x_idx] += 1

        # Normalize density map (using log(N+1) helps stabilize)
        density_map = torch.log1p(density_map)

        # 创建多通道BEV特征 (Example: 16 height bins + intensity + density = 18)
        bev_feats = []
        num_height_bins = 16  # Configurable
        # Height Bins
        height_bin_indices = torch.floor((torch.clamp(
            height_map, zbound[0], zbound[1]) - zbound[0]) / ((zbound[1]-zbound[0])/num_height_bins)).long()
        height_one_hot = torch.zeros(
            1, num_height_bins, ny, nx, dtype=torch.float32)
        # Handle case where height_map might be empty (all zeros)
        if height_bin_indices.numel() > 0:
            height_one_hot.scatter_(
                1, height_bin_indices.unsqueeze(1), 1)  # Add channel dim
        # Remove the batch dim added for scatter
        bev_feats.append(height_one_hot.squeeze(0))

        # Add intensity and density features
        bev_feats.append(intensity_map)
        bev_feats.append(density_map)

        # 合并所有特征
        # Concatenate along channel dimension
        bev_feats = torch.cat(bev_feats, dim=1)

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

    # Create DataLoaders using the top-level custom_collate
    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init,
                                              collate_fn=custom_collate)  # Use top-level function
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers,
                                            collate_fn=custom_collate)  # Use top-level function

    return trainloader, valloader
