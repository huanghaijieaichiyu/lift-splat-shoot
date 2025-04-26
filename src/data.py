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
import torch.utils.data  # 显式导入以备后用

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx

# --- Define custom_collate at the top level --- #

# === MODIFICATION START: Revise custom_collate for CenterPoint targets ===


def custom_collate(batch):
    """
    自定义的 collate 函数，用于处理来自 Detection3DData 或 MultiModalDetectionData 的批次。
    它使用 default_collate 对初始项目 (0-5) 进行整理，
    并正确地填充和堆叠 targets_dict (索引 6) 中的张量。
    sample_tokens (索引 7) 保持为列表。
    """
    # 检查批次是否为空
    if not batch:
        return None  # 或引发错误

    elem = batch[0]
    num_items = len(elem)

    # 检查一致性
    if not all(len(item) == num_items for item in batch):
        raise ValueError("批次项目的长度不一致")

    if num_items < 8:  # 期望至少 8 个项目
        raise ValueError(
            f"每个批次元素期望至少 8 个项目，但得到 {num_items}")

    # --- 新的 Collation 逻辑 ---
    B = len(batch)
    list_of_targets_dicts = [item[6] for item in batch]
    sample_tokens = [item[7] for item in batch]

    # --- Add Check ---
    if not all(isinstance(d, dict) for d in list_of_targets_dicts):
        print("错误：list_of_targets_dicts 包含非字典项！")
        for i, item in enumerate(list_of_targets_dicts):
            if not isinstance(item, dict):
                print(f" 索引 {i} 处的项类型：{type(item)}，值：{item}")
        raise TypeError("在整理过程中检测到 list_of_targets_dicts 中的非字典项。")
    # --- End Check ---

    # 1. 整理初始元素 (0 到 5)
    items_to_collate = [item[:6] for item in batch]
    try:
        collated_items_0_to_5 = torch.utils.data.default_collate(
            items_to_collate)
    except Exception as e:
        print(f"对项目 0-5 进行 default_collate 时出错: {e}")
        raise

    # 2. 整理 targets_dict
    collated_targets_dict = {}
    # 获取第一个有效字典的键（处理可能的空批次或错误项）
    first_valid_dict = next(
        (d for d in list_of_targets_dicts if isinstance(d, dict)), None)
    if first_valid_dict is None:
        raise ValueError("批次中未找到有效的 targets_dict。")
    target_keys = first_valid_dict.keys()

    # 检查所有样本是否具有相同的目标键
    for i in range(B):
        if isinstance(list_of_targets_dicts[i], dict) and list_of_targets_dicts[i].keys() != target_keys:
            raise ValueError(
                f"目标字典键在批次中不一致。样本 0: {target_keys}, 样本 {i}: {list_of_targets_dicts[i].keys()}")

    # 定义需要填充的稀疏键和直接堆叠的密集键
    sparse_keys = ['target_indices', 'target_offset', 'target_z_coord',
                   'target_dimension', 'target_rotation', 'target_velocity', 'reg_mask']
    dense_keys = ['target_heatmap', 'target_mask']
    scalar_keys = ['num_objs']  # 标量键直接转换为张量

    # 确定最大对象数
    max_objs = 0
    num_objs_list = []
    try:
        for targets in list_of_targets_dicts:
            # 确保 targets 是字典并且包含 num_objs
            if isinstance(targets, dict) and 'num_objs' in targets:
                num = targets['num_objs'].item()  # 获取标量值
                num_objs_list.append(num)
                max_objs = max(max_objs, num)
            else:
                # 如果某个样本的目标无效，则添加 0 并继续
                num_objs_list.append(0)
                print(f"警告：批次中发现无效的目标字典或缺少 'num_objs' 键。类型：{type(targets)}")
        # CenterPoint 通常限制最大对象数
        max_objs = min(max_objs, 500)  # 使用 CenterPoint 中常见的限制
    except KeyError:
        # 这个错误不应该发生，因为我们在上面检查了 'num_objs'
        raise KeyError("'num_objs' 键在 targets_dict 中是必需的，用于确定最大对象数和填充。")
    except Exception as e:
        print(f"获取 'num_objs' 时出错: {e}")
        raise

    # 处理每个目标键
    for key in target_keys:
        # 安全地获取张量列表，跳过无效的目标字典
        tensor_list = []
        valid_indices = []  # 跟踪哪些样本具有此键的有效张量
        for idx, targets in enumerate(list_of_targets_dicts):
            if isinstance(targets, dict) and key in targets:
                tensor_list.append(targets[key])
                valid_indices.append(idx)
            # else: # 可选：如果键丢失，则添加警告
            #     print(f"警告：样本 {idx} 的目标字典中缺少键 '{key}'")

        if not tensor_list:  # 如果没有样本具有此键，则跳过
            print(f"警告：在批次中没有样本找到键 '{key}'，跳过此键的整理。")
            continue

        if key in dense_keys:
            # === Pre-stack check ===
            valid_types = True
            for i, t in enumerate(tensor_list):
                if not isinstance(t, torch.Tensor):
                    print(f"!!! PRE-STACK CHECK FAILED for key '{key}' !!!")
                    print(
                        f"  Item {i} (from original batch index {valid_indices[i]}) is NOT a Tensor.")
                    print(f"  Type: {type(t)}")
                    print(f"  Value: {t}")
                    valid_types = False
            if not valid_types:
                # Maybe raise a different error here to pinpoint the pre-stack issue
                raise TypeError(
                    f"Non-tensor found in list for key '{key}' before torch.stack.")
            # === End Pre-stack check ===

            # 直接堆叠密集张量
            try:
                collated_targets_dict[key] = torch.stack(tensor_list)
            # === MODIFICATION START: Enhanced Debugging for Stacking Error ===
            except Exception as e:  # 捕获更广泛的异常以查看确切类型
                print(f"--- 在密集键 '{key}' 的 torch.stack 期间出错 ---")
                print(f"错误类型: {type(e)}")
                print(f"错误消息: {e}")
                print(f"列表中的张量数量: {len(tensor_list)}")
                for i, t in enumerate(tensor_list):
                    print(f" 张量 {i} (来自原始批次索引 {valid_indices[i]}):")
                    if isinstance(t, torch.Tensor):
                        print(f"  形状: {t.shape}")
                        print(f"  数据类型: {t.dtype}")
                        print(f"  设备: {t.device}")
                        # === ADDITION: Check tensor properties ===
                        print(f"  是否连续: {t.is_contiguous()}")
                        print(f"  Strides: {t.stride()}")
                        # === END ADDITION ===
                    else:
                        print(f"  类型: {type(t)}")  # 检查它是否真的是张量
                        print(f"  值: {t}")
                print("--- 错误信息结束 ---")
                raise  # 重新引发原始异常
            # === MODIFICATION END ===
        elif key in sparse_keys:
            # 填充和堆叠稀疏张量
            first_tensor = tensor_list[0]  # 获取第一个有效张量
            element_shape = first_tensor.shape[1:]  # 获取除对象数量之外的维度
            padded_shape = (B, max_objs) + element_shape  # 填充形状基于完整批次大小 B
            # 根据第一个张量的数据类型和设备创建填充张量
            padded_tensor = torch.zeros(
                padded_shape, dtype=first_tensor.dtype, device=first_tensor.device)

            for i, original_idx in enumerate(valid_indices):  # 仅迭代具有此键的有效样本
                # 使用来自完整列表的正确 num_objs
                num_obj_in_sample = num_objs_list[original_idx]
                if num_obj_in_sample > 0:
                    # 确保我们不会尝试填充超过 max_objs 的对象
                    num_to_copy = min(num_obj_in_sample, max_objs)
                    current_tensor = tensor_list[i]  # 从有效张量列表中获取当前张量
                    # 确保源张量足够大
                    if current_tensor.shape[0] >= num_to_copy:
                        # 使用原始批次索引 original_idx 进行填充
                        padded_tensor[original_idx,
                                      :num_to_copy] = current_tensor[:num_to_copy]
                    else:  # 源张量小于 num_to_copy（可能由于裁剪或其他问题）
                        print(
                            f"警告：样本 {original_idx}, 键 '{key}', 张量形状 {current_tensor.shape[0]} < num_to_copy {num_to_copy} (num_objs: {num_obj_in_sample}). 仅复制 {current_tensor.shape[0]} 个元素。")
                        if current_tensor.shape[0] > 0:
                            padded_tensor[original_idx,
                                          :current_tensor.shape[0]] = current_tensor

            collated_targets_dict[key] = padded_tensor
        elif key in scalar_keys:
            # 将标量列表转换为张量（使用完整的 num_objs_list）
            collated_targets_dict[key] = torch.tensor(
                num_objs_list, dtype=torch.long)
        else:
            print(f"警告：跳过未知的目标键 '{key}' 的整理。")

    # 3. 返回整理好的项目
    return (*collated_items_0_to_5, collated_targets_dict, sample_tokens)
# === MODIFICATION END ===

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
    # Ensure m, n are calculated from numerical types
    try:
        h_dim = float(shape[0])
        w_dim = float(shape[1])
    except (TypeError, IndexError) as e:
        raise ValueError(
            f"Invalid shape tuple provided to gaussian_2d: {shape}. Error: {e}")

    m = (h_dim - 1.) / 2.
    n = (w_dim - 1.) / 2.

    # Use meshgrid for safer coordinate generation
    y_range = np.arange(-m, m + 1, dtype=float)
    x_range = np.arange(-n, n + 1, dtype=float)
    x, y = np.meshgrid(x_range, y_range)

    # === MODIFICATION START: Explicitly set dtype for h ===
    denominator = (2 * sigma * sigma)
    if denominator == 0:
        # Handle sigma=0 case to avoid division by zero
        # Return an array of zeros or a delta function depending on desired behavior
        # Here, returning zeros except at the center might be appropriate
        h = np.zeros((len(y_range), len(x_range)), dtype=np.float64)
        center_y_idx = len(y_range) // 2
        center_x_idx = len(x_range) // 2
        if 0 <= center_y_idx < h.shape[0] and 0 <= center_x_idx < h.shape[1]:
            h[center_y_idx, center_x_idx] = 1.0
    else:
        # Explicitly cast numerator to float64 to avoid potential type issues hinted by linter
        numerator = -(x * x + y * y).astype(np.float64)
        h = np.exp(numerator / denominator, dtype=np.float64)
    # === MODIFICATION END ===

    # Ensure h is float before comparison
    h = h.astype(np.float64)
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
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
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
            resize_dims = (int(W*resize), int(H*resize))
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
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
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
        pts = get_lidar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
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
            box = Box(inst['translation'], inst['size'],
                      Quaternion(inst['rotation']), velocity=velocity)
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
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
        # --- 修改：直接获取 CenterPoint 格式的目标字典 ---
        targets_dict = self.get_detection_targets(rec)
        # 移除旧的 sparse_gt_list，因为它现在包含在 targets_dict 或用于评估
        # sparse_gt_list = targets_dict.pop('sparse_gt_list', []) # Example if needed elsewhere
        # ---

        # 返回包含sample_token的元组 (移除 dense_target_maps, sparse_gt_list)
        return imgs, rots, trans, intrins, post_rots, post_trans, targets_dict, sample_token

    def get_detection_targets(self, rec):
        """
        生成 CenterPoint 风格的密集目标图和稀疏回归目标。
        Returns:
            targets_dict (dict): Dictionary containing targets for CenterPointLoss.
        """
        # 获取BEV网格尺寸
        H, W = int(self.nx[1].item()), int(self.nx[0].item())
        num_classes = len(self.DETECTION_CLASSES)
        # --- CenterPoint Target Generation ---
        max_objs = 500  # Define maximum objects per sample

        # Initialize dense maps
        target_heatmap = np.zeros((num_classes, H, W), dtype=np.float32)
        # Mask for positive locations
        target_mask = np.zeros((H, W), dtype=np.bool_)
        # Note: Regression maps are not stored densely, only gathered values

        # Initialize sparse target arrays (size max_objs)
        target_indices = np.zeros((max_objs), dtype=np.int64)
        target_offset = np.zeros((max_objs, 2), dtype=np.float32)
        target_z_coord = np.zeros((max_objs, 1), dtype=np.float32)
        target_dimension = np.zeros(
            (max_objs, 3), dtype=np.float32)  # Will store log(dim)
        target_rotation = np.zeros((max_objs, 2), dtype=np.float32)  # sin, cos
        target_velocity = np.zeros((max_objs, 2), dtype=np.float32)  # vx, vy
        obj_count = 0  # Counter for valid objects found
        # ---

        # --- 新增：用于指示有效对象的掩码 ---
        reg_mask = np.zeros((max_objs), dtype=np.bool_)
        # --- 结束 ---

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

            # --- 生成 CenterPoint Targets ---
            # 边界框中心在BEV图中的位置
            center = box.center[:2]  # 只关心 x, y

            # 将中心点坐标转换为网格索引 (浮点数，用于高斯中心)
            center_in_grid_x = (
                center[0] - self.bx[0].item()) / self.dx[0].item()
            center_in_grid_y = (
                center[1] - self.bx[1].item()) / self.dx[1].item()

            # 获取整数网格索引用于赋值
            grid_x = int(np.round(center_in_grid_x))
            grid_y = int(np.round(center_in_grid_y))

            # 检查中心点是否在网格范围内
            if 0 <= grid_x < W and 0 <= grid_y < H and obj_count < max_objs:
                # 计算高斯半径
                det_size_bev = (
                    box.wlh[1] / self.dx[1].item(), box.wlh[0] / self.dx[0].item())
                radius = gaussian_radius(det_size_bev)
                radius = max(0, int(radius))

                # 在对应类别的热图上绘制高斯分布
                draw_heatmap_gaussian(
                    target_heatmap[class_id], (center_in_grid_x, center_in_grid_y), radius)

                # --- 填充稀疏回归目标和索引 ---
                flat_idx = grid_y * W + grid_x
                target_indices[obj_count] = flat_idx
                # Mark this location as positive
                target_mask[grid_y, grid_x] = True

                # Offset (precise location within the grid cell)
                target_offset[obj_count, 0] = center_in_grid_x - grid_x
                target_offset[obj_count, 1] = center_in_grid_y - grid_y

                # Z-coordinate
                target_z_coord[obj_count, 0] = box.center[2]

                # Dimension (log scale)
                # log(width) Add epsilon for stability before log
                target_dimension[obj_count, 0] = np.log(box.wlh[0] + 1e-6)
                target_dimension[obj_count, 1] = np.log(
                    box.wlh[1] + 1e-6)  # log(length)
                target_dimension[obj_count, 2] = np.log(
                    box.wlh[2] + 1e-6)  # log(height)

                # Rotation (sin, cos)
                yaw = box.orientation.yaw_pitch_roll[0]
                target_rotation[obj_count, 0] = np.sin(yaw)
                target_rotation[obj_count, 1] = np.cos(yaw)

                # Velocity (vx, vy)
                vel_xy = box.velocity[:2]
                if vel_xy is None or np.any(np.isnan(vel_xy)):
                    vel_xy = np.array([0.0, 0.0])
                if len(vel_xy) == 1:  # Ensure 2D
                    vel_xy = np.append(vel_xy, 0.0)
                elif len(vel_xy) > 2:
                    vel_xy = vel_xy[:2]
                target_velocity[obj_count, 0] = vel_xy[0]
                target_velocity[obj_count, 1] = vel_xy[1]

                # --- 新增：标记此对象为有效 ---
                reg_mask[obj_count] = True
                # --- 结束 ---

                obj_count += 1  # Increment object counter
                # --- 稀疏目标填充结束 ---

            # --- 移除旧的 Dense 回归图填充 ---
            # if 0 <= grid_x < W and 0 <= grid_y < H:
                # ... (移除 reg_map, reg_weight, iou_map 填充)
            # ---

            # --- 生成Sparse Target (可选，如果评估需要) ---
            # vel_xy_sparse = ...
            # gt_dict = { ... }
            # sparse_gt_boxes_for_eval.append(gt_dict)
            # ---

        # --- 构建最终的 targets_dict ---
        targets_dict = {
            # Dense heatmap
            'target_heatmap': torch.from_numpy(target_heatmap),
            # Add channel dim [1, H, W]
            'target_mask': torch.from_numpy(target_mask).unsqueeze(0),

            # --- 修改：返回完整的、填充过的稀疏目标 ---
            'target_indices': torch.from_numpy(target_indices),
            'target_offset': torch.from_numpy(target_offset),
            'target_z_coord': torch.from_numpy(target_z_coord),
            'target_dimension': torch.from_numpy(target_dimension),
            'target_rotation': torch.from_numpy(target_rotation),
            'target_velocity': torch.from_numpy(target_velocity),
            # --- 结束 ---

            # --- 新增：包含用于稀疏目标的掩码 ---
            'reg_mask': torch.from_numpy(reg_mask),
            # --- 结束 ---

            # Add obj_count for potential use elsewhere (e.g., logging)
            'num_objs': torch.tensor(obj_count, dtype=torch.int)
            # 'sparse_gt_list': sparse_gt_boxes_for_eval # Optional: Add if needed elsewhere
        }
        # --- 结束 ---

        # --- 移除旧的返回逻辑 ---
        # cls_map_tensor = torch.from_numpy(cls_map)
        # ...
        # dense_maps = { ... }
        # return dense_maps, sparse_gt_boxes
        # ---

        return targets_dict


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
        targets_dict = self.get_detection_targets(rec)
        # ---

        # --- 修改：返回稀疏GT列表和sample_token ---
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_bev, targets_dict, sample_token
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
            height_map[0, y_idx, x_idx] = torch.max(
                height_map[0, y_idx, x_idx], z[i])  # Max height in the cell
            intensity_map[0, y_idx, x_idx] = torch.max(
                intensity_map[0, y_idx, x_idx], intensity[i])  # Max intensity in the cell
            density_map[0, y_idx, x_idx] += 1  # Count points per cell

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
        if height_bin_indices.numel() > 0:  # Handle case where height_map might be empty (all zeros)
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
