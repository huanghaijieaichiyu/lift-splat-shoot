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
from nuscenes.utils.data_classes import Box, LidarPointCloud
from glob import glob

from .tools import img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, nusc_maps, is_train, data_aug_conf, grid_conf):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        self.fix_nuscenes_formatting()

        print(self, 'found scenes', len(self.scenes))
        print(self, 'found samples', len(self.ixes))

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/10sec folders, combine strings appropriately.
        """
        # Check if self.scenes is empty before proceeding
        if not self.scenes:
            print("Warning: self.scenes is empty in fix_nuscenes_formatting. Skipping.")
            return

        # check if default file format exists in scenes
        scene_folder_token = self.scenes[0].split(
            '/')[-1].replace('scene-', '')
        token_len = len(scene_folder_token)
        scene_folder = os.path.join(
            self.nusc.dataroot, self.scenes[0][:-(token_len+1)])

        # default file format check
        scene_json_path = os.path.join(
            scene_folder, 'scene-{}.json'.format(scene_folder_token))
        if not os.path.exists(scene_json_path):
            print(
                f"Info: Standard scene file {scene_json_path} not found. Checking for older 10sec format.")
            # change paths to use the 10sec folders
            # determine keyframe or sweep
            # Use glob directly from the glob module
            import glob as pyglob  # Use an alias to avoid conflict if self.glob exists
            fs = pyglob.glob(os.path.join(scene_folder, '*__{}*__*'.format('LIDAR_TOP',
                                                                           'sweep' if 'sweeps' in scene_folder else 'keyframe')))

            # Check if glob found any files before proceeding
            if not fs:
                print(
                    f"Warning: No LIDAR_TOP files found matching pattern in {scene_folder}. Skipping scene path modification logic.")
                # If no files are found, we assume the structure is unexpected or doesn't match the old format.
                # We do *not* modify self.scenes in this case.
            else:
                # Original logic, now safe because fs is not empty
                _is_keyframe = 'keyframe' in os.path.basename(
                    fs[0])  # Check basename for keyframe/sweep string

                # update scene strings
                new_scenes = []
                print(
                    "Info: Attempting to update scene paths based on detected 10sec format.")
                for scene in self.scenes:
                    scene_folder_token_loop = scene.split(
                        '/')[-1].replace('scene-', '')
                    token_len_loop = len(scene_folder_token_loop)
                    # Construct the path to the scene folder more reliably
                    # Assuming scene is like 'samples/scene-XXXX' or similar relative path from dataroot
                    scene_base_path = os.path.join(self.nusc.dataroot, scene)
                    # Get the directory containing the scene folder
                    scene_folder_loop = os.path.dirname(scene_base_path)

                    new_paths = []
                    glob_pattern = ''  # Define glob_pattern before use

                    if _is_keyframe:
                        glob_pattern = os.path.join(
                            scene_folder_loop, '*__{}*__*keyframe*'.format('LIDAR_TOP'))
                        inner_fs = pyglob.glob(glob_pattern)
                        if inner_fs:
                            # Use os.path.basename
                            new_paths = sorted([os.path.basename(f)
                                               for f in inner_fs])
                            new_paths = [
                                f.replace('__LIDAR_TOP__keyframe', '') for f in new_paths]
                    else:
                        glob_pattern = os.path.join(
                            scene_folder_loop, '*__{}*__*sweep*'.format('LIDAR_TOP'))
                        inner_fs = pyglob.glob(glob_pattern)
                        if inner_fs:
                            # Use os.path.basename
                            new_paths = sorted([os.path.basename(f)
                                               for f in inner_fs])
                            new_paths = [
                                f.replace('__LIDAR_TOP__sweep', '') for f in new_paths]

                    if new_paths:
                        # Construct the new scene string relative to dataroot?
                        # Original code's output path construction was ambiguous.
                        # Let's assume the goal is just to use the original scene identifier
                        # if modification fails or isn't needed, or construct a new identifier if successful.
                        # The original scene string seems sufficient if we don't need the 10sec format.
                        # Re-evaluating the purpose: It seems to modify self.scenes list strings.
                        # Let's try to stick closer to the original output format if possible.
                        # The new path seems to be: scene_path_prefix/scene_token/first__last
                        # Path part before 'scene-XXXX'
                        scene_prefix = scene[:-(token_len_loop+1)]
                        new_scene_string = os.path.join(
                            scene_prefix, scene_folder_token_loop, new_paths[0]+'__' + new_paths[-1])
                        # Use forward slashes for consistency
                        new_scene_string = new_scene_string.replace('\\', '/')
                        new_scenes.append(new_scene_string)
                        # print(f"Debug: Updated scene {scene} to {new_scene_string}") # Optional debug
                    else:
                        print(
                            f"Warning: Could not determine new path for scene folder {scene_folder_loop} using pattern {glob_pattern}. Appending original scene path: {scene}")
                        # Append original if update fails for this scene
                        new_scenes.append(scene)

                # Update self.scenes only if modifications were made and successful
                self.scenes = new_scenes

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
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = sum(self.data_aug_conf['resize_lim'])/2
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - sum(self.data_aug_conf['bot_pct_lim'])/2)*newH) - fH
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

    def get_cam_info(self, rec, cam_front_token):
        # get camera intrinsics and extrinsics
        cam = self.nusc.get('sample_data', cam_front_token)
        sens = self.nusc.get('calibrated_sensor',
                             cam['calibrated_sensor_token'])
        pose = self.nusc.get('ego_pose', cam['ego_pose_token'])
        return sens, pose

    def get_binimg(self, rec, cam_info):
        egopose = self.nusc.get('ego_pose', self.nusc.get(
            'sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category filtering if necessary
            # if not inst['category_name'].split.'.'[0] == 'vehicle':
            #     continue
            box = Box(inst['translation'], inst['size'],
                      Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], (1,))

        return torch.Tensor(img).unsqueeze(0)

    def __len__(self):
        return len(self.ixes)


class SegmentationData(NuscData):
    def __init__(self, nusc, nusc_maps, is_train, data_aug_conf, grid_conf):
        super(SegmentationData, self).__init__(
            nusc, nusc_maps, is_train, data_aug_conf, grid_conf)

    def __getitem__(self, index):
        rec = self.ixes[index]

        # Get sensor data tokens
        lidar_token = rec['data']['LIDAR_TOP']
        cam_front_token = rec['data']['CAM_FRONT']

        # Check if CAM_BACK is available, otherwise skip (relevant for older/partial data)
        # This check might need adjustment based on your exact dataset structure
        if 'CAM_BACK' not in rec['data']:
            # Handle missing camera, e.g., return None or skip
            # For simplicity, let's just use the front cam token for cam_info
            # print(f"Warning: CAM_BACK missing for record {rec['token']}. Using CAM_FRONT for cam_info.")
            pass  # Proceed, assuming cam_info from front is sufficient

        cams = self.data_aug_conf['cams']
        # Ensure all required cams are present in the record
        cams = [cam for cam in cams if cam in rec['data']]
        if len(cams) != self.data_aug_conf['Ncams']:
            print(
                f"Warning: Expected {self.data_aug_conf['Ncams']} cameras, but found {len(cams)} for record {rec['token']}. Required: {self.data_aug_conf['cams']}")
            # Depending on the model, might need to handle this case (e.g., pad data or skip sample)
            # For now, proceeding with available cameras.

        # Get camera info (using CAM_FRONT as reference)
        cam_info = self.get_cam_info(rec, cam_front_token)

        # Get image data for available cameras
        imgs, rots, trans, intrins, post_rots, post_trans = \
            self.get_image_data(rec, cams)

        # Get binary segmentation map
        binimg = self.get_binimg(rec, cam_info)

        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


# --- Added LiDAR BEV Helper ---
def create_lidar_bev(points, grid_conf, lidar_inC=1):
    """
    Creates a BEV map from LiDAR point cloud.
    Args:
        points (np.array): LiDAR points (N x 3 or N x 4+).
        grid_conf (dict): Configuration for the BEV grid.
            Requires 'lidar_xbound', 'lidar_ybound'.
        lidar_inC (int): Number of channels for the BEV map (e.g., 1 for height).
    Returns:
        torch.Tensor: LiDAR BEV map (lidar_inC x H_bev x W_bev).
    """
    xbound = grid_conf['lidar_xbound']
    ybound = grid_conf['lidar_ybound']
    # Get grid dimensions and resolution
    # Z bounds don't matter here
    dx, bx, nx = gen_dx_bx(xbound, ybound, [0, 1, 1])
    # Convert to numpy int array [nx, ny, nz] -> [nx, ny]
    nx = nx.numpy().astype(int)
    dx = dx.numpy()
    bx = bx.numpy()

    # BEV grid dimensions
    W_bev, H_bev = nx[0], nx[1]

    # Filter points within the bounds (x, y)
    mask = (points[:, 0] >= xbound[0]) & (points[:, 0] < xbound[1]) & \
           (points[:, 1] >= ybound[0]) & (points[:, 1] < ybound[1])
    points = points[mask]

    # Convert points to grid coordinates
    points_grid = ((points[:, :2] - bx[:2] +
                   dx[:2] / 2.0) / dx[:2]).astype(int)

    # Create BEV map (example: height map)
    # Initialize BEV map
    bev_map = np.zeros((H_bev, W_bev, lidar_inC), dtype=np.float32)

    # Sort points by height (descending) so highest point determines cell value
    # Assuming Z is the 3rd column (index 2)
    # Check if there are points left and they have z-coord
    if points.shape[0] > 0 and points.shape[1] > 2:
        sort_idx = np.argsort(-points[:, 2])
        points_grid = points_grid[sort_idx]
        points = points[sort_idx]

        # Populate BEV map (height map example)
        # Assign points to grid cells, overwriting with higher points
        # Clamp coordinates to be within grid bounds just in case
        points_grid[:, 0] = np.clip(points_grid[:, 0], 0, W_bev - 1)
        points_grid[:, 1] = np.clip(points_grid[:, 1], 0, H_bev - 1)

        if lidar_inC == 1:
            # Simple height map: use z-coordinate
            bev_map[points_grid[:, 1], points_grid[:, 0], 0] = points[:, 2]
        else:
            # Example for multi-channel (e.g., height, intensity)
            # This requires points to have intensity (e.g., column 3 or 4)
            # Removed lines causing z_min/z_max errors
            # intensity_norm = points[:, 3] / 255.0 # Assuming intensity is 4th col
            # Assign channels - needs careful implementation based on desired features
            # bev_map[points_grid[:, 1], points_grid[:, 0], 0] = height_norm
            # bev_map[points_grid[:, 1], points_grid[:, 0], 1] = intensity_norm
            # ... fill other channels ...
            # For now, just duplicate height if lidar_inC > 1 as placeholder
            bev_map[points_grid[:, 1], points_grid[:, 0],
                    :] = points[:, 2, np.newaxis]

    # Convert to tensor (C x H x W)
    bev_tensor = torch.from_numpy(bev_map).permute(2, 0, 1).float()

    return bev_tensor
# --- End LiDAR BEV Helper ---


# --- Added FusionData Class ---
class FusionData(NuscData):
    def __init__(self, nusc, nusc_maps, is_train, data_aug_conf, grid_conf, lidar_inC=1):
        super(FusionData, self).__init__(
            nusc, nusc_maps, is_train, data_aug_conf, grid_conf)
        self.lidar_inC = lidar_inC
        print(
            f"Initialized FusionData with is_train={is_train}, lidar_inC={self.lidar_inC}")

    def __getitem__(self, index):
        rec = self.ixes[index]

        # Get sensor data tokens
        cam_front_token = rec['data']['CAM_FRONT']
        lidar_token = rec['data']['LIDAR_TOP']

        # Get sample token for validation
        sample_token = rec['token']

        cams = self.data_aug_conf['cams']
        cams = [cam for cam in cams if cam in rec['data']]
        if len(cams) != self.data_aug_conf['Ncams']:
            print(
                f"Warning: Expected {self.data_aug_conf['Ncams']} cameras, but found {len(cams)} for record {rec['token']}")
            pass

        # Get camera info and data
        cam_info = self.get_cam_info(rec, cam_front_token)
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(
            rec, cams)
        binimg = self.get_binimg(rec, cam_info)

        # Get LiDAR data
        lidar_path = self.nusc.get_sample_data_path(lidar_token)
        points_full = LidarPointCloud.from_file(lidar_path).points.T
        points = points_full[:, :3]

        # Transform points to ego vehicle frame
        lidar_sd_rec = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get(
            'calibrated_sensor', lidar_sd_rec['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', lidar_sd_rec['ego_pose_token'])

        lidar_to_ego_trans = np.array(pose_record['translation'])
        lidar_to_ego_rot = Quaternion(pose_record['rotation'])

        points_ego = np.dot(lidar_to_ego_rot.rotation_matrix, points.T).T
        points_ego += lidar_to_ego_trans

        # Create LiDAR BEV map
        lidar_bev = create_lidar_bev(
            points_ego, self.grid_conf, self.lidar_inC)

        # Return all data including sample_token
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg, lidar_bev, sample_token
# --- End FusionData Class ---


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf=None, grid_conf=None, bsz=4, nworkers=4, parser_name='segmentationdata', **kwargs):
    nusc = NuScenes(version='v1.0-' + version,
                    dataroot=dataroot, verbose=False)
    # Corrected initialization: pass maps to NuscData derivatives
    # Assuming nusc_maps object is available or None
    # Load maps if needed, e.g.:
    # from nuscenes.map_expansion.map_api import NuScenesMapExplorer
    # nusc_maps = {
    #     map_name: NuScenesMapExplorer(dataroot=dataroot, map_name=map_name)
    #     for map_name in [
    #         "singapore-hollandvillage",
    #         "singapore-queenstown",
    #         "boston-seaport",
    #         "singapore-onenorth",
    #     ]
    # }
    nusc_maps = None  # Placeholder if maps are not used directly by these classes

    # --- Updated Parser Logic ---
    if parser_name == 'segmentationdata':
        print("Using SegmentationData parser.")
        traindata = SegmentationData(
            nusc, nusc_maps, is_train=True, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
        valdata = SegmentationData(
            nusc, nusc_maps, is_train=False, data_aug_conf=data_aug_conf, grid_conf=grid_conf)
    elif parser_name == 'fusiondata':
        print("Using FusionData parser.")
        # Get lidar_inC from kwargs or default to 1
        lidar_inC = kwargs.get('lidar_inC', 1)
        traindata = FusionData(nusc, nusc_maps, is_train=True,
                               data_aug_conf=data_aug_conf, grid_conf=grid_conf, lidar_inC=lidar_inC)
        valdata = FusionData(nusc, nusc_maps, is_train=False,
                             data_aug_conf=data_aug_conf, grid_conf=grid_conf, lidar_inC=lidar_inC)
    else:
        raise ValueError(f"Unknown parser_name: {parser_name}")
    # --- End Updated Parser Logic ---

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz, shuffle=True,
                                              num_workers=nworkers, drop_last=True, pin_memory=False, worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=bsz, shuffle=False, num_workers=nworkers, pin_memory=False)

    return trainloader, valloader
