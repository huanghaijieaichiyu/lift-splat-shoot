"""
NuScenes数据集信息缓存工具

提供了生成并缓存NuScenes数据集的标签信息的功能，以避免每次训练或测试时
重新读取数据集原始文件，通过将数据序列化到info.pkl文件中提高数据加载效率。

参考了原始项目中的数据加载逻辑并进行了优化。
"""

import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.splits import create_splits_scenes


def get_available_scenes(nusc):
    """
    获取可用的场景列表

    Args:
        nusc: NuScenes数据集实例

    Returns:
        available_scenes: 可用场景列表，每个元素是(场景名称, 场景token)元组
    """
    available_scenes = []
    print("总场景数:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False

        # 检查该场景的所有帧是否存在
        while has_more_frames:
            lidar_path = os.path.join(nusc.dataroot, sd_rec['filename'])
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False

        if scene_not_exist:
            continue

        available_scenes.append((scene["name"], scene_token))

    print("可用场景数:", len(available_scenes))
    return available_scenes


def fill_trainval_infos(nusc, train_scenes, val_scenes, test=False, max_sweeps=10):
    """
    生成训练和验证集的信息

    Args:
        nusc: NuScenes数据集实例
        train_scenes: 训练集场景列表
        val_scenes: 验证集场景列表
        test: 是否为测试模式（不包含标签信息）
        max_sweeps: 最大叠加的激光雷达扫描数量

    Returns:
        train_nusc_infos: 训练集信息
        val_nusc_infos: 验证集信息
    """
    train_nusc_infos = []
    val_nusc_infos = []

    ref_chan = "LIDAR_TOP"  # 参考传感器

    # 加载训练集场景
    for sample in tqdm(nusc.sample, desc="处理训练和验证样本"):
        # 获取场景信息
        scene_token = sample["scene_token"]
        scene = nusc.get("scene", scene_token)
        scene_name = scene["name"]

        # 确定该样本属于训练集还是验证集
        if scene_name in train_scenes:
            info_list = train_nusc_infos
        elif scene_name in val_scenes:
            info_list = val_nusc_infos
        else:
            continue

        # 获取参考传感器数据
        ref_sd_token = sample["data"][ref_chan]
        ref_sd = nusc.get("sample_data", ref_sd_token)
        ref_cs_token = ref_sd["calibrated_sensor_token"]
        ref_cs = nusc.get("calibrated_sensor", ref_cs_token)

        # 参考位姿信息
        ref_pose = nusc.get("ego_pose", ref_sd["ego_pose_token"])
        ref_lidar_path = os.path.join(nusc.dataroot, ref_sd["filename"])

        # 获取扫描和相关的位姿信息
        ref_time = 1e-6 * ref_sd["timestamp"]

        # 获取参考传感器相关的标定参数
        ref_lidar_rotation = Quaternion(ref_cs["rotation"]).rotation_matrix
        ref_lidar_translation = np.array(ref_cs["translation"])

        # 获取相机信息
        camera_info = {}
        for cam_name in ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                         "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            camera_info[cam_name] = get_camera_info(nusc, sample, cam_name)

        # 生成带有传感器信息的样本数据
        info = {
            "lidar_path": ref_lidar_path,
            "token": sample["token"],
            "timestamp": ref_time,
            "sweeps": [],
            "ref_from_car": {
                "rotation": ref_lidar_rotation,
                "translation": ref_lidar_translation,
            },
            "camera": camera_info,
        }

        # 添加历史帧（sweeps）
        if not test:
            sweep_data = get_sweeps(nusc, ref_sd, ref_time, max_sweeps)
            info["sweeps"] = sweep_data

        # 添加标注信息（测试模式下不需要）
        if not test:
            annotations = get_annotations(nusc, sample, ref_pose)
            info["annotations"] = annotations

        info_list.append(info)

    return train_nusc_infos, val_nusc_infos


def get_camera_info(nusc, sample, cam_name):
    """
    获取相机信息

    Args:
        nusc: NuScenes数据集实例
        sample: 样本数据
        cam_name: 相机名称

    Returns:
        camera_info: 相机信息字典
    """
    cam_token = sample["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    cam_path = os.path.join(nusc.dataroot, cam_data["filename"])

    cs_token = cam_data["calibrated_sensor_token"]
    cs_rec = nusc.get("calibrated_sensor", cs_token)
    cam_intrinsic = np.array(cs_rec["camera_intrinsic"])

    cam_info = {
        "data_path": cam_path,
        "type": cam_name,
        "token": cam_token,
        "cs_token": cs_token,
        "intrinsic": cam_intrinsic,
        "sensor2lidar_rotation": Quaternion(cs_rec["rotation"]).rotation_matrix,
        "sensor2lidar_translation": np.array(cs_rec["translation"]),
    }

    return cam_info


def get_sweeps(nusc, ref_sd, ref_time, max_sweeps):
    """
    获取历史帧数据

    Args:
        nusc: NuScenes数据集实例
        ref_sd: 参考传感器数据
        ref_time: 参考时间戳
        max_sweeps: 最大叠加的激光雷达扫描数量

    Returns:
        sweeps: 历史帧列表
    """
    sweeps = []

    # 获取历史sweep帧
    cur_sd = ref_sd
    for _ in range(max_sweeps):
        if cur_sd["prev"] == "":
            break

        cur_sd = nusc.get("sample_data", cur_sd["prev"])
        cs_token = cur_sd["calibrated_sensor_token"]
        cs_rec = nusc.get("calibrated_sensor", cs_token)
        pose_token = cur_sd["ego_pose_token"]
        pose_rec = nusc.get("ego_pose", pose_token)

        lidar_path = os.path.join(nusc.dataroot, cur_sd["filename"])
        time_lag = ref_time - 1e-6 * cur_sd["timestamp"]

        # 历史帧信息
        sweep = {
            "data_path": lidar_path,
            "token": cur_sd["token"],
            "time_lag": time_lag,
            "sensor2ego_translation": np.array(cs_rec["translation"]),
            "sensor2ego_rotation": Quaternion(cs_rec["rotation"]).rotation_matrix,
            "ego2global_translation": np.array(pose_rec["translation"]),
            "ego2global_rotation": Quaternion(pose_rec["rotation"]).rotation_matrix,
        }
        sweeps.append(sweep)

    return sweeps


def get_annotations(nusc, sample, ref_pose):
    """
    获取标注信息

    Args:
        nusc: NuScenes数据集实例
        sample: 样本数据
        ref_pose: 参考位姿

    Returns:
        annotations: 标注信息字典
    """
    # 初始化标注数据结构
    annotations = {
        "gt_names": [],
        "gt_boxes": [],
        "num_points_in_gt": [],
        "visibility": [],
        "tokens": [],
    }

    # 计算从全局坐标系转换到参考自车坐标系的逆变换
    # 注意：我们需要平移量的负值和旋转的逆
    ref_translation_neg = -np.array(ref_pose["translation"])
    ref_rotation_inv = Quaternion(ref_pose["rotation"]).inverse

    # 处理所有标注
    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)

        # 跳过没有激光雷达点的实例
        num_lidar_pts = ann["num_lidar_pts"]
        if num_lidar_pts <= 0:
            continue

        # 获取3D边界框
        box = Box(
            ann["translation"],
            ann["size"],
            Quaternion(ann["rotation"]),
            name=ann["category_name"],
        )

        # 将边界框从全局坐标系转换到自车坐标系
        # 必须先平移，再旋转
        box.translate(ref_translation_neg)
        box.rotate(ref_rotation_inv)  # 使用逆四元数进行旋转

        # 获取3D边界框参数
        box_xyz = box.center
        box_dxdydz = box.wlh
        box_yaw = box.orientation.yaw_pitch_roll[0]

        # 组合成标准格式的边界框 [x, y, z, dx, dy, dz, yaw]
        gt_box = np.array([
            box_xyz[0], box_xyz[1], box_xyz[2],
            box_dxdydz[0], box_dxdydz[1], box_dxdydz[2],
            box_yaw
        ])

        # 添加到标注列表
        annotations["gt_boxes"].append(gt_box)
        annotations["gt_names"].append(ann["category_name"])
        annotations["num_points_in_gt"].append(num_lidar_pts)
        annotations["visibility"].append(ann["visibility_token"])
        annotations["tokens"].append(ann_token)

    # 转换为NumPy数组
    if len(annotations["gt_boxes"]) > 0:
        annotations["gt_boxes"] = np.array(annotations["gt_boxes"])
    else:
        annotations["gt_boxes"] = np.zeros((0, 7))

    return annotations


def create_nuscenes_infos(dataroot, version="v1.0-trainval", max_sweeps=10):
    """
    创建NuScenes数据集信息并保存到info.pkl文件

    Args:
        dataroot: 数据集根目录
        version: 数据集版本 (例如 'v1.0-mini', 'v1.0-trainval')
        max_sweeps: 最大叠加的激光雷达扫描数量

    Returns:
        None
    """
    print(f"=== 正在生成 NuScenes {version} 信息 ===")

    # 初始化NuScenes数据集
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # 获取可用的场景
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s[0] for s in available_scenes]

    # 获取数据集分割名称
    splits = create_splits_scenes()
    version_suffix = version.split('-')[-1]  # 获取 'mini' 或 'trainval' 或 'test'

    if version_suffix == 'mini':
        train_split_name = 'mini_train'
        val_split_name = 'mini_val'
        test_split_name = None  # mini 数据集通常没有明确的test分割
    elif version_suffix == 'trainval':
        train_split_name = 'train'
        val_split_name = 'val'
        test_split_name = None  # trainval不直接用于测试
    elif version_suffix == 'test':
        train_split_name = None  # test集没有训练分割
        val_split_name = None   # test集没有验证分割
        test_split_name = 'test'
    else:
        raise ValueError(f"无法识别的数据集版本后缀: {version_suffix}")

    # 获取对应分割的场景列表
    train_scenes_names = splits.get(train_split_name, [])
    val_scenes_names = splits.get(val_split_name, [])
    test_scenes_names = splits.get(test_split_name, [])

    # 确认场景是否都可用
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes_names))
    val_scenes = list(
        filter(lambda x: x in available_scene_names, val_scenes_names))
    test_scenes = list(
        filter(lambda x: x in available_scene_names, test_scenes_names))

    print(
        f"训练场景数: {len(train_scenes)}, 验证场景数: {len(val_scenes)}, 测试场景数: {len(test_scenes)}")

    # 生成训练和验证集信息
    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        nusc, train_scenes, val_scenes, test=False, max_sweeps=max_sweeps
    )

    # 生成测试集信息（如适用）
    test_nusc_infos = None
    if test_split_name and test_scenes:
        print(f"为测试集生成信息...")
        # 注意：测试集可能没有标注，fill_trainval_infos的test参数应为True
        test_nusc_infos, _ = fill_trainval_infos(
            nusc, test_scenes, [], test=True, max_sweeps=max_sweeps  # test=True
        )

    # 准备要保存的信息
    metadata = {
        "version": version,
        "max_sweeps": max_sweeps,
    }

    print(f"训练样本数: {len(train_nusc_infos)}, 验证样本数: {len(val_nusc_infos)}")

    data = {
        "infos": {
            "train": train_nusc_infos,
            "val": val_nusc_infos,
        },
        "metadata": metadata,
    }

    if test_nusc_infos is not None:
        data["infos"]["test"] = test_nusc_infos
        print(f"测试样本数: {len(test_nusc_infos)}")

    # 创建输出文件夹
    info_dir = os.path.join(dataroot, "infos")
    os.makedirs(info_dir, exist_ok=True)

    # 保存信息到pickle文件
    info_path = os.path.join(
        info_dir, f"nuscenes_{version}_infos_{max_sweeps}sweeps.pkl")
    print(f"将信息保存到: {info_path}")

    with open(info_path, "wb") as f:
        pickle.dump(data, f)

    print("=== NuScenes信息生成完成 ===")


def load_nuscenes_infos(dataroot, version="v1.0-trainval", max_sweeps=10):
    """
    加载NuScenes数据集信息，如果不存在则创建

    Args:
        dataroot: 数据集根目录
        version: 数据集版本
        max_sweeps: 最大叠加的激光雷达扫描数量

    Returns:
        infos: 数据集信息
    """
    info_path = os.path.join(
        dataroot, "infos", f"nuscenes_{version}_infos_{max_sweeps}sweeps.pkl")

    # 如果缓存文件不存在，就立即生成并返回结果
    if not os.path.exists(info_path):
        print(f"未找到info文件: {info_path}")
        print("开始生成新的info文件...")
        # 创建信息文件并直接获取创建的结果
        create_nuscenes_infos(dataroot, version, max_sweeps)

        # 确认文件已创建
        if os.path.exists(info_path):
            print(f"已成功创建缓存文件: {info_path}")
        else:
            raise FileNotFoundError(f"缓存文件创建失败: {info_path}")

    # 加载缓存文件
    print(f"加载NuScenes信息从: {info_path}")
    try:
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        return infos
    except Exception as e:
        # 如果加载失败，尝试重新生成
        print(f"加载缓存文件失败: {e}，尝试重新生成...")
        create_nuscenes_infos(dataroot, version, max_sweeps)
        with open(info_path, "rb") as f:
            infos = pickle.load(f)
        return infos


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NuScenes数据集信息生成工具")
    parser.add_argument("--dataroot", type=str,
                        default="/data/nuscenes", help="数据集根目录")
    parser.add_argument("--version", type=str,
                        default="v1.0-trainval", help="数据集版本")
    parser.add_argument("--max_sweeps", type=int,
                        default=10, help="最大叠加的激光雷达扫描数量")

    args = parser.parse_args()

    create_nuscenes_infos(
        dataroot=args.dataroot,
        version=args.version,
        max_sweeps=args.max_sweeps
    )
