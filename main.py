"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src
import src.vedio_writer
import src.evaluate_3d  # 添加3D评估模块


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,
        'train': src.train.train,
        'train_3d': src.train.train_3d,  # 训练3D目标检测模型
        'train_fusion': src.train.train_fusion,  # 训练融合目标检测模型
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
        'viz_3d_detection': src.explore.viz_3d_detection,
        'viz_fusion_detection': src.explore.viz_fusion_detection,
        'vedio_writer': src.vedio_writer.vedio_writer,
        'train_3d_detection': src.train.train_3d,  # 训练3D目标检测模型
        'eval_3d_detection': src.evaluate_3d.evaluate_3d_detection  # 评估3D目标检测性能
    })
