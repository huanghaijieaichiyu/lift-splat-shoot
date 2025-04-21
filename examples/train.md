# src/train.py 技术文档

## 1. 概述

`src/train.py` 脚本包含了训练不同类型 BEV（鸟瞰图）模型的函数，主要涵盖了 BEV 分割、仅使用相机的 3D 目标检测以及多模态（相机+LiDAR）融合的 3D 目标检测任务。脚本集成了数据加载、模型编译、训练循环、验证、损失计算、优化器配置、学习率调度、混合精度训练、梯度裁剪、日志记录（TensorBoard）以及模型保存等功能。

**主要功能:**

*   **`train()`**: 训练用于 BEV 语义分割的模型（例如 Lift-Splat-Shoot 变体）。
*   **`train_3d()`**: 训练用于仅相机输入的 3D 目标检测模型（例如 `BEVENet`）。
*   **`train_fusion()`**: 训练用于多模态（相机+LiDAR）输入的 3D 目标检测模型。
*   **辅助函数**: 包括数据集缓存检查 (`check_and_ensure_cache`) 和简化的非极大值抑制 (`distance_based_nms`)。

## 2. 训练函数详解

### 2.1 `train()` - BEV 分割训练

*   **目的**: 训练一个模型，将 BEV 空间区分为前景和背景（二值分割）。
*   **核心逻辑**:
    1.  **参数解析**: 接收数据路径、训练周期、GPU ID、模型恢复路径、权重加载路径、混合精度启用标志、图像尺寸、数据增强参数、BEV 网格配置、批次大小、工作线程数、学习率、权重衰减等参数。
    2.  **数据缓存检查**: 调用 `check_and_ensure_cache` 确认 NuScenes 数据集信息缓存是否存在。
    3.  **数据加载**: 调用 `compile_data` 加载训练和验证数据集，使用 `segmentationdata` 解析器。
    4.  **模型编译**: 调用 `compile_model` 编译 BEV 分割模型 (默认为 `lss` 架构)。
    5.  **损失函数**: 使用 `nn.BCEWithLogitsLoss` (带 `pos_weight` 以处理类别不平衡)。
    6.  **优化器**: 使用 `Adam` 优化器。
    7.  **设备与 cuDNN**: 设置训练设备 (CPU 或 GPU)，可选启用 cuDNN 优化。
    8.  **恢复与加载**: 支持从检查点恢复训练 (`resume`) 或加载预训练权重 (`load_weight`)。
    9.  **日志记录**: 使用 `TensorBoardX` 的 `SummaryWriter` 记录训练和验证过程中的损失、IoU 等指标。
    10. **训练循环**:
        *   遍历训练数据加载器。
        *   执行模型前向传播，获取预测结果。
        *   计算 BCE 损失。
        *   执行反向传播和优化器步骤。
        *   可选使用混合精度 (`autocast`) 和梯度裁剪 (`clip_grad_norm_`)。
        *   记录批次损失和训练进度 (使用 `tqdm`)。
    11. **验证与保存**:
        *   定期（`val_step`）在验证集上评估模型，计算损失和 IoU (`get_val_info`, `get_batch_iou`)。
        *   记录验证指标到 TensorBoard。
        *   保存最新的模型检查点 (`last.pt`) 以便恢复。
        *   如果当前验证 IoU 超过历史最佳，则保存最佳模型 (`best.pt`)。
    12. **训练结束**: 关闭 `SummaryWriter`。

*   **关键参数**:
    *   `version`: 数据集版本 ('mini', 'trainval')。
    *   `dataroot`: 数据集根目录。
    *   `nepochs`: 训练总轮数。
    *   `gpuid`: 使用的 GPU ID (-1 表示 CPU)。
    *   `cuDNN`: 是否启用 cuDNN 加速。
    *   `resume`: 恢复训练的检查点路径。
    *   `load_weight`: 加载权重的路径。
    *   `amp`: 是否启用自动混合精度 (AMP)。
    *   `H`, `W`: 原始图像高宽。
    *   `resize_lim`, `final_dim`, `bot_pct_lim`, `rot_lim`, `rand_flip`, `ncams`: 数据增强和相机相关配置。
    *   `max_grad_norm`: 梯度裁剪阈值。
    *   `pos_weight`: BCE 损失的正样本权重。
    *   `logdir`: TensorBoard 日志和模型保存目录。
    *   `xbound`, `ybound`, `zbound`, `dbound`: BEV 网格和深度配置。
    *   `bsz`, `nworkers`: 批次大小和数据加载工作线程数。
    *   `lr`, `weight_decay`: 学习率和权重衰减。

### 2.2 `train_3d()` - 仅相机 3D 检测训练

*   **目的**: 训练一个基于相机输入的 3D 目标检测模型 (如 `BEVENet`)。
*   **核心逻辑**:
    1.  **参数解析**: 同 `train()`，但增加了 `num_classes`, `enable_multiscale`, `use_enhanced_bev` 等 3D 检测特定参数。
    2.  **数据缓存检查**: 同 `train()`。
    3.  **数据加载**: 调用 `compile_data`，使用 `detection3d` 解析器，返回图像、相机参数以及包含 GT 信息的 `targets_list`。
    4.  **模型编译**: 调用 `compile_model` 编译 3D 检测模型 (通常为 `beve`)，传递 `num_classes` 和计算出的总输出通道数。
    5.  **损失函数**: 使用自定义的 `DetectionBEVLoss`，计算分类、回归（包含 BEV Diou, Z, H, Vel 等子项）和 IoU 损失。
    6.  **优化器**: 使用 `Adam` 优化器 (betas=(0.5, 0.999))。
    7.  **学习率调度器**: 使用 `OneCycleLR` 余弦退火策略。
    8.  **设备与 cuDNN**: 同 `train()`。
    9.  **恢复与加载**: 同 `train()`，额外处理调度器状态和 `best_map`。
    10. **日志记录**: 同 `train()`，记录更详细的损失分量（cls, iou, bev_diou, z, h, vel）和学习率。
    11. **训练循环**:
        *   遍历训练数据加载器，获取图像、相机参数和 `targets_list`。
        *   将 `targets_list` 转换为设备上的 `targets_dict`。
        *   执行模型前向传播，获取预测字典 `preds`。
        *   使用 `loss_fn` 计算损失字典 `losses`。
        *   执行反向传播和优化器步骤，支持混合精度。
        *   应用梯度裁剪。
        *   **更新学习率调度器** (`scheduler.step()`)。
        *   累加各损失分量用于 epoch 记录。
        *   记录批次损失和训练进度。
    12. **验证与评估**:
        *   定期在验证集上进行评估。
        *   **收集预测和 GT**: 在 `torch.no_grad()` 环境下遍历验证集：
            *   模型前向传播获取 `preds`。
            *   计算并累加验证损失。
            *   调用 `decode_predictions` (来自 `evaluate_3d`) 将模型输出 `preds` 解码为实际的检测框列表 `batch_dets`（包含坐标、类别、分数等）。
            *   **手动处理 GT**: 从 `targets_dict` (`cls_targets`, `reg_targets`) 中提取并格式化 GT 框列表 `batch_gts`，与 `batch_dets` 格式对齐。
            *   将 `batch_dets` 和 `batch_gts` 添加到 `all_predictions` 和 `all_targets` 列表中。
        *   **计算简化 mAP**: 在验证循环结束后，调用内部定义的 `calculate_simple_ap` 函数，使用 `all_predictions` 和 `all_targets` 计算基于中心点距离匹配的简化 mAP（例如，在 1 米和 2 米距离阈值下）。
        *   记录平均验证损失和简化 mAP 到 TensorBoard。
        *   **保存模型**: 保存最新的模型 (`last.pt`)。如果当前验证的简化 mAP (例如 @dist=2m) 优于历史最佳 `best_map`，则保存最佳模型 (`best_map.pt`)。
        *   **可视化**: (可选) 在每个验证周期的第一个批次，将输入图像、BEV 预测热力图等可视化结果添加到 TensorBoard。
    13. **训练结束**: 关闭 `SummaryWriter`。

*   **关键参数**:
    *   除 `train()` 中的参数外，还包括：
    *   `num_classes`: 检测目标的类别数量。
    *   `enable_multiscale`: (当前实现中未完全启用) 是否启用多尺度特征训练。
    *   `use_enhanced_bev`: (当前实现中未完全启用) 是否使用增强的 BEV 投影。
    *   `zbound`: BEV 网格 Z 轴范围，对 3D 检测更重要。

### 2.3 `train_fusion()` - 多模态 3D 检测训练

*   **目的**: 训练一个融合相机和 LiDAR 输入的 3D 目标检测模型。
*   **核心逻辑**: 与 `train_3d()` 非常相似，主要区别在于：
    1.  **数据加载**: 调用 `compile_data`，使用 `multimodal_detection` 解析器，额外返回 `lidar_bev` 特征。
    2.  **模型编译**: 调用 `compile_model`，指定 `model='fusion'`，并传递 `lidar_channels` 参数。融合模型内部会处理相机和 LiDAR 特征的融合。
    3.  **模型前向传播**: 输入增加了 `lidar_bev.to(device)`。
    4.  **梯度累积**: 支持梯度累积 (`grad_accum_steps`)，在累积多个批次的梯度后再更新模型参数，用于模拟更大的批次大小。损失需要相应地进行缩放。
    5.  **混合精度处理**: 明确使用 `GradScaler` 进行混合精度训练的梯度缩放和优化器步骤。
    6.  **验证评估**: 在验证循环中，增加了对 NMS (`distance_based_nms`) 的应用。在 `decode_predictions` 之后，对每个样本的检测结果 `batch_dets` 应用 NMS，得到 `batch_dets_nms`，再用于后续的 mAP 计算。
    7.  **其他**: 参数、损失函数、优化器、调度器、日志、检查点保存、mAP 计算逻辑与 `train_3d()` 基本一致。

*   **关键参数**:
    *   除 `train_3d()` 中的参数外，还包括：
    *   `lidar_channels`: 输入的 LiDAR BEV 特征通道数。
    *   `bsz`: 由于内存限制，多模态训练通常使用较小的批次大小。
    *   `nworkers`: 可能因为数据加载复杂性设为 0。
    *   `grad_accum_steps`: 梯度累积步数。

## 3. 辅助函数详解

### 3.1 `distance_based_nms()`

*   **目的**: 执行一个简化的非极大值抑制 (NMS) 过程，主要基于检测框中心点在 BEV 平面上的距离。用于抑制同一类别中距离过近且分数较低的框。
*   **输入**:
    *   `boxes_xyz`: `[N, 3]` 或 `[N, 2]` 检测框中心坐标 (使用 x, y 计算距离)。
    *   `scores`: `[N]` 检测框置信度。
    *   `classes`: `[N]` 检测框类别。
    *   `dist_threshold`: 同类别内抑制的距离阈值。
    *   `score_threshold`: 预过滤的最低分数阈值。
*   **处理**:
    1.  按 `score_threshold` 过滤框。
    2.  遍历每个唯一类别。
    3.  对当前类别的框按分数降序排序。
    4.  迭代排序后的框：如果当前框未被抑制，则保留它，并抑制掉所有与它距离小于 `dist_threshold` 且尚未被抑制的其他同类框。
*   **输出**: 保留的检测框的原始索引列表。

### 3.2 `check_and_ensure_cache()`

*   **目的**: 在训练开始前检查 NuScenes 数据集的信息缓存文件是否存在且有效。如果缓存文件不存在或无效，`nuscenes.nuscenes.NuScenes` 类（通过 `load_nuscenes_infos` 间接调用）通常会自动尝试创建缓存。
*   **输入**:
    *   `dataroot`: 数据集根目录。
    *   `version`: 数据集版本 ('mini', 'trainval')。
*   **处理**:
    1.  尝试调用 `load_nuscenes_infos` 加载指定版本的数据信息。
    2.  检查返回的 `nusc_infos` 结构是否完整（包含 'infos', 'train', 'val' 键）。
    3.  打印缓存验证结果或错误信息。
*   **输出**: 布尔值，指示缓存是否看似有效（注意：即使返回 False，训练仍可能继续，但初始加载会较慢）。

### 3.3 `calculate_simple_ap()` (定义在 `train_3d` 和 `train_fusion` 内部)

*   **目的**: 计算一个简化的平均精度均值 (mAP)，作为官方 NuScenes 指标的快速替代评估。它不计算精确的 PR 曲线，而是基于预测框和 GT 框之间的中心点距离进行匹配。
*   **输入**:
    *   `preds_list`: 包含多个样本预测结果的列表，每个样本是一个字典 (如 `decode_predictions` 或 NMS 后输出的格式)。
    *   `targets_list`: 包含多个样本 GT 结果的列表，每个样本是一个字典 (格式与 `preds_list` 对齐)。
    *   `num_classes`: 类别总数。
    *   `dist_threshold`: 匹配预测框和 GT 框的最大距离阈值。
*   **处理**:
    1.  遍历每个类别 (1 到 `num_classes`)。
    2.  遍历每个样本：
        *   统计该类别的 GT 总数。
        *   获取该类别的预测框和 GT 框。
        *   **简化匹配**: (可能实现方式) 对每个预测框，查找最近的未匹配 GT 框。如果距离小于 `dist_threshold`，则视为 True Positive (TP)，并将该 GT 标记为已匹配；否则视为 False Positive (FP)。
        *   累加样本的 TP 和 FP。
    3.  计算每个类别的 Precision 和 Recall。
    4.  计算每个类别的 AP (可以使用 Precision * Recall 或 F1 分数作为简化替代)。
    5.  计算所有类别的平均 AP (mAP)。
*   **输出**: mAP 分数和包含各类别 AP 的字典。
*   **局限性**: 这是一个**简化**指标，与官方 NuScenes mAP 计算方法（基于 IoU 和 PR 曲线）不同，仅用于训练过程中的快速反馈。

## 4. 依赖项

*   **PyTorch**: 核心框架 (`torch`, `torch.nn`, `torch.optim`, `torch.cuda.amp`, `torch.backends.cudnn`)。
*   **NumPy**: 用于数值计算。
*   **TensorBoardX**: 用于训练过程可视化和日志记录。
*   **tqdm**: 用于显示进度条。
*   **自定义模块**:
    *   `.models`: 包含 `compile_model` 函数，用于根据配置构建模型。
    *   `.data`: 包含 `compile_data` 函数，用于加载和预处理数据。
    *   `.tools`: 包含 `save_path`, `get_batch_iou`, `get_val_info`, `DetectionBEVLoss` 等工具函数和类。
    *   `.nuscenes_info`: 包含 `load_nuscenes_infos` 用于处理 NuScenes 数据集缓存。
    *   `.evaluate_3d`: (间接依赖) 包含 `decode_predictions` 用于解码模型输出。
*   **os, time, contextlib**: Python 标准库。

## 5. 使用示例 (命令行)

```bash
# 示例：训练 BEVENet 3D 检测模型 (mini 数据集)
python src/train.py train_3d \
    --version mini \
    --dataroot /path/to/nuscenes \
    --gpuid 0 \
    --logdir ./runs_3d_mini \
    --nepochs 50 \
    --bsz 4 \
    --nworkers 4 \
    --lr 2e-4 \
    --num_classes 10 \
    # ... 其他参数 ...

# 示例：训练多模态融合模型 (mini 数据集，使用梯度累积)
python src/train.py train_fusion \
    --version mini \
    --dataroot /path/to/nuscenes \
    --gpuid 0 \
    --logdir ./runs_fusion_mini \
    --nepochs 50 \
    --bsz 2 \
    --nworkers 0 \
    --grad_accum_steps 2 \ # 有效批次大小为 2*2=4
    --lr 1e-3 \
    --num_classes 10 \
    --lidar_channels 64 \
    # ... 其他参数 ...

# 示例：恢复 3D 检测训练
python src/train.py train_3d \
    --resume ./runs_3d_mini/last.pt \
    # ... 其他参数保持一致 ...
```

**注意**: 上述命令行示例假设 `src/train.py` 被修改为接受命令行参数（例如使用 `argparse`）来调用相应的训练函数。原始代码直接通过函数调用启动训练，需要相应调整才能通过命令行运行。