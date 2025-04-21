# BEVENet 技术文档

## 1. 概述

`BEVENet` 是一个用于自动驾驶场景下 3D 目标检测的深度学习模型。它主要基于相机输入，通过视图变换将多视角图像特征转换到鸟瞰图（Bird's-Eye-View, BEV）空间，并在 BEV 空间中进行目标检测。该模型的实现参考了 BEVDepth 和 Lift-Splat-Shoot 等工作的思想，特别是引入了显式的深度预测和利用相机参数增强特征表示，并采用纯卷积（Convolutional-Only）的方式构建 BEV 编码器和检测头，旨在提高效率。

**核心特点:**

*   **多相机输入:** 处理来自多个车载摄像头的数据。
*   **视图变换:** 将 2D 图像特征 "提升" (Lift) 到 3D 空间，"散布" (Splat) 到 BEV 网格上。
*   **深度感知:** 集成了深度预测网络 (`DepthNet`)，利用相机内外参信息显式地估计深度分布，并用深度概率加权特征。
*   **BEV 空间检测:** 在生成的 BEV 特征图上进行目标分类和边界框回归。
*   **纯卷积架构:** BEV 编码器和检测头主要使用卷积层，避免了 Transformer 等复杂结构，以提升效率。

## 2. 模型架构

`BEVENet` 的整体处理流程如下：

1.  **相机特征提取 (`CamEncode_rep`)**: 对每个相机的输入图像进行特征提取，并初步估计深度分布和上下文特征。
2.  **深度细化与特征增强 (`DepthNet`)**: 结合相机内外参等几何信息，进一步细化深度预测和上下文特征。
3.  **几何变换与视锥体生成 (`create_frustum`, `get_geometry`)**:
    *   根据深度范围和特征图尺寸动态创建 3D 采样点（视锥体）。
    *   利用相机标定参数（内外参、数据增强变换），将视锥体点从像素坐标系转换到自车（ego）坐标系。
4.  **特征加权**: 使用 `DepthNet` 输出的最终深度概率对上下文特征进行加权。
5.  **体素池化 (`voxel_pooling`)**: 将加权后的 3D 特征点根据其在自车坐标系中的位置，“散布”并聚合（例如 Max Pooling）到预定义的 BEV 体素网格中，生成 BEV 特征图。
6.  **BEV 编码与检测 (`BEVEncoder_BEVE`)**:
    *   使用卷积神经网络对 BEV 特征图进行进一步编码。
    *   通过分离的检测头预测目标的类别、3D 边界框参数（位置、尺寸、朝向、速度）和 IoU（可选）。

## 3. 配置参数

模型行为受以下配置字典控制：

*   **`grid_conf` (字典):** 定义 BEV 网格和深度离散化。
    *   `xbound`: `[xmin, xmax, x_resolution]` BEV 网格 X 轴范围和分辨率（单位：米）。
    *   `ybound`: `[ymin, ymax, y_resolution]` BEV 网格 Y 轴范围和分辨率（单位：米）。
    *   `zbound`: `[zmin, zmax, z_resolution]` 体素高度范围和分辨率（用于 `voxel_pooling` 中间步骤）。
    *   `dbound`: `[dmin, dmax, d_step]` 相机深度轴离散化的范围和步长（单位：米），决定了深度维度 `D`。
*   **`data_aug_conf` (字典):** 定义数据处理和增强相关参数。
    *   `final_dim`: `(H, W)` 输入网络图像的最终尺寸。用于 `create_frustum` 计算像素坐标。
    *   `...`: 可能包含其他数据增强相关的配置（如 `resize_lim`, `rot_lim` 等，虽然在模型内部不直接使用，但在数据加载时可能用到）。

## 4. 核心模块详解

### 4.1 `BEVENet`

*   **目的**: 顶层模块，整合整个处理流程。
*   **初始化参数 (`__init__`)**:
    *   `grid_conf`: BEV 网格配置。
    *   `data_aug_conf`: 数据增强/图像配置。
    *   `outC`: 理论上的 BEV 编码器输出通道，但实际输出由 `BEVEncoder_BEVE` 内部决定。
    *   `num_classes`: 目标检测的类别数。
    *   `detection_head`: (当前未使用) 可能用于控制是否构建检测头。
    *   `model_type`: 在 `compile_model` 中用于选择此模型。
*   **主要子模块**:
    *   `camencode (CamEncode_rep)`: 相机编码器。
    *   `depth_net (DepthNet)`: 深度预测网络。
    *   `bevencode (BEVEncoder_BEVE)`: BEV 编码和检测头。
*   **前向传播 (`forward`)**:
    *   **输入**:
        *   `x`: 图像张量 `[B, N, 3, H, W]`。
        *   `rots`: 旋转矩阵 `[B, N, 3, 3]`。
        *   `trans`: 平移向量 `[B, N, 3]`。
        *   `intrins`: 内参矩阵 `[B, N, 3, 3]`。
        *   `post_rots`: 后处理旋转 `[B, N, 3, 3]`。
        *   `post_trans`: 后处理平移 `[B, N, 3]`。
    *   **处理**: 调用 `get_voxels` 完成视图转换和特征提取，得到 BEV 特征图；然后调用 `bevencode` 对 BEV 特征进行编码并生成预测。
    *   **输出**: 包含预测结果的字典 `preds = {'cls_pred': ..., 'reg_pred': ..., 'iou_pred': ...}`。

### 4.2 `CamEncode_rep`

*   **目的**: 提取每个相机图像的特征，并初步分离深度和上下文信息。
*   **初始化参数 (`__init__`)**:
    *   `D`: 深度维度，由 `grid_conf['dbound']` 决定。
    *   `C`: 输出的上下文特征通道数 (`self.camC`)。
*   **主要子模块**:
    *   `trunk (CamEncoder)`: 基于 RepViT 等模块构建的图像特征提取主干。
    *   `depthnet (Gencov)`: 一个卷积层，用于从主干特征中预测混合的深度 logits 和上下文特征。
*   **核心逻辑 (`get_depth_feat`)**:
    1.  使用 `trunk` 提取特征。
    2.  使用 `depthnet` 预测 `D+C` 通道的混合特征。
    3.  分离前 `D` 通道作为深度 logits，后 `C` 通道作为上下文特征。
    4.  对深度 logits 应用 `softmax` 得到深度概率 `depth`。
    5.  使用 `depth` 概率对上下文特征 `context` 进行加权，得到 `new_x`。
*   **前向传播 (`forward`)**: 实际调用 `get_depth_feat`。
*   **输出**: 返回 `depth` 和 `new_x` (加权的上下文特征)。在 `BEVENet` 中，实际使用的是 `new_x` (或类似经过深度加权的特征) 送入后续模块。

### 4.3 `DepthNet`

*   **目的**: 结合相机几何参数（内外参）显式地预测深度，并增强上下文特征。
*   **初始化参数 (`__init__`)**:
    *   `in_channels`: 输入特征通道数 (来自 `camencode` 的输出，通常是 `self.camC`)。
    *   `mid_channels`: 内部卷积层的通道数。
    *   `context_channels`: 输出的上下文特征通道数 (`self.camC`)。
    *   `depth_channels`: 输出的深度 logits 通道数 (`self.D`)。
*   **主要子模块**:
    *   `reduce_conv`: 初始卷积层。
    *   `bn`: 对展平的相机参数进行批归一化。
    *   `depth_mlp`, `context_mlp`: 从相机参数生成特征引导。
    *   `depth_se`, `context_se`: SE (Squeeze-and-Excitation) 层，利用引导信息调整特征通道的权重。
    *   `depth_conv`, `context_conv2`: 最终的卷积头，分别输出深度 logits 和上下文特征。
*   **前向传播 (`forward`)**:
    *   **输入**:
        *   `x`: 来自 `camencode` (或其变体) 的 2D 图像特征 `[B*N, C_in, H, W]`。
        *   `mats_dict`: 包含 `sensor2ego_mats`, `intrin_mats` 等相机参数的字典。
    *   **处理**:
        1.  对输入特征 `x` 应用 `reduce_conv`。
        2.  从 `mats_dict` 提取、展平并归一化相机参数。
        3.  使用 MLP 生成 `depth_guide` 和 `context_guide`。
        4.  使用 SE 层根据 `guide` 和 `reduced_feat` 计算注意力权重。
        5.  将注意力权重应用到 `reduced_feat`。
        6.  通过最终的卷积头得到 `depth_digit` 和 `context`。
    *   **输出**: 拼接后的深度 logits 和上下文特征 `[B*N, D+C, H, W]`。

### 4.4 `BEVEncoder_BEVE`

*   **目的**: 对 `voxel_pooling` 生成的 BEV 特征图进行编码，并通过不同的头进行目标检测预测。
*   **初始化参数 (`__init__`)**:
    *   `inC`: 输入 BEV 特征的通道数 (`self.camC`)。
    *   `outC`: (未使用) 理论上的输出通道，但实际由头部决定。
    *   `num_classes`: 目标类别数。
*   **主要子模块**:
    *   卷积层 (`layer1` - `layer4`): 基于 RepViTBlock, C2f, SPPF, PSA, C2fCIB 等构建的卷积主干，用于提取 BEV 特征。包含下采样和上采样操作。
    *   `shared_features`: 在送入检测头之前共享的特征提取层。
    *   `cls_head`: 分类头，输出 `num_classes` 通道的 logits。
    *   `reg_head`: 回归头，输出 9 通道的回归值 (x,y,z, w,l,h, sin(yaw), cos(yaw), vel)。
    *   `iou_head`: IoU 预测头，输出 1 通道的 logit。
*   **前向传播 (`forward`)**:
    *   **输入**: BEV 特征图 `[B, C, X, Y]`。
    *   **处理**: 通过卷积主干和共享特征层，然后将结果分别送入三个独立的检测头。
    *   **输出**: 包含三个预测张量的字典 `{'cls_pred': ..., 'reg_pred': ..., 'iou_pred': ...}`，每个张量的空间维度通常是输入 BEV 图的一半或四分之一 (`X'`, `Y'`)。

## 5. 关键辅助函数/概念

*   **`create_frustum(fH, fW)`**:
    *   根据特征图尺寸 (`fH`, `fW`) 和 `grid_conf['dbound']` (深度范围/步长) 生成一个表示相机视锥体的点云。
    *   每个点包含其在特征图上的 (x, y) 像素坐标、深度值 d 和齐次坐标 1，形状为 `[D, fH, fW, 4]`。
*   **`get_geometry(frustum, rots, trans, ...)`**:
    *   接收 `frustum` 点云和所有相机相关的变换参数。
    *   将 `frustum` 中的点从相机像素坐标系（结合深度）依次通过数据增强逆变换、相机内参逆变换、传感器到自车变换，最终转换到自车（ego）坐标系下的 3D 坐标。
    *   输出 `[B, N, D, fH, fW, 3]`。
*   **`voxel_pooling(geom_feats, x)`**:
    *   接收 `get_geometry` 输出的 3D 坐标 `geom_feats` 和对应的加权特征 `x` `[B, N, C, D, H, W]`。
    *   根据 `grid_conf` 定义的 BEV 网格，将 `geom_feats` 转换为离散的体素索引。
    *   过滤掉超出 BEV 网格范围的点。
    *   将有效特征 `x` 根据其对应的体素索引聚合到最终的 BEV 特征网格中 (使用 `scatter_add` / `index_put_` 实现)。
    *   通过在 Z 轴上进行 Max Pooling 来降维。
    *   输出 BEV 特征图 `[B, C, X, Y]`。

## 6. 输入与输出

*   **模型输入 (`BEVENet.forward`)**:
    *   `x`: `torch.Tensor` - 形状 `[B, N, 3, H, W]`，多相机图像。
    *   `rots`: `torch.Tensor` - 形状 `[B, N, 3, 3]`，旋转矩阵。
    *   `trans`: `torch.Tensor` - 形状 `[B, N, 3]`，平移向量。
    *   `intrins`: `torch.Tensor` - 形状 `[B, N, 3, 3]`，内参矩阵。
    *   `post_rots`: `torch.Tensor` - 形状 `[B, N, 3, 3]`，数据增强旋转。
    *   `post_trans`: `torch.Tensor` - 形状 `[B, N, 3]`，数据增强平移。
*   **模型输出 (`BEVENet.forward`)**:
    *   `preds`: `dict` - 包含模型预测结果的字典：
        *   `'cls_pred'`: `torch.Tensor` - 形状 `[B, num_classes, X', Y']`，分类 logits。
        *   `'reg_pred'`: `torch.Tensor` - 形状 `[B, 9, X', Y']`，回归预测值。
        *   `'iou_pred'`: `torch.Tensor` - 形状 `[B, 1, X', Y']`，IoU 预测 logits。

## 7. 使用示例 (概念性)

```python
import torch
from src.models import compile_model # 假设 compile_model 在这里

# 1. 定义配置 (示例)
grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-5.0, 3.0, 0.5], # Z范围影响 voxel_pooling 中间步骤
    'dbound': [4.0, 45.0, 1.0], # 深度范围和步长
}
data_aug_conf = {
    'final_dim': (128, 352), # 示例图像尺寸
    # ... 其他增强参数
}
num_classes = 10
# outC 参数在 BEVENet 中影响不大，但 compile_model 可能需要它
# 理论上它应该等于 num_classes + 9 (回归) + 1 (IoU) 或其他值
# 但 BEVEncoder_BEVE 内部决定了输出
compile_outC = num_classes + 9 + 1

# 2. 编译模型
model = compile_model(grid_conf, data_aug_conf, outC=compile_outC, model='beve', num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. 准备输入数据 (示例形状)
B, N = 2, 6 # 批次大小, 相机数量
H, W = data_aug_conf['final_dim']
dummy_imgs = torch.randn(B, N, 3, H, W).to(device)
dummy_rots = torch.randn(B, N, 3, 3).to(device)
dummy_trans = torch.randn(B, N, 3).to(device)
dummy_intrins = torch.randn(B, N, 3, 3).to(device)
dummy_post_rots = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).to(device)
dummy_post_trans = torch.zeros(B, N, 3).to(device)

# 4. 模型前向传播
model.eval() # 或 model.train()
with torch.no_grad(): # 如果是推理
    predictions = model(dummy_imgs, dummy_rots, dummy_trans, dummy_intrins, dummy_post_rots, dummy_post_trans)

# 5. 处理输出
cls_preds = predictions['cls_pred']
reg_preds = predictions['reg_pred']
iou_preds = predictions['iou_pred']

print("Classification Preds Shape:", cls_preds.shape)
print("Regression Preds Shape:", reg_preds.shape)
print("IoU Preds Shape:", iou_preds.shape)

```

## 8. 依赖项

*   **PyTorch**: 核心深度学习框架。
*   **自定义模块**:
    *   `src.tools`: 包含 `gen_dx_bx`, `cumsum_trick` 等辅助函数。
    *   `utils.RepVit`, `utils.common`: 包含 `RepViTBlock`, `EMA`, `PSA`, `SPPELAN`, `Conv`, `Gencov`, `C2fCIB`, `C2f`, `SPPF` 等自定义网络层。

**注意**: 文档基于提供的 `src/models.py` 代码。具体的层实现（如 `RepViTBlock`）和辅助函数（如 `cumsum_trick`）的细节需要参考对应的源文件。