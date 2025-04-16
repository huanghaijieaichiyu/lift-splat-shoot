# Lift-Splat-Shoot增强版：集成语义分割与3D目标检测

本项目是基于ECCV 2020的论文[Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)的扩展实现，增加了完整的3D目标检测功能以提升自动驾驶感知能力。

## 项目特点

- **原始LSS功能**：支持原论文中提出的BEV语义分割功能
- **3D目标检测**：新增完整的3D目标检测功能，能够预测物体的位置、尺寸、朝向和类别
- **统一框架**：在一个框架中集成了BEV语义分割和3D目标检测
- **混合精度训练**：支持PyTorch混合精度训练，提高训练效率
- **全面评估**：提供详细的mAP评估功能，支持多IoU阈值和类别分析

## 效果展示

### 语义分割效果
<img src="./imgs/eval.gif" width="600">

### 数据投影检查
<img src="./imgs/check.gif" width="600">

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖项：
- PyTorch
- nuscenes-devkit
- tensorboardX
- efficientnet_pytorch==0.7.0

## 数据准备

从[nuScenes官方网站](https://www.nuscenes.org/)下载数据集，然后在运行命令时更新`dataroot`参数。

## 功能使用

### 语义分割（原始LSS方法）

训练语义分割模型：
```bash
python main.py train \
        --dataroot="/data/nuscenes" \
        --nepochs=10000 \
        --gpuid=0 \
        --H=900 \
        --W=1600 \
        --train_step=5 \
        --bsz=4
```

评估语义分割性能：
```bash
python main.py eval_model_iou mini --modelf=runs/train/best.pt --dataroot=/path/to/nuscenes
```

可视化分割结果：
```bash
python main.py viz_model_preds mini --modelf=runs/train/best.pt --dataroot=/path/to/nuscenes
```

### 3D目标检测（新增功能）

训练3D目标检测模型：
```bash
python main.py train_3d_detection \
        --dataroot="/data/nuscenes" \
        --nepochs=10000 \
        --gpuid=0 \
        --H=900 \
        --W=1600 \
        --bsz=4 \
        --num_classes=10
```

评估3D目标检测性能：
```bash
python main.py eval_3d_detection \
        --dataroot="/data/nuscenes" \
        --checkpoint_path="./runs_3d/exp_000/best.pt" \
        --gpuid=0
```

可视化3D检测结果：
```bash
python main.py viz_3d_detection \
        --dataroot="/data/nuscenes" \
        --checkpoint_path="./runs_3d/exp_000/best.pt" \
        --gpuid=0
```

## 3D目标检测模块说明

我们的3D目标检测模块扩展了原始LSS架构，主要改进包括：

1. **检测头设计**：增加了多任务检测头，同时预测类别、3D边界框和置信度
2. **IoU感知学习**：添加IoU预测分支，提高检测质量和置信度校准
3. **高效BEV特征提取**：保留了原始LSS的BEV特征提取能力，确保高效的特征表示
4. **完整评估系统**：实现了基于mAP的详细评估系统，支持多IoU阈值和类别分析

## 评估指标

3D目标检测模块使用以下指标进行评估：
- **mAP@0.5, mAP@0.7**：在不同IoU阈值下的平均精度
- **类别AP**：每个类别的平均精度
- **精度/召回率曲线**：评估模型在不同阈值下的性能

## 创新点

- **端到端训练**：从多视角图像直接预测3D目标，无需额外的点云输入
- **空间敏感检测头**：设计了考虑BEV空间特性的检测头，提高小目标检测能力
- **高效实现**：优化了计算流程，支持高效训练和推理

## 联系方式

如果您对代码或项目有任何疑问，请提交[issue](https://github.com/huangxiaohaiaichiyu/lift-splat-shoot/issues)。

## 致谢

本项目基于NVIDIA研究院的[Lift-Splat-Shoot](https://github.com/nv-tlabs/lift-splat-shoot)项目。感谢原作者Jonah Philion和Sanja Fidler的开创性工作。如果您使用了本项目的代码，请同时引用原始论文。

## 闲谈
Q1 为什么2025年了，还用2020年的老框架？
A：因为LSS真的项目结构太清晰了，而且训练速度、内存处理等都很顶级！并且没有臃肿的某些3D检测库，对用着8G内存GPU的学生党太友好了！
Q2 。。。。。
