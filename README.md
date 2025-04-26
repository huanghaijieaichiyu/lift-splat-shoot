# Lift-Splat-Shoot Pytorch

本仓库包含 [Lift-Splat-Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711) 论文的 PyTorch 实现。

## 使用 UV 进行项目设置

本项目使用 [uv](https://github.com/astral-sh/uv) 进行快速依赖管理和虚拟环境创建。

1.  **安装 `uv`:**
    请遵循 [官方 `uv` 安装指南](https://astral.sh/uv/install.sh) 中的说明进行安装。例如：
    ```bash
    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows (Powershell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
    或者，通过 pip 安装：`pip install uv`

2.  **克隆仓库:**
    ```bash
    git clone <your-repository-url> # 更新为你的仓库 URL
    cd lift-splat-shoot
    ```

3.  **创建虚拟环境:**
    ```bash
    uv venv
    ```
    这将在你的项目根目录下创建一个名为 `.venv` 的虚拟环境。

4.  **激活虚拟环境:**
    *   macOS / Linux: `source .venv/bin/activate`
    *   Windows (PowerShell): `.venv\\Scripts\\Activate.ps1`
    *   Windows (CMD): `.venv\\Scripts\\activate.bat`

5.  **(首次或更新依赖时) 生成锁定文件:**
    将 `pyproject.toml` 文件中指定的依赖项编译成一个锁定文件 (`requirements.lock`)。这能确保可复现的构建。
    ```bash
    # 将 cuXXX 替换为你的 CUDA 版本 (例如, cu118, cu121) 或使用 'cpu'
    # 查看 https://pytorch.org/get-started/locally/ 获取正确的索引 URL
    uv pip compile pyproject.toml -o requirements.lock --extra-index-url https://download.pytorch.org/whl/cu118 
    ```
    *   **重要:** 请根据你系统的 CUDA 版本 (例如 `cu118`, `cu121`, `cu124`) 调整 `--extra-index-url`，或者如果你没有支持 CUDA 的 GPU，请使用仅 CPU 的索引 (`https://download.pytorch.org/whl/cpu`)。请参阅 [PyTorch 安装指南](https://pytorch.org/get-started/locally/) 获取适合你配置的正确 URL。
    *   将生成的 `requirements.lock` 文件提交到你的仓库。

6.  **安装依赖:**
    使用锁定文件同步你的虚拟环境。
    ```bash
    uv pip sync requirements.lock
    ```
    当 `requirements.lock` 文件发生更改时（例如，在拉取更新或重新生成它之后），再次运行 `uv pip sync requirements.lock`。

## 训练

(在此添加如何运行训练脚本的说明)

示例:
```bash
python src/train.py --version mini --dataroot /path/to/nuscenes --gpuid 0 ...
```

## 评估

(在此添加如何运行评估的说明)

## 贡献

(如果适用，请在此添加贡献指南)

## 许可证

本项目采用 MIT 许可证授权 - 详情请参阅 LICENSE 文件。(如果你还没有 LICENSE 文件，可以考虑创建一个)。

## 项目特点

- **原始LSS功能**：支持原论文中提出的BEV语义分割功能
- **3D目标检测**：新增完整的3D目标检测功能，能够预测物体的位置、尺寸、朝向和类别
- **统一框架**：在一个框架中集成了BEV语义分割和3D目标检测
- **混合精度训练**：支持PyTorch混合精度训练，提高训练效率
- **全面评估**：提供详细的mAP评估功能，支持多IoU阈值和类别分析

## 主要改进与增强

本项目在原始 LSS 的基础上进行了大量改进和增强，以提高模型的性能、灵活性和鲁棒性。主要包括：

*   **灵活的骨干网络选择** (如 EfficientNet, MobileNetV3)
*   **先进的 3D 检测头** (多任务学习, FPN 结构)
*   **优化的 BEV 投影与体素处理** (借鉴 BEVDepth)
*   **改进的损失函数** (如 DIoU/CIoU, Focal Loss 修正)
*   **增强的训练/评估流程** (距离 NMS, mAP 问题修复, 稳定性提升)
*   **更准确的数据处理** (nuScenes 标签修正)
*   **增强的可视化与调试能力** (TensorBoard 集成)

更详细的改进内容请参考 [IMPROVEMENTS.md](IMPROVEMENTS.md)。

## 效果展示

### 语义分割效果
<img src="./imgs/eval.gif" width="600">

### 数据投影检查
<img src="./imgs/check.gif" width="600">

## 安装依赖

**注意:** 以下 `pip install -r requirements.txt` 的说明现在已被上面新的 `uv` 工作流取代。主要依赖项列表仍然相关。

```bash
# 旧的安装方式 (已被 UV 取代)
# pip install -r requirements.txt 
```

主要依赖项 (通过 `uv` 管理):
- PyTorch (torch, torchvision, torchaudio)
- nuscenes-devkit
- tensorboardX
- numpy
- tqdm
- pillow
- opencv-python-headless
- efficientnet_pytorch==0.7.0 # 需要确认是否仍需此特定版本，若需要，请添加到 pyproject.toml

## 数据准备

从[nuScenes官方网站](https://www.nuscenes.org/)下载数据集，然后在运行命令时更新`dataroot`参数。

## 功能使用

### 语义分割（原始LSS方法）

训练语义分割模型：
```bash
# 激活你的 uv 虚拟环境
# source .venv/bin/activate  (Linux/macOS)
# .\.venv\Scripts\Activate.ps1 (Windows PowerShell)

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
# 激活你的 uv 虚拟环境

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
