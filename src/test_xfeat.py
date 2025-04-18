import torch
import argparse
import os
from .train import train_fusion
from .models.XFeatNet import create_xfeat_bevnet
from .models.xfeat_cam_encoder import XFeatCamEncoder, XFeatBEVEncoder


def parse_args():
    parser = argparse.ArgumentParser(description='测试XFeat模型')
    parser.add_argument('--version', type=str, default='mini',
                        help='数据集版本 (mini或full)')
    parser.add_argument('--dataroot', type=str, default='/data/nuscenes',
                        help='数据根目录')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='使用的GPU ID')
    parser.add_argument('--model_type', type=str, default='xfeat',
                        choices=['xfeat', 'fusion'],
                        help='使用的模型类型')
    parser.add_argument('--load_weight', type=str, default='',
                        help='加载权重文件路径')
    parser.add_argument('--logdir', type=str, default='./runs_xfeat',
                        help='日志目录')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅执行评估')
    parser.add_argument('--amp', action='store_true',
                        help='使用自动混合精度')
    return parser.parse_args()


def test_xfeat_encoder():
    """测试XFeatCamEncoder组件"""
    # 创建随机输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 352)

    # 创建编码器
    D = 41  # 深度维度，与frustum中的D相同
    C = 64  # 输出通道数
    encoder = XFeatCamEncoder(D, C, downsample=16)

    # 前向传播
    print(f"输入形状: {x.shape}")
    try:
        with torch.no_grad():
            features = encoder(x)
            print(f"输出形状: {features.shape}")
            print(
                f"特征值范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
            print(f"NaN值数量: {torch.isnan(features).sum().item()}")
            print(f"Inf值数量: {torch.isinf(features).sum().item()}")
            print("XFeatCamEncoder测试通过！")
    except Exception as e:
        print(f"XFeatCamEncoder测试失败: {e}")


def test_xfeat_bev_encoder():
    """测试XFeatBEVEncoder组件"""
    # 创建随机输入
    batch_size = 2
    inC = 64  # 输入通道数
    H, W = 200, 200  # 特征图高度和宽度
    x = torch.randn(batch_size, inC, H, W)

    # 创建编码器
    outC = 9 * 10  # 9个回归参数 x 10个类别
    encoder = XFeatBEVEncoder(inC, outC)

    # 前向传播
    print(f"输入形状: {x.shape}")
    try:
        with torch.no_grad():
            features = encoder(x)
            print(f"输出形状: {features.shape}")
            print(
                f"特征值范围: [{features.min().item():.4f}, {features.max().item():.4f}]")
            print(f"NaN值数量: {torch.isnan(features).sum().item()}")
            print(f"Inf值数量: {torch.isinf(features).sum().item()}")
            print("XFeatBEVEncoder测试通过！")
    except Exception as e:
        print(f"XFeatBEVEncoder测试失败: {e}")


def main():
    args = parse_args()

    print("=== 测试XFeat模型组件 ===")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpuid}')
        print(f"使用GPU: {torch.cuda.get_device_name(args.gpuid)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")

    # 测试编码器组件
    test_xfeat_encoder()
    test_xfeat_bev_encoder()

    # 测试完整模型
    print("=== 测试完整的XFeat模型 ===")

    # 设置训练/评估参数
    model_params = {
        'version': args.version,
        'dataroot': args.dataroot,
        'nepochs': 1 if not args.eval_only else 0,  # 如果仅评估则设为0
        'gpuid': args.gpuid,
        'cuDNN': True,  # 启用cuDNN加速
        'load_weight': args.load_weight,
        'amp': args.amp,  # 是否使用自动混合精度
        'logdir': args.logdir,
        'model_type': args.model_type,  # 使用XFeat模型
        'eval_only': args.eval_only,  # 仅评估
        'bsz': 2,  # 较小的批量大小
        'nworkers': 2,  # 数据加载线程数
        'lr': 1e-4,  # 较小的学习率
        'weight_decay': 1e-6,  # 权重衰减
        'num_classes': 10,  # 类别数
        'enable_multiscale': True,  # 启用多尺度特征
        'use_enhanced_fusion': True,  # 启用增强的特征融合
    }

    # 运行训练/评估
    try:
        best_map = train_fusion(**model_params)
        print(f"最佳mAP: {best_map:.4f}")
    except Exception as e:
        print(f"模型训练/评估失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
