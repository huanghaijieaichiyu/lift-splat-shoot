import torch
from torch import nn
from torchvision.models.resnet import resnet18
import timm
import torch.nn.functional as F
import os

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum

# 添加EfficientNet的条件导入
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("警告: efficientnet_pytorch 未安装。如需使用 CamEncode 类，请先安装: pip install efficientnet-pytorch")
    # 创建一个空的EfficientNet类，以避免导入错误

    class EfficientNet:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError(
                "未安装 efficientnet_pytorch 模块。请使用 pip install efficientnet-pytorch 安装。")


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(
            512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(
            1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH //
                   self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # --- 检查 cumsum_trick 的输入 ---
        assert torch.all(torch.isfinite(
            x)), "!!! NaN/Inf detected in input x to cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            geom_feats)), "!!! NaN/Inf detected in input geom_feats to cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            ranks)), "!!! NaN/Inf detected in input ranks to cumsum_trick !!!"
        # -----------------------------

        # Use cumsum trick for summing features in the same voxel
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # --- 检查 cumsum_trick 的输出 ---
        assert torch.all(torch.isfinite(
            x)), "!!! NaN/Inf detected in output x from cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            geom_feats)), "!!! NaN/Inf detected in output geom_feats from cumsum_trick !!!"
        # ----------------------------

        # --- 恢复缺失的代码块 ---
        # Create the final BEV grid (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[0].item()), int(self.nx[1].item())), device=x.device)
        # Scatter summed features into the grid
        # Indices: batch, channel, z, x, y
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z by concatenation: B x (C * Z) x X x Y
        final = torch.cat(final.unbind(dim=2), 1)
        # --- 代码块恢复结束 ---

        # --- 检查最终的 final 张量 ---
        assert torch.all(torch.isfinite(
            final)), "!!! NaN/Inf detected in the final output tensor of voxel_pooling !!!"
        # --------------------------

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


class CamEncodeFPN(nn.Module):
    def __init__(self, D, C, fpn_out_channels=256):
        super(CamEncodeFPN, self).__init__()
        self.D = D
        self.C = C
        self.fpn_out_channels = fpn_out_channels

        # --- 使用 RegNetY-400MF 作为 Backbone ---
        self.trunk = timm.create_model(
            'regnety_004',
            pretrained=False,  # 确保不加载任何预训练权重
            features_only=True,
            out_indices=[1, 2, 3, 4]
        )

        # 在模型创建后手动加载并处理权重
        checkpoint_path = 'weights/pytorch_model.bin'
        if os.path.exists(checkpoint_path):
            try:
                state_dict = torch.load(checkpoint_path, map_location='cpu')

                # 处理不同格式的检查点文件
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']

                # 获取 trunk 模型期望的 state_dict 键
                trunk_keys = self.trunk.state_dict().keys()

                # 过滤掉不属于 trunk 的键，并处理 'module.' 前缀
                filtered_state_dict = {}
                ignored_keys_count = 0
                loaded_keys_count = 0
                for k, v in state_dict.items():
                    # 检查原始键或移除 'module.' 前缀后的键是否存在于 trunk 中
                    plain_k = k.replace('module.', '', 1)
                    if plain_k in trunk_keys:
                        filtered_state_dict[plain_k] = v
                        loaded_keys_count += 1
                    else:
                        ignored_keys_count += 1

                # 使用strict=True加载过滤后的权重，因为我们只保留了期望的键
                missing_keys, unexpected_keys = self.trunk.load_state_dict(
                    filtered_state_dict, strict=True)

                print(
                    f"权重加载: 加载 {loaded_keys_count} 个键, 忽略 {ignored_keys_count} 个键.")
                if missing_keys or unexpected_keys:
                    print(
                        f"  警告: 加载权重后发现缺失或意外的键。缺失: {missing_keys}, 意外: {unexpected_keys}")

            except Exception as e:
                print(f"加载权重时出错: {e}")
                print(f"将使用随机初始化的权重继续")
        else:
            print(f"警告: 在{checkpoint_path}未找到检查点文件。将使用随机初始化的权重。")

        # --- 获取特征通道信息 ---
        feature_info = self.trunk.feature_info.get_dicts(
            keys=['num_chs', 'reduction'])
        if len(feature_info) < 4:
            raise ValueError(
                f"timm模型'regnety_004'未返回预期数量的特征阶段 (预期4个, 得到{len(feature_info)}个)")

        fpn_channels = [info['num_chs'] for info in feature_info]

        # --- FPN侧向连接 ---
        self.lat_c2 = nn.Conv2d(
            fpn_channels[0], self.fpn_out_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(
            fpn_channels[1], self.fpn_out_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(
            fpn_channels[2], self.fpn_out_channels, kernel_size=1)
        self.lat_c5 = nn.Conv2d(
            fpn_channels[3], self.fpn_out_channels, kernel_size=1)

        # --- FPN自顶向下路径 ---
        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, self.fpn_out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_out_channels),
            nn.ReLU(inplace=True)
        )

        # --- 改进的深度头 (输出不确定性, 从P2输入) ---
        depth_intermediate_channels = self.fpn_out_channels // 2
        self.depthnet = nn.Sequential(
            nn.Conv2d(self.fpn_out_channels, depth_intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_intermediate_channels, depth_intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_intermediate_channels, 2 *
                      self.D + self.C, kernel_size=1, padding=0)
        )

        # --- 添加深度细化模块 (Input from P2) ---
        self.refinement_net = DepthRefinementNetV2(
            D, self.fpn_out_channels, intermediate_channels=fpn_out_channels // 2)

    def get_depth_dist(self, depth_logits, eps=1e-20):
        # 检查 softmax 的输入
        if not torch.all(torch.isfinite(depth_logits)):
            print("!!! NaN/Inf detected in input logits to get_depth_dist (softmax) !!!")
            # 这里可以考虑直接引发错误以停止执行
            # raise ValueError("NaN/Inf detected in input to softmax")
        return depth_logits.softmax(dim=1)

    def forward(self, x):
        # 检查 CamEncodeFPN 的输入
        assert torch.all(torch.isfinite(
            x)), "NaN/Inf detected in input x to CamEncodeFPN forward"

        # --- Backbone Feature Extraction ---
        features = self.trunk(x)
        if len(features) != 4:
            raise ValueError(
                f"Expected 4 feature maps from backbone, got {len(features)}")
        c2, c3, c4, c5 = features
        # 检查骨干网输出的关键部分
        assert torch.all(torch.isfinite(
            c2)), "NaN/Inf detected in c2 from backbone"
        assert torch.all(torch.isfinite(
            c5)), "NaN/Inf detected in c5 from backbone"

        # --- FPN Lateral Connections and Top-Down Pathway ---
        # 逐层检查 FPN 输出
        p5 = self.lat_c5(c5)
        assert torch.all(torch.isfinite(p5)), "NaN/Inf detected in p5"
        p4_in = self.lat_c4(
            c4) + F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=True)
        assert torch.all(torch.isfinite(p4_in)), "NaN/Inf detected in p4_in"
        p4 = self.smooth_p4(p4_in)
        assert torch.all(torch.isfinite(p4)), "NaN/Inf detected in p4"

        p3_in = self.lat_c3(
            c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=True)
        assert torch.all(torch.isfinite(p3_in)), "NaN/Inf detected in p3_in"
        p3 = self.smooth_p3(p3_in)
        assert torch.all(torch.isfinite(p3)), "NaN/Inf detected in p3"

        p2_in = self.lat_c2(
            c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=True)
        assert torch.all(torch.isfinite(p2_in)), "NaN/Inf detected in p2_in"
        p2 = self.smooth_p2(p2_in)
        assert torch.all(torch.isfinite(p2)), "NaN/Inf detected in p2"

        # --- 设定 log_variance 的安全范围 ---
        log_var_min = -10.0
        log_var_max = 10.0

        # --- Initial Depth, Uncertainty, and Feature Calculation from P2 ---
        output = self.depthnet(p2)
        # --- 检查 depthnet 的整体输出 ---
        assert torch.all(torch.isfinite(
            output)), "!!! NaN/Inf detected immediately after self.depthnet(p2) call !!!"
        # ---------------------------------
        initial_depth_logits = output[:, :self.D]
        initial_depth_log_variance_raw = output[:, self.D:2*self.D]
        context_features = output[:, 2*self.D:]
        # --- 检查 depthnet 输出的分割部分 ---
        assert torch.all(torch.isfinite(initial_depth_logits)
                         ), "NaN/Inf detected in initial_depth_logits split from depthnet output"
        assert torch.all(torch.isfinite(initial_depth_log_variance_raw)
                         ), "NaN/Inf detected in initial_depth_log_variance_raw split from depthnet output"
        assert torch.all(torch.isfinite(
            context_features)), "NaN/Inf detected in context_features split from depthnet output"
        # ----------------------------------

        # --- 稳定化: Clamp 初始 log_variance ---
        initial_depth_log_variance = torch.clamp(
            initial_depth_log_variance_raw, min=log_var_min, max=log_var_max
        )
        assert torch.all(torch.isfinite(initial_depth_log_variance)
                         ), "NaN/Inf detected after clamping initial_depth_log_variance"

        # --- Depth Refinement using P2 features ---
        # 检查 refinement_net 的输入
        assert torch.all(torch.isfinite(initial_depth_logits.detach(
        ))), "NaN/Inf in input initial_logits.detach() to refinement_net"
        assert torch.all(torch.isfinite(initial_depth_log_variance.detach(
        ))), "NaN/Inf in input initial_depth_log_variance.detach() to refinement_net"
        assert torch.all(torch.isfinite(
            p2)), "NaN/Inf in input p2 to refinement_net"

        refined_depth_logits_raw, refined_depth_log_variance_raw = self.refinement_net(
            initial_depth_logits.detach(),
            initial_depth_log_variance.detach(),  # 使用 clamp 后的值 detach
            p2
        )
        # --- 检查 refinement_net 的输出 ---
        assert torch.all(torch.isfinite(refined_depth_logits_raw)
                         ), "!!! NaN/Inf detected in refined_depth_logits_raw output from refinement_net !!!"
        assert torch.all(torch.isfinite(refined_depth_log_variance_raw)
                         ), "!!! NaN/Inf detected in refined_depth_log_variance_raw output from refinement_net !!!"
        # ---------------------------------

        # --- 稳定化: Clamp 细化后的 log_variance ---
        refined_depth_logits = refined_depth_logits_raw  # 使用 refinement_net 的原始 logits 输出
        refined_depth_log_variance = torch.clamp(
            refined_depth_log_variance_raw, min=log_var_min, max=log_var_max
        )
        assert torch.all(torch.isfinite(refined_depth_logits)
                         ), "NaN/Inf detected in refined_depth_logits before softmax"
        assert torch.all(torch.isfinite(refined_depth_log_variance)
                         ), "NaN/Inf detected after clamping refined_depth_log_variance"

        # --- Final Feature Combination using Refined Depth ---
        refined_depth_prob = self.get_depth_dist(refined_depth_logits)
        assert torch.all(torch.isfinite(refined_depth_prob)
                         ), "NaN/Inf detected in refined_depth_prob after softmax"

        refined_confidence = torch.exp(-refined_depth_log_variance)
        assert torch.all(torch.isfinite(refined_confidence)
                         ), "NaN/Inf detected in refined_confidence after exp"

        # 检查最终乘法的输入
        assert torch.all(torch.isfinite(refined_depth_prob.unsqueeze(
            1))), "NaN/Inf in input 1 (prob) to final multiplication"
        assert torch.all(torch.isfinite(refined_confidence.unsqueeze(
            1))), "NaN/Inf in input 2 (conf) to final multiplication"
        assert torch.all(torch.isfinite(context_features.unsqueeze(
            2))), "NaN/Inf in input 3 (feat) to final multiplication"

        epsilon = 1e-6
        new_x = refined_depth_prob.unsqueeze(
            1) * refined_confidence.unsqueeze(1) * context_features.unsqueeze(2) + epsilon
        # 检查最终输出 new_x
        assert torch.all(torch.isfinite(
            new_x)), "!!! NaN/Inf detected in the final new_x output of CamEncodeFPN !!!"

        return new_x, refined_depth_prob


class LidarEncode(nn.Module):
    def __init__(self, inC_lidar, outC_lidar):
        super().__init__()
        # Example: A simple ConvNet encoder for LiDAR BEV features
        self.encoder = nn.Sequential(
            nn.Conv2d(inC_lidar, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC_lidar, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC_lidar),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class FusionNet(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, lidar_inC=1, lidar_enc_out_channels=128, fused_bev_channels=256):
        super(FusionNet, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        # --- Grid and Frustum Setup (Moved from LiftSplatShoot) ---
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        # Ensure nx are integers for grid creation
        nx = nx.long()
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        # Determine frustum based on FPN output stride
        # Use stride 4 (from p2) to match get_cam_feats output
        self.feature_map_stride = 4
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        # --- Grid and Frustum End ---

        # --- Camera Branch Components ---
        fpn_feature_channels = 256  # Corresponds to FPN output channels
        context_channels_per_depth_bin = 64  # Desired C per depth bin for voxel pooling
        self.camC = context_channels_per_depth_bin  # Channels per depth bin feature
        self.camencode = CamEncodeFPN(
            self.D, self.camC, fpn_out_channels=fpn_feature_channels)
        # --- Camera Branch End ---

        # --- LiDAR Branch Components ---
        self.lidar_enc_out_channels = lidar_enc_out_channels
        self.lidarencode = LidarEncode(lidar_inC, self.lidar_enc_out_channels)
        # --- LiDAR Branch End ---

        # --- Fusion and BEV Encoding ---
        num_z_bins = self.nx[2].item()  # Get integer Z dimension
        # Input channels for camera BEV processor after Z-collapse (concatenation)
        cam_bev_interim_channels = self.camC * num_z_bins

        # Camera BEV Processor (using BevEncode architecture)
        self.cam_bev_processor = BevEncode(
            inC=cam_bev_interim_channels, outC=fused_bev_channels)

        # Fusion layer: combines processed camera BEV and processed lidar BEV
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fused_bev_channels + self.lidar_enc_out_channels,
                      fused_bev_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fused_bev_channels),
            nn.ReLU(inplace=True),
            # Final output layer
            nn.Conv2d(fused_bev_channels, outC, kernel_size=1, padding=0)
        )
        # --- Fusion and BEV End ---

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # Use self.feature_map_stride determined in __init__
        fH, fW = ogfH // self.feature_map_stride, ogfW // self.feature_map_stride
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H_feat x W_feat x 3
        """
        B, N, _ = trans.shape

        # --- 检查输入 ---
        assert torch.all(torch.isfinite(self.frustum)
                         ), "NaN/Inf in self.frustum"
        assert torch.all(torch.isfinite(post_trans)
                         ), "NaN/Inf in input post_trans"
        assert torch.all(torch.isfinite(post_rots)
                         ), "NaN/Inf in input post_rots"
        assert torch.all(torch.isfinite(rots)), "NaN/Inf in input rots"
        assert torch.all(torch.isfinite(intrins)), "NaN/Inf in input intrins"
        assert torch.all(torch.isfinite(trans)), "NaN/Inf in input trans"
        # ---------------

        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        assert torch.all(torch.isfinite(points)
                         ), "NaN/Inf after frustum - post_trans"

        # --- 检查 post_rots 求逆 ---
        try:
            inv_post_rots = torch.inverse(post_rots)
            assert torch.all(torch.isfinite(
                inv_post_rots)), "!!! NaN/Inf detected in torch.inverse(post_rots) !!!"
        except Exception as e:
            print(f"Error during torch.inverse(post_rots): {e}")
            raise e  # 重新抛出异常以停止
        # --------------------------
        points = inv_post_rots.view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        assert torch.all(torch.isfinite(points)
                         ), "NaN/Inf after inv_post_rots matmul"

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        assert torch.all(torch.isfinite(
            points)), "NaN/Inf after points[:,:,:,:,:,2:3] multiplication"

        # --- 检查 intrins 求逆 ---
        try:
            inv_intrins = torch.inverse(intrins)
            assert torch.all(torch.isfinite(
                inv_intrins)), "!!! NaN/Inf detected in torch.inverse(intrins) !!!"
        except Exception as e:
            print(f"Error during torch.inverse(intrins): {e}")
            raise e  # 重新抛出异常以停止
        # --------------------------
        combine = rots.matmul(inv_intrins)
        assert torch.all(torch.isfinite(combine)
                         ), "NaN/Inf after rots.matmul(inv_intrins)"

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        assert torch.all(torch.isfinite(points)
                         ), "NaN/Inf after combine matmul"

        points += trans.view(B, N, 1, 1, 1, 3)
        # --- 检查最终输出 ---
        assert torch.all(torch.isfinite(
            points)), "!!! NaN/Inf detected in final points output of get_geometry !!!"
        # ------------------

        return points

    def get_cam_feats(self, x):
        """Return features: B x N x D x H_feat x W_feat x C_feat (camC)
                 depth_prob: B x N x D x H_feat x W_feat
        """
        B, N, C_img, imH, imW = x.shape
        x_batch_n = x.view(B*N, C_img, imH, imW)

        # CamEncodeFPN returns (features, depth_prob)
        # Features shape: (B*N, C_feat, D, H_feat, W_feat) where C_feat = self.camC
        # Depth prob shape: (B*N, D_feat, H_feat, W_feat)
        features_bn, depth_prob_bn = self.camencode(x_batch_n)

        # Reshape and permute features for voxel_pooling
        _, C_feat, D_feat, H_feat, W_feat = features_bn.shape
        assert D_feat == self.D, f"Depth dim mismatch: CamEncode output {D_feat} vs Frustum {self.D}"
        assert C_feat == self.camC, f"Channel dim mismatch: CamEncode output {C_feat} vs self.camC {self.camC}"

        features_bn = features_bn.view(B, N, C_feat, D_feat, H_feat, W_feat)
        # Permute features to B x N x D x H x W x C for voxel_pooling
        features_permuted = features_bn.permute(0, 1, 3, 4, 5, 2)

        # Reshape depth probability
        depth_prob = depth_prob_bn.view(B, N, D_feat, H_feat, W_feat)

        return features_permuted, depth_prob  # 返回特征和深度概率

    def voxel_pooling(self, geom_feats, x):
        """Projects camera features into voxels.
        Args:
            geom_feats: Geometry tensor (B x N x D x H_feat x W_feat x 3)
            x: Camera features (B x N x D x H_feat x W_feat x C)
        Returns:
            BEV features (B x (C * Z) x X x Y)
        """
        # --- 检查输入 ---
        assert torch.all(torch.isfinite(
            x)), "NaN/Inf detected in input x to voxel_pooling"
        assert torch.all(torch.isfinite(
            geom_feats)), "!!! NaN/Inf detected in input geom_feats to voxel_pooling !!!"  # 重点检查这里
        # ---------------

        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        x = x.reshape(Nprime, C)
        assert torch.all(torch.isfinite(
            x)), "NaN/Inf after x.reshape in voxel_pooling"

        # flatten indices
        geom_feats_normalized = (geom_feats - (self.bx - self.dx/2.)) / self.dx
        assert torch.all(torch.isfinite(geom_feats_normalized)
                         ), "NaN/Inf after normalizing geom_feats"
        geom_feats_long = geom_feats_normalized.long()
        # 注意: long() 后的检查意义不大，因为 nan/inf 转 long 行为未定义
        # assert torch.all(torch.isfinite(geom_feats_long)), "NaN/Inf after geom_feats.long()"

        geom_feats = geom_feats_long.view(Nprime, 3)
        # assert torch.all(torch.isfinite(geom_feats)), "NaN/Inf after geom_feats.view" # .long() 可能产生奇怪值

        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # assert torch.all(torch.isfinite(geom_feats)), "NaN/Inf after cat batch_ix" # 检查意义不大

        # filter out points that are outside grid boundaries
        # 可以在这里检查 geom_feats 的值范围，但如果已经是 nan/inf 就晚了
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]
        assert torch.all(torch.isfinite(
            x)), "NaN/Inf in x after filtering (kept)"
        # assert torch.all(torch.isfinite(geom_feats)), "NaN/Inf in geom_feats after filtering (kept)" # 检查意义不大

        # Combine indices for efficient sorting and summing
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        # --- 检查 ranks ---
        assert torch.all(torch.isfinite(
            ranks)), "!!! NaN/Inf detected in ranks calculation !!!"
        # -----------------

        sorts = ranks.argsort()
        # 检查 sorts 的有效性 (例如，是否包含超出范围的索引) 可能比较困难
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        assert torch.all(torch.isfinite(x)), "NaN/Inf in x after sorting"
        # assert torch.all(torch.isfinite(geom_feats)), "NaN/Inf in geom_feats after sorting"
        assert torch.all(torch.isfinite(
            ranks)), "NaN/Inf in ranks after sorting"

        # Use cumsum trick for summing features in the same voxel
        # --- 检查 cumsum_trick 的输入 ---
        assert torch.all(torch.isfinite(
            x)), "!!! NaN/Inf detected in input x to cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            geom_feats)), "!!! NaN/Inf detected in input geom_feats to cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            ranks)), "!!! NaN/Inf detected in input ranks to cumsum_trick !!!"
        # -----------------------------
        x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        # --- 检查 cumsum_trick 的输出 ---
        assert torch.all(torch.isfinite(
            x)), "!!! NaN/Inf detected in output x from cumsum_trick !!!"
        assert torch.all(torch.isfinite(
            geom_feats)), "!!! NaN/Inf detected in output geom_feats from cumsum_trick !!!"
        # ----------------------------

        # --- 恢复缺失的代码块 ---
        # Create the final BEV grid (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, int(self.nx[2].item()), int(self.nx[0].item()), int(self.nx[1].item())), device=x.device)
        # Scatter summed features into the grid
        # Indices: batch, channel, z, x, y
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z by concatenation: B x (C * Z) x X x Y
        final = torch.cat(final.unbind(dim=2), 1)
        # --- 代码块恢复结束 ---

        # --- 检查最终的 final 张量 ---
        assert torch.all(torch.isfinite(
            final)), "!!! NaN/Inf detected in the final output tensor of voxel_pooling !!!"
        # --------------------------

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        """Gets camera features projected onto voxel grid and depth probabilities.
        Returns:
            cam_bev_interim: (B, C_interim, X, Y)
            depth_prob: (B, N, D, H_feat, W_feat)
        """
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # Get features and depth probability
        cam_features_per_pixel, depth_prob = self.get_cam_feats(x)
        cam_bev_interim = self.voxel_pooling(geom, cam_features_per_pixel)
        return cam_bev_interim, depth_prob  # 返回 BEV 特征和深度概率

    def forward(self, x_cam, rots, trans, intrins, post_rots, post_trans, lidar_bev):
        """Forward pass for multi-modal fusion.
        Returns:
            output: Final BEV prediction (B, outC, H_bev, W_bev)
            depth_prob: Camera depth probability (B, N, D, H_feat, W_feat)
        """
        # 1. Get Camera BEV features and depth probability
        cam_bev_interim, depth_prob = self.get_voxels(  # 接收 depth_prob
            x_cam, rots, trans, intrins, post_rots, post_trans)

        # 2. Process Camera BEV features
        cam_bev_processed = self.cam_bev_processor(cam_bev_interim)

        # 3. Process LiDAR BEV features
        lidar_bev_processed = self.lidarencode(lidar_bev)

        # 4. Fuse Features
        if cam_bev_processed.shape[2:] != lidar_bev_processed.shape[2:]:
            lidar_bev_processed = F.interpolate(lidar_bev_processed,
                                                size=cam_bev_processed.shape[2:],
                                                mode='bilinear',
                                                align_corners=False)
        fused_features = torch.cat(
            [cam_bev_processed, lidar_bev_processed], dim=1)

        # 5. Final processing
        output = self.fusion_conv(fused_features)

        return output, depth_prob  # 返回最终输出和深度概率


def compile_model_fusion(grid_conf, data_aug_conf, outC, lidar_inC=1):
    """
    创建并返回一个FusionNet实例，用于摄像头和LiDAR融合的BEV表示。

    Args:
        grid_conf: 网格配置参数
        data_aug_conf: 数据增强配置参数
        outC: 输出通道数
        lidar_inC: LiDAR输入通道数，默认为1

    Returns:
        FusionNet对象实例
    """
    return FusionNet(grid_conf, data_aug_conf, outC, lidar_inC=lidar_inC)


# 为了保持向后兼容性，保留原名称的函数接口
def compile_model(grid_conf, data_aug_conf, outC, lidar_inC=1):
    """
    创建并返回一个模型实例。根据是否提供lidar_inC参数来决定返回FusionNet还是LiftSplatShoot。

    Args:
        grid_conf: 网格配置参数
        data_aug_conf: 数据增强配置参数
        outC: 输出通道数
        lidar_inC: LiDAR输入通道数（如果不为None，则返回FusionNet）

    Returns:
        FusionNet或LiftSplatShoot模型实例
    """
    if lidar_inC is not None:
        # 返回多模态融合模型
        return FusionNet(grid_conf, data_aug_conf, outC, lidar_inC=lidar_inC)
    else:
        # 返回原始LSS模型
        return LiftSplatShoot(grid_conf, data_aug_conf, outC)


class DepthRefinementNetV2(nn.Module):
    """
    改进版深度细化网络，用于细化深度预测。
    接收初始深度logits、log方差和FPN特征作为输入。
    输出细化后的深度logits和log方差。
    """

    def __init__(self, D, fpn_channels, intermediate_channels=128):
        super().__init__()
        self.D = D
        # Input channels: D (logits) + D (log_var) + fpn_channels (p3)
        input_channels = 2 * D + fpn_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(intermediate_channels, intermediate_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU(inplace=True),
            # Output refined logits and log_variance (2 * D channels)
            nn.Conv2d(intermediate_channels, 2 * D, kernel_size=1, padding=0)
        )

    def forward(self, initial_logits, initial_log_var, fpn_feat):
        # Concatenate inputs along the channel dimension
        x = torch.cat([initial_logits, initial_log_var, fpn_feat], dim=1)
        # Pass through convolutional block
        refined_output = self.conv_block(x)
        # Split into refined logits and log variance
        refined_logits = refined_output[:, :self.D]
        refined_log_var = refined_output[:, self.D:]
        return refined_logits, refined_log_var
