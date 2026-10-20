"""SRM detector (Luo et al. 2021): dual-stream Xception (RGB + noise) with
cross-modal attention.

Ported from DeepfakeBench ``training/detectors/srm_detector.py``.  The RGB and
SRM-filtered (noise) streams are fused by two dual cross-modal attention blocks
and a channel-attention fusion module.  The checkpoint (``srm_best.pth``) is
the full detector state dict.

Note: the released checkpoint fixes the spatial size used by the
``DualCrossModalAttention`` linear layers and the CMA input channels, so
``build_model`` derives both from the checkpoint's own tensor shapes rather
than assuming the source defaults.
"""
from __future__ import annotations

import math
import numbers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


def _srm_filters():
    filter1 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter2 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    return (np.asarray(filter1, dtype=float) / 4.,
            np.asarray(filter2, dtype=float) / 12.,
            np.asarray(filter3, dtype=float) / 2.)


class SRMConv2d_simple(nn.Module):
    def __init__(self, inc=3):
        super().__init__()
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2)
        return self.truc(out)

    def _build_kernel(self, inc):
        f1, f2, f3 = _srm_filters()
        filters = np.array([[f1], [f2], [f3]])
        filters = np.repeat(filters, inc, axis=1)
        return torch.FloatTensor(filters)


class SRMConv2d_Separate(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.inc = inc
        self.truc = nn.Hardtanh(-3, 3)
        kernel = self._build_kernel(inc)
        self.kernel = nn.Parameter(data=kernel, requires_grad=False)
        self.out_conv = nn.Sequential(
            nn.Conv2d(3 * inc, outc, 1, 1, 0, 1, 1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True),
        )
        for ly in self.out_conv.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)

    def forward(self, x):
        out = F.conv2d(x, self.kernel, stride=1, padding=2, groups=self.inc)
        out = self.truc(out)
        return self.out_conv(out)

    def _build_kernel(self, inc):
        f1, f2, f3 = _srm_filters()
        filters = np.array([[f1], [f2], [f3]])
        filters = np.repeat(filters, inc, axis=0)
        return torch.FloatTensor(filters)


class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma=0.1, dim=2):
        super().__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        self.groups = channels
        self.conv = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[dim]

    def forward(self, x):
        if self.training:
            return self.conv(x, weight=self.weight, groups=self.groups,
                             padding=self.kernel_size // 2)
        return x


class GaussianNoise(nn.Module):
    def __init__(self, mean=0, std=0.1, clip=1):
        super().__init__()
        self.mean = mean
        self.std = std
        self.clip = clip

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            return torch.clamp(x + noise, -self.clip, self.clip)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        return self.sigmoid(self.conv(x))


class SRMPixelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.srm = SRMConv2d_simple()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pa = SpatialAttention()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_srm = self.srm(x)
        fea = self.conv(x_srm)
        return self.pa(fea)


class DualCrossModalAttention(nn.Module):
    def __init__(self, in_dim, size=16, ratio=8):
        super().__init__()
        self.chanel_in = in_dim
        self.key_conv1 = nn.Conv2d(in_dim, in_dim // ratio, 1)
        self.key_conv2 = nn.Conv2d(in_dim, in_dim // ratio, 1)
        self.key_conv_share = nn.Conv2d(in_dim // ratio, in_dim // ratio, 1)
        self.linear1 = nn.Linear(size * size, size * size)
        self.linear2 = nn.Linear(size * size, size * size)
        self.value_conv1 = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.value_conv2 = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        B, C, H, W = x.size()
        n = H * W

        def _get_att(a, b):
            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, n).permute(0, 2, 1)
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(B, -1, n)
            energy = torch.bmm(proj_key1, proj_key2)
            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(energy.permute(0, 2, 1)))
            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        proj_value_y_on_x = self.value_conv2(y).view(B, -1, n)
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1 * out_y_on_x + x
        proj_value_x_on_y = self.value_conv1(x).view(B, -1, n)
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        out_y = self.gamma2 * out_x_on_y + y
        return out_x, out_y


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048 * 2, out_chan=2048):
        super().__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        return fuse_fea * self.ca(fuse_fea)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class SRMDetector(nn.Module):
    def __init__(self, num_classes=2, mode="original", inc=3, dropout=False,
                 cma_dim=728, cma_size=16):
        super().__init__()
        self.backbone_rgb = Xception(num_classes=num_classes, mode=mode, inc=inc,
                                     dropout=dropout)
        self.backbone_srm = Xception(num_classes=num_classes, mode=mode, inc=inc,
                                     dropout=dropout)
        self.noise = GaussianNoise(clip=1)
        self.blur = GaussianSmoothing(channels=3, kernel_size=7, sigma=0.8)
        self.srm_conv0 = SRMConv2d_simple(inc=3)
        self.srm_conv1 = SRMConv2d_Separate(32, 32)
        self.srm_conv2 = SRMConv2d_Separate(64, 64)
        self.relu = nn.ReLU(inplace=True)
        self.att_map = None
        self.srm_sa = SRMPixelAttention(3)
        self.srm_sa_post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.dual_cma0 = DualCrossModalAttention(in_dim=cma_dim, size=cma_size)
        self.dual_cma1 = DualCrossModalAttention(in_dim=cma_dim, size=cma_size)
        self.fusion = FeatureFusionModule()

    def features(self, x):
        srm = self.srm_conv0(x)
        x = self.backbone_rgb.fea_part1_0(x)
        y = self.backbone_srm.fea_part1_0(srm) + self.srm_conv1(x)
        y = self.relu(y)
        x = self.backbone_rgb.fea_part1_1(x)
        y = self.backbone_srm.fea_part1_1(y) + self.srm_conv2(x)
        y = self.relu(y)
        self.att_map = self.srm_sa(srm)
        x = x * self.att_map + x
        x = self.srm_sa_post(x)
        x = self.backbone_rgb.fea_part2(x)
        y = self.backbone_srm.fea_part2(y)
        x, y = self.dual_cma0(x, y)
        x = self.backbone_rgb.fea_part3(x)
        y = self.backbone_srm.fea_part3(y)
        x, y = self.dual_cma1(x, y)
        x = self.backbone_rgb.fea_part4(x)
        y = self.backbone_srm.fea_part4(y)
        x = self.backbone_rgb.fea_part5(x)
        y = self.backbone_srm.fea_part5(y)
        return self.fusion(x, y)

    def classifier(self, feats):
        return self.backbone_rgb.classifier(feats)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": feats,
            "attention_map": self.att_map,
        }


def _strip_module(state):
    if all(k.startswith("module.") for k in state.keys()):
        return {k[7:]: v for k, v in state.items()}
    return state


def build_model(weights_path, device):
    state = torch.load(weights_path, map_location="cpu")
    state = _strip_module(state)

    cma_dim = state.get("dual_cma0.key_conv1.weight")
    cma_dim = cma_dim.shape[1] if cma_dim is not None else 728
    linear = state.get("dual_cma0.linear1.weight")
    cma_size = int(round(math.sqrt(linear.shape[1]))) if linear is not None else 16

    model = SRMDetector(num_classes=2, mode="original", inc=3, dropout=False,
                        cma_dim=cma_dim, cma_size=cma_size)

    fusion_conv = state.get("fusion.convblk.0.weight")
    if fusion_conv is not None and fusion_conv.shape[1] != 4096:
        model.fusion = FeatureFusionModule(in_chan=fusion_conv.shape[1],
                                           out_chan=fusion_conv.shape[0])
    if "backbone_rgb.adjust_channel.conv.weight" in state:
        for name in ("backbone_rgb", "backbone_srm"):
            backbone = getattr(model, name)
            backbone.mode = "adjust_channel"
            backbone.adjust_channel = nn.Sequential(
                nn.Conv2d(2048, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=False),
            )

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
