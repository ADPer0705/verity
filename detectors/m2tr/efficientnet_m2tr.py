"""EfficientNet backbone exactly as vendored by M2TR (``M2TR/models/efficientnet.py``,
MIT): the lukemelas EfficientNet-PyTorch architecture with *static* TF-SAME
padding computed from the nominal model resolution (380 for B4) in **every**
convolution, and the ``extract_textures`` / ``extract_features`` split used
by M2TR (stem + first 6 blocks = texture stage ``b2``; the remaining blocks +
head = ``final``).  Kept separate from ``efficientnet_pytorch`` because that
package derives per-block padding from the running feature-map size, which
differs by one pixel at some layers for the 320 px M2TR input.
"""
from __future__ import annotations

import collections
import math
import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

GlobalParams = collections.namedtuple("GlobalParams", [
    "batch_norm_momentum", "batch_norm_epsilon", "dropout_rate", "num_classes", "width_coefficient",
    "depth_coefficient", "depth_divisor", "min_depth", "drop_connect_rate", "image_size"])
BlockArgs = collections.namedtuple("BlockArgs", [
    "kernel_size", "num_repeat", "input_filters", "output_filters", "expand_ratio", "id_skip", "stride", "se_ratio"])

EFFICIENTNET_PARAMS = {
    "efficientnet-b0": (1.0, 1.0, 224, 0.2), "efficientnet-b1": (1.0, 1.1, 240, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3), "efficientnet-b3": (1.2, 1.4, 300, 0.3),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4), "efficientnet-b5": (1.6, 2.2, 456, 0.4),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5), "efficientnet-b7": (2.0, 3.1, 600, 0.5),
}
BLOCKS_ARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25", "r2_k3_s22_e6_i16_o24_se0.25", "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25", "r3_k5_s11_e6_i80_o112_se0.25", "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


def _decode(block_string):
    options = {}
    for op in block_string.split("_"):
        splits = re.split(r"(\d.*)", op)
        if len(splits) >= 2:
            options[splits[0]] = splits[1]
    return BlockArgs(kernel_size=int(options["k"]), num_repeat=int(options["r"]),
                     input_filters=int(options["i"]), output_filters=int(options["o"]),
                     expand_ratio=int(options["e"]), id_skip=("noskip" not in block_string),
                     se_ratio=float(options["se"]) if "se" in options else None, stride=[int(options["s"][0])])


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, gp):
    if not gp.width_coefficient:
        return filters
    divisor = gp.depth_divisor
    filters *= gp.width_coefficient
    new_filters = max(gp.min_depth or divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, gp):
    return int(math.ceil(gp.depth_coefficient * repeats)) if gp.depth_coefficient else repeats


class Conv2dStaticSamePadding(nn.Conv2d):
    """TF-SAME padding for a fixed (nominal) image size."""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        ih, iw = image_size if isinstance(image_size, list) else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class MBConvBlock(nn.Module):
    def __init__(self, block_args, gp):
        super().__init__()
        self._block_args = block_args
        bn_mom, bn_eps = 1 - gp.batch_norm_momentum, gp.batch_norm_epsilon
        self.has_se = block_args.se_ratio is not None and 0 < block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        Conv2d = partial(Conv2dStaticSamePadding, image_size=gp.image_size)
        inp, oup = block_args.input_filters, block_args.input_filters * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)
        self._depthwise_conv = Conv2d(oup, oup, groups=oup, kernel_size=block_args.kernel_size,
                                      stride=block_args.stride, bias=False)
        self._bn1 = nn.BatchNorm2d(oup, momentum=bn_mom, eps=bn_eps)
        if self.has_se:
            n_sq = max(1, int(block_args.input_filters * block_args.se_ratio))
            self._se_reduce = Conv2d(oup, n_sq, kernel_size=1)
            self._se_expand = Conv2d(n_sq, oup, kernel_size=1)
        self._project_conv = Conv2d(oup, block_args.output_filters, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(block_args.output_filters, momentum=bn_mom, eps=bn_eps)
        self._swish = Swish()

    def forward(self, inputs):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            s = F.adaptive_avg_pool2d(x, 1)
            s = self._se_expand(self._swish(self._se_reduce(s)))
            x = torch.sigmoid(s) * x
        x = self._bn2(self._project_conv(x))
        if self.id_skip and self._block_args.stride == 1 and \
                self._block_args.input_filters == self._block_args.output_filters:
            x = x + inputs
        return x


class EfficientNetM2TR(nn.Module):
    def __init__(self, model_name="efficientnet-b4", num_classes=2):
        super().__init__()
        w, d, s, p = EFFICIENTNET_PARAMS[model_name]
        gp = GlobalParams(batch_norm_momentum=0.99, batch_norm_epsilon=1e-3, dropout_rate=p, drop_connect_rate=0.2,
                          num_classes=num_classes, width_coefficient=w, depth_coefficient=d, depth_divisor=8,
                          min_depth=None, image_size=s)
        self._global_params = gp
        Conv2d = partial(Conv2dStaticSamePadding, image_size=gp.image_size)
        bn_mom, bn_eps = 1 - gp.batch_norm_momentum, gp.batch_norm_epsilon
        out_channels = round_filters(32, gp)
        self._conv_stem = Conv2d(3, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)
        self._blocks = nn.ModuleList([])
        self.stage_map = []
        for stage_count, block_string in enumerate(BLOCKS_ARGS, start=1):
            ba = _decode(block_string)
            ba = ba._replace(input_filters=round_filters(ba.input_filters, gp),
                             output_filters=round_filters(ba.output_filters, gp),
                             num_repeat=round_repeats(ba.num_repeat, gp))
            self.stage_map += [""] * (ba.num_repeat - 1)
            self.stage_map.append(f"b{stage_count}")
            self._blocks.append(MBConvBlock(ba, gp))
            if ba.num_repeat > 1:
                ba = ba._replace(input_filters=ba.output_filters, stride=1)
            for _ in range(ba.num_repeat - 1):
                self._blocks.append(MBConvBlock(ba, gp))
        in_channels = ba.output_filters
        out_channels = round_filters(1280, gp)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_channels, momentum=bn_mom, eps=bn_eps)
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(gp.dropout_rate)
        self._fc = nn.Linear(out_channels, gp.num_classes)   # unused by M2TR, present in the checkpoint
        self._swish = Swish()

    def extract_textures(self, inputs, layers):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        layers["b0"] = x
        for idx, block in enumerate(self._blocks[:6]):
            x = block(x)
            if self.stage_map[idx]:
                layers[self.stage_map[idx]] = x
        return x

    def extract_features(self, x, layers):
        for idx, block in enumerate(self._blocks[6:]):
            idx += 6
            x = block(x)
            if self.stage_map[idx]:
                layers[self.stage_map[idx]] = x
        return self._swish(self._bn1(self._conv_head(x)))
