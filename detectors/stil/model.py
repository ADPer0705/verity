"""STIL detector (Gu et al. 2021): SCNet50-v1d spatiotemporal backbone with
TIM/ISM inconsistency modules for deepfake video detection.

Ported from DeepfakeBench ``training/detectors/stil_detector.py``.  Input is a
channel-first clip ``(B, 3, T, H, W)`` (clip_size 8); it is rearranged to
frame-major ``(B, T*3, H, W)``, the SCNet scores every frame independently and
the per-frame logits are averaged over the clip (the SCNet features of the
last stage are also returned as a ``(B, T, 2048, 7, 7)`` tensor for CAM).  The
ImageNet SCNet weights the source fetches at runtime are not downloaded; the
model is randomly initialized unless ``weights_path`` is given (checkpoint
keys are remapped ``base_``/``model.``-strip + ``base_model.``-prefix exactly
as in the source).  Training-only code (losses, metrics, ``set_segment``) is
dropped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISM_Module(nn.Module):
    """Information Supplement Module: channel attention on the s2t features."""

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TIM_Module(nn.Module):
    """Temporal Inconsistency Module: H-T and T-W inconsistency attention."""

    def __init__(self, in_channels, reduction=16, n_segment=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.n_segment = n_segment
        self.reduced_channels = self.in_channels // self.reduction

        self.conv1 = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.reduced_channels)

        self.conv_ht = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(3, 1), padding=(1, 0), groups=self.reduced_channels, bias=False)
        self.conv_tw = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(1, 3), padding=(0, 1), groups=self.reduced_channels, bias=False)

        self.avg_pool_ht = nn.AvgPool2d((2, 1), (2, 1))
        self.avg_pool_tw = nn.AvgPool2d((1, 2), (1, 2))

        self.htie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.htie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.vtie_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
        )
        self.tw_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def feat_ht(self, feat):
        n, t, c, h, w = feat.size()
        feat_h = feat.permute(0, 4, 2, 3, 1).contiguous().view(-1, c, h, t)

        feat_h_fwd, _ = feat_h.split([self.n_segment - 1, 1], dim=3)
        feat_h_conv = self.conv_ht(feat_h)
        _, feat_h_conv_fwd = feat_h_conv.split([1, self.n_segment - 1], dim=3)

        diff_feat_fwd = feat_h_conv_fwd - feat_h_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 1], value=0)

        diff_feat_fwd1 = self.avg_pool_ht(diff_feat_fwd)
        diff_feat_fwd1 = self.htie_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])
        diff_feat_fwd2 = self.htie_conv2(diff_feat_fwd)

        feat_ht_out = self.ht_up_conv(1 / 3. * diff_feat_fwd + 1 / 3. * diff_feat_fwd1 + 1 / 3. * diff_feat_fwd2)
        feat_ht_out = self.sigmoid(feat_ht_out) - 0.5
        feat_ht_out = feat_ht_out.view(n, w, self.in_channels, h, t).permute(0, 4, 2, 3, 1).contiguous()
        feat_ht_out = feat_ht_out.view(-1, self.in_channels, h, w)
        return feat_ht_out

    def feat_tw(self, feat):
        n, t, c, h, w = feat.size()
        feat_w = feat.permute(0, 3, 2, 1, 4).contiguous().view(-1, c, t, w)

        feat_w_fwd, _ = feat_w.split([self.n_segment - 1, 1], dim=2)
        feat_w_conv = self.conv_tw(feat_w)
        _, feat_w_conv_fwd = feat_w_conv.split([1, self.n_segment - 1], dim=2)

        diff_feat_fwd = feat_w_conv_fwd - feat_w_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 0, 0, 1], value=0)

        diff_feat_fwd1 = self.avg_pool_tw(diff_feat_fwd)
        diff_feat_fwd1 = self.vtie_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])
        diff_feat_fwd2 = self.vtie_conv2(diff_feat_fwd)

        feat_tw_out = self.tw_up_conv(1 / 3. * diff_feat_fwd + 1 / 3. * diff_feat_fwd1 + 1 / 3. * diff_feat_fwd2)
        feat_tw_out = self.sigmoid(feat_tw_out) - 0.5
        feat_tw_out = feat_tw_out.view(n, h, self.in_channels, t, w).permute(0, 3, 2, 1, 4).contiguous()
        feat_tw_out = feat_tw_out.view(-1, self.in_channels, h, w)
        return feat_tw_out

    def forward(self, x):
        bottleneck = self.conv1(x)
        bottleneck = self.bn1(bottleneck)
        bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])

        F_h = self.feat_ht(bottleneck)
        F_w = self.feat_tw(bottleneck)

        att = 0.5 * (F_h + F_w)
        return x + x * att


class ShiftModule(nn.Module):
    """Depth-wise conv on the segment level (channel shifting across frames)."""

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super().__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div
        self.conv = nn.Conv1d(self.fold_div * self.fold, self.fold_div * self.fold,
                              kernel_size=3, padding=1, groups=self.fold_div * self.fold, bias=False)
        if mode == 'shift':
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1
        elif mode == 'fixed':
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)
        x = x.contiguous().view(nt, c, h, w)
        return x


class SCConv(nn.Module):
    """Spatial conv used in SCBottleneck: s2t (space-to-time) information."""

    def __init__(self, inplanes, planes, stride, padding, dilation, groups, pooling_r, norm_layer):
        super().__init__()
        self.f_w = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(inplanes, planes, kernel_size=(1, 3), stride=1,
                      padding=(0, padding), dilation=(1, dilation), groups=groups, bias=False),
            norm_layer(planes), nn.ReLU(inplace=True))
        self.f_h = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=(3, 1), stride=1,
                      padding=(padding, 0), dilation=(dilation, 1), groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(planes),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(planes),
        )

    def forward(self, x):
        identity = x
        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.f_h(self.f_w(x)), identity.size()[2:]))
        )
        out = torch.mul(self.k3(x), out)
        s2t_info = out
        out = self.k4(out)
        return out, s2t_info


class SCBottleneck(nn.Module):
    """SCNet bottleneck: trace (temporal) branch a and spatial branch b."""

    expansion = 4
    pooling_r = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None,
                 cardinality=1, bottleneck_width=32, avd=False, dilation=1,
                 is_first=False, norm_layer=None):
        super().__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1_a = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_a = norm_layer(group_width)
        self.conv1_b = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1_b = norm_layer(group_width)
        self.avd = avd and (stride > 1 or is_first)
        self.tim = TIM_Module(group_width, n_segment=num_segments)
        self.ism = ISM_Module()
        self.shift = ShiftModule(group_width, n_segment=num_segments, n_div=8, mode='shift')

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.k1 = nn.Sequential(
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride,
                      padding=dilation, dilation=dilation, groups=cardinality, bias=False),
            norm_layer(group_width),
        )

        self.scconv = SCConv(
            group_width, group_width, stride=stride,
            padding=dilation, dilation=dilation, groups=cardinality,
            pooling_r=self.pooling_r, norm_layer=norm_layer)

        self.conv3 = nn.Conv2d(group_width * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out_a = self.relu(self.bn1_a(self.conv1_a(x)))
        out_b = self.relu(self.bn1_b(self.conv1_b(x)))

        out_b, s2t_info = self.scconv(out_b)
        out_b = self.relu(out_b)

        out_a = self.tim(out_a)
        out_a = self.shift(out_a + self.ism(s2t_info))
        out_a = self.relu(self.k1(out_a))

        if self.avd:
            out_a = self.avd_layer(out_a)
            out_b = self.avd_layer(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SCNet(nn.Module):
    """ResNet-50 variant (deep stem, avg_down, avd) with SCBottleneck blocks."""

    def __init__(self, num_segments, block, layers, groups=1, bottleneck_width=32,
                 num_classes=1000, dilated=False, dilation=1,
                 deep_stem=False, stem_width=64, avg_down=False,
                 avd=False, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.avd = avd
        self.num_segments = num_segments

        conv_layer = nn.Conv2d
        if deep_stem:
            self.conv1 = nn.Sequential(
                conv_layer(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                conv_layer(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    norm_layer=None, is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.num_segments, self.inplanes, planes, stride,
                                downsample=downsample, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd,
                                dilation=1, is_first=is_first, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.num_segments, self.inplanes, planes, stride,
                                downsample=downsample, cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width, avd=self.avd,
                                dilation=2, is_first=is_first, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.num_segments, self.inplanes, planes,
                                cardinality=self.cardinality,
                                bottleneck_width=self.bottleneck_width,
                                avd=self.avd, dilation=dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def scnet50_v1d(num_segments, pretrained=False, **kwargs):
    model = SCNet(num_segments, SCBottleneck, [3, 4, 6, 3],
                  deep_stem=True, stem_width=32, avg_down=True, avd=True, **kwargs)
    return model


class STIL_Model(nn.Module):
    def __init__(self, num_class=2, num_segment=8, add_softmax=False):
        super().__init__()
        self.num_class = num_class
        self.num_segment = num_segment
        self.add_softmax = add_softmax
        self.base_model = scnet50_v1d(num_segment, pretrained=False)
        fc_feature_dim = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(fc_feature_dim, num_class)
        if add_softmax:
            self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.base_model(x.view((-1, 3) + x.size()[2:]))
        out = out.view(-1, self.num_segment, self.num_class)
        out = out.mean(1, keepdim=False)
        if self.add_softmax:
            out = self.softmax_layer(out)
        return out


class STILDetector(nn.Module):
    def __init__(self, num_class=2, num_segment=8):
        super().__init__()
        self.model = STIL_Model(num_class=num_class, num_segment=num_segment, add_softmax=False)

    def features(self, x):
        # (B, 3, T, H, W) -> (B, T*3, H, W) frame-major, as in the source
        bs, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(bs, t * c, h, w)

    def forward_detailed(self, x):
        x = self.features(x)
        frames = x.view((-1, 3) + x.size()[2:])
        feats = self.model.base_model.features(frames)
        logits = self.model.base_model.logits(feats)
        logits = logits.view(-1, self.model.num_segment, self.model.num_class).mean(1, keepdim=False)
        feats = feats.view(-1, self.model.num_segment, feats.size(1), feats.size(2), feats.size(3))
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = STILDetector(num_class=2, num_segment=8)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        state = {k.replace("base_", "").replace("model.", ""): v for k, v in state.items()}
        state = {"base_model." + k: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
