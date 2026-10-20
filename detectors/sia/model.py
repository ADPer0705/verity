"""SIA detector (Sun et al. 2022): EfficientNet-B4 + SAIA attention head.

Ported from DeepfakeBench ``training/detectors/sia_detector.py``.  The SAIA
attention convs drive the inference path inside ``extract_features``; the
source's ``att0conv``/``att3conv``/``att5conv``/``att6conv`` heads and the dead
``linear`` module only served training-time attention losses, so they are
dropped (absent keys are ignored via ``strict=False``).  The checkpoint is the
full detector state dict (keys ``backbone.efficientnet.*``,
``backbone.last_layer.*``, ``att1conv.*``, ``att2conv.*``, ``att4conv.*``,
``conv1.*``, ``conv2.*``, ``conv3.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super().__init__()
        self.dropout = dropout
        self.mode = mode
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.last_layer = nn.Linear(1792, num_classes)

    def classifier(self, feats):
        x = F.adaptive_avg_pool2d(feats, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        self.last_emb = x
        return self.last_layer(x)


class SAIAConv(nn.Module):
    def __init__(self, outdim, kernel_size=3, padding=1, isspace=True, ischannel=True):
        super().__init__()
        self.band_width = 1.0
        self.isspace = isspace
        self.ischannel = ischannel
        self.outdim = outdim
        kernel = torch.ones((outdim, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        kernel2 = torch.ones((outdim, 1, 1, 1)) * (kernel_size * kernel_size)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)
        self.pad = padding
        self.channel_range = 5

    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            num_channel = x.shape[1]
            x1 = F.conv2d(x, self.weight, padding=self.pad, groups=self.outdim)
            x2 = F.conv2d(x, self.weight2, padding=0, groups=self.outdim)
            intra_distance = torch.abs(x2 - x1)

            pad_x = torch.cat([x, x[:, :self.channel_range + 1, :, :]], dim=1)
            distances = []
            for i in range(1, self.channel_range + 1):
                distances.append(x - pad_x[:, i:num_channel + i, :, :])
            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = distance.view(batch_size, -1, self.channel_range, h_dis, w_dis).sum(dim=2)
            inter_distance = torch.abs(distance.view(batch_size, -1, h_dis, w_dis))
            att = intra_distance + 0.5 * inter_distance

        if self.ischannel:
            distance_channel = att
            distance_channel = torch.exp(
                -distance_channel / distance_channel.mean() / 2 / self.band_width ** 2)
            distance_channel = -torch.log(distance_channel + 0.1)
            channel_attention = torch.mean(distance_channel.view(batch_size, self.outdim, -1), dim=2)
            channel_attention = channel_attention.view(batch_size, -1, 1, 1) + 1

        if self.isspace:
            distance_space = att
            distance_space = distance_space / distance_space.mean() / 2 / self.band_width ** 2
            space_attention = distance_space
            attention_image = (torch.sigmoid(space_attention) + 1) * x

        if self.isspace and self.ischannel:
            return attention_image * (channel_attention.expand_as(x)), space_attention
        elif self.isspace:
            return attention_image, x
        else:
            return x * (channel_attention.expand_as(x)), x


class SIADetector(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super().__init__()
        self.backbone = EfficientNetB4(num_classes=num_classes, inc=inc, dropout=dropout, mode=mode)

        self.att1conv = SAIAConv(32, kernel_size=3, isspace=True, ischannel=True)
        self.att2conv = SAIAConv(56, kernel_size=3, isspace=True, ischannel=True)
        self.att4conv = SAIAConv(160, kernel_size=3, isspace=True, ischannel=True)

        self.avgpool1 = nn.AdaptiveMaxPool2d((32, 32))
        self.avgpool2 = nn.AdaptiveMaxPool2d((16, 16))

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 56, 1, 1, 0),
            nn.BatchNorm2d(56),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(56, 160, 1, 1, 0),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

    def extract_features(self, inputs):
        x = self.backbone.efficientnet._conv_stem(inputs)
        x = self.backbone.efficientnet._bn0(x)
        x = self.backbone.efficientnet._swish(x)

        for block in self.backbone.efficientnet._blocks[:2]:
            x = block(x)
        for block in self.backbone.efficientnet._blocks[2:6]:
            x = block(x)

        x, att1 = self.att1conv(x)
        res12 = self.avgpool1(self.conv1(att1))
        res14 = self.avgpool2(self.conv2(att1))

        for block in self.backbone.efficientnet._blocks[6:10]:
            x = block(x)

        x, att2 = self.att2conv(x + res12)
        res24 = self.avgpool2(self.conv3(att2))

        for block in self.backbone.efficientnet._blocks[10:22]:
            x = block(x)

        x, att4 = self.att4conv(x + res24 + res14)

        for block in self.backbone.efficientnet._blocks[22:]:
            x = block(x)

        x = self.backbone.efficientnet._swish(
            self.backbone.efficientnet._bn1(self.backbone.efficientnet._conv_head(x)))
        return x

    def features(self, x):
        return self.extract_features(x)

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = SIADetector(num_classes=2, inc=3, dropout=False, mode="Original")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
