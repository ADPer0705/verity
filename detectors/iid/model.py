"""IID detector (Huang et al. 2023): Xception classifier fed the difference
between the image embedding and an explicit identity embedding from a frozen
iresnet50 face-recognition extractor (implicit identity detection).

Ported from DeepfakeBench ``training/detectors/iid_detector.py``.  Input is a
``(B, 3, 256, 256)`` image.  The image is also resized to 112x112 and embedded
by ``explicit_extractor`` (iresnet50, 512-d; the final BN is skipped for
batch size 1 exactly as in the source).  The Xception backbone runs in
``adjust_channel_iid`` mode (1x1 conv 2048->512) and predicts
``last_linear(x_emb - id_feat)``.  ``weights_path`` is the iresnet50
face-recognition backbone (``backbone.pth``); the Xception backbone has no
released weights and stays randomly initialized.  Identity-indexed losses
(BCE/IIE/EIC), DDP helpers and the dataset code are dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None):
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.features(x)
        return x


def iresnet50(pretrained=False, progress=True, **kwargs):
    model = IResNet(IBasicBlock, [3, 4, 14, 3], **kwargs)
    if pretrained:
        raise ValueError()
    return model


class IIDDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes, mode="adjust_channel_iid",
                                 inc=3, dropout=False)
        self.explicit_extractor = iresnet50(pretrained=False)

    def features(self, x):
        return self.backbone.features(x)

    def classifier(self, features, id_feat=None):
        return self.backbone.classifier(features, id_feat)

    def forward_detailed(self, x):
        resized_images = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)
        id_feat = self.explicit_extractor(resized_images)
        features = self.features(x)
        logits = self.classifier(features, id_feat)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "feat": features, "id_feat": id_feat, "embed": self.backbone.last_emb}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = IIDDetector(num_classes=2)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        # DeepfakeBench's IID port expands 2D pointwise conv weights to
        # (C_out, C_in, 1, 1) and drops the fc head before loading.
        for name, weights in state.items():
            if "pointwise" in name:
                state[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state = {k: v for k, v in state.items() if "fc" not in k}
        # The configured checkpoint is the ImageNet-pretrained Xception
        # backbone (DeepfakeBench iid.yaml `pretrained:`). The iresnet50
        # identity extractor additionally uses backbone.pth when present;
        # without it the identity branch stays randomly initialized.
        model.backbone.load_state_dict(state, strict=False)
        identity_weights = Path(__file__).resolve().parent / "weights" / "backbone.pth"
        if identity_weights.exists():
            identity_state = torch.load(identity_weights, map_location="cpu")
            if all(k.startswith("module.") for k in identity_state.keys()):
                identity_state = {k[7:]: v for k, v in identity_state.items()}
            model.explicit_extractor.load_state_dict(identity_state, strict=False)
    model.to(device)
    model.eval()
    return model
