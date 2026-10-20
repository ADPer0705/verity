"""SAFE detector (Li et al., KDD 2025, arXiv 2408.06741): "Improving Synthetic
Image Detection Towards Generalization: An Image Transformation Perspective".

Ported from ``Ouxiang-Li/SAFE`` (Apache-2.0), ``models/resnet.py`` +
``main_finetune.py`` (``model = resnet50(num_classes=2)``).

Architecture (``models/resnet.py::ResNet``): the input is first transformed
inside ``forward`` by ``_preprocess_dwt``: a one-level 2-D discrete wavelet
transform with ``pytorch_wavelets.DWTForward(J=1, mode="symmetric",
wave="bior1.3")``; only the **HH** (diagonal detail) sub-band ``Yh[0][:, :, 2]``
is kept and bilinearly resized back to the input size
(``transforms.Resize([H, W])`` on the tensor == ``F.interpolate(bilinear,
antialias=True)``; antialiasing is a no-op when upsampling).  The 3-channel
HH map then goes through a truncated ResNet-50 (3x3 stem conv stride 2,
``layer1`` 3 bottlenecks, ``layer2`` 4 bottlenecks, no layer3/4), global
average pool and ``fc1 = Linear(512, 2)``: 1,448,410 parameters.  The head is
2-class CrossEntropy with label 1 = fake (``1_fake`` folders in
``data/datasets.py``), so ``softmax(logits)[:, 1]`` is directly P(fake) and no
logit stacking is needed.

Preprocessing (``data/datasets.py::Get_Transforms`` with the official
``scripts/eval.sh`` arguments ``--input_size 256 --transform_mode crop``):
eval transform = ``transforms.CenterCrop([256, 256])`` -> ``ToTensor()``.  No
resize and **no mean/std normalisation**: the network sees raw [0, 1] pixels at
native resolution (SAFE's point is that resizing destroys the high-frequency
evidence).  ``torchvision.CenterCrop`` zero-pads images smaller than 256; the
same is done here.  ``preprocess`` implements this and the model's
``forward_detailed`` applies the wavelet transform exactly as upstream.

Checkpoint: ``checkpoint/checkpoint-best.pth`` committed in the official repo,
a dict ``{"model": state_dict}`` (146 keys: ``conv1.*``, ``bn1.*``,
``layer1.*``, ``layer2.*``, ``fc1.*``), loaded with ``strict=True``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CROP = 256
WAVELET = "bior1.3"
DWT_MODE = "symmetric"


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SAFEDetector(nn.Module):
    """``models/resnet.py::ResNet(Bottleneck, [3, 4, 6, 3], num_classes=2)``
    (only ``layers[0]``/``layers[1]`` are instantiated upstream)."""

    def __init__(self, num_classes=2):
        super().__init__()
        from pytorch_wavelets import DWTForward

        self.dwt = DWTForward(J=1, mode=DWT_MODE, wave=WAVELET)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # -- preprocessing (CenterCrop([256, 256]) + ToTensor, no normalisation) --
    @staticmethod
    def preprocess(pil: Image.Image) -> torch.Tensor:
        pil = pil.convert("RGB")
        arr = np.asarray(pil, dtype="float32") / 255.0  # (H, W, 3) in [0, 1]
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        _, _, h, w = x.shape
        # torchvision CenterCrop semantics: zero-pad when smaller, then crop.
        pad_h, pad_w = max(0, CROP - h), max(0, CROP - w)
        if pad_h or pad_w:
            left, top = pad_w // 2, pad_h // 2
            x = F.pad(x, (left, pad_w - left, top, pad_h - top), value=0.0)
            _, _, h, w = x.shape
        top, left = int(round((h - CROP) / 2.0)), int(round((w - CROP) / 2.0))
        return x[:, :, top:top + CROP, left:left + CROP]

    # -- forward (== upstream ResNet.forward) --------------------------------
    def _preprocess_dwt(self, x):
        _, yh = self.dwt(x)
        hh = yh[0][:, :, 2, :, :]
        # transforms.Resize([H, W]) on a tensor: bilinear, antialias (no-op for upsampling)
        return F.interpolate(hh, size=x.shape[-2:], mode="bilinear",
                             align_corners=False, antialias=True)

    def forward_detailed(self, x):
        h = self._preprocess_dwt(x)
        h = self.maxpool(self.relu(self.bn1(self.conv1(h))))
        h = self.layer1(h)
        feat = self.layer2(h)
        logits = self.fc1(self.avgpool(feat).flatten(1))
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feat}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = SAFEDetector(num_classes=2)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        # DWTForward holds its filters as (non-persistent-equivalent) buffers
        # that are not in the upstream checkpoint; load only the network.
        own = model.state_dict()
        unexpected = [k for k in state if k not in own]
        missing = [k for k in own if k not in state and not k.startswith("dwt.")]
        if unexpected or missing:
            raise RuntimeError(f"safe checkpoint mismatch: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]}")
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
