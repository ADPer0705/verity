"""NPR detector (Tan et al., CVPR 2024).

Ported from ``chuangchuangtan/NPR-DeepfakeDetection`` (``networks/resnet.py``,
``test.py``).  The backbone is a truncated ResNet-50: only ``conv1`` +
``layer1`` (3 blocks) + ``layer2`` (4 blocks) exist in the official model and
checkpoint, followed by ``avgpool`` and ``fc1`` (Linear(512, 1)); there is no
``layer3``/``layer4``.  The official ``forward`` feeds the no-reference
pixel-level representation ``NPR = x - upsample2x(downsample2x(x))`` (nearest)
scaled by 2/3 into ``conv1``.

Class mapping: the head is a single BCE logit (trained with
``BCEWithLogitsLoss``, label 1 = fake per the ``0_real``/``1_fake`` dataset
folders and ``validate.py``); ``sigmoid(logit)`` is the official fake
probability.  ``forward_detailed`` emits 2-class logits ``[0, logit]`` whose
softmax index 1 equals ``sigmoid(logit)``, satisfying the engine contract
(index 1 = fake).

Preprocessing (``input_mode: full_frame``): the official ``test.py`` sets
``no_crop=True`` for every benchmark and ``no_resize=True`` for
GANGen-Detection (whole image at native resolution); the other sets get
``transforms.Resize((256, 256))`` only "due to the different shapes of images
in the dataset" (batching), and those images are ~256 px anyway, so the
checkpoint effectively always saw native-resolution pixels.  NPR's signal is
the pixel-grid up-sampling residual, which resampling destroys, so
``preprocess`` mirrors the ``no_resize + no_crop`` path: the whole frame at
native resolution, ToTensor -> Normalize(ImageNet mean/std), with the sides
floored to even numbers (the 2x down/up residual needs even dims) and a
centre crop to at most ``MAX_SIDE`` (1024) px per side as a memory guard for
very large inputs.  The yaml mean/std match but are applied here, not by the
engine.

Checkpoint: ``NPR.pth`` is a wrapped dict ``{model, optimizer, total_steps}``
(see ``networks/base_model.py``; the file size of the committed checkpoint
matches model + Adam optimizer state); ``test.py``/bare exports load the state
dict directly.  Both forms are handled here, plus ``module.``/``net.``
prefixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MAX_SIDE = 1024
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _native_crop(pil, max_side=MAX_SIDE, multiple=2):
    """Whole image at native resolution; sides floored to ``multiple`` and
    centre-cropped to at most ``max_side`` (no resampling)."""
    import numpy as np

    pil = pil.convert("RGB")
    w, h = pil.size
    cw = (min(w, max_side) // multiple) * multiple
    ch = (min(h, max_side) // multiple) * multiple
    cw, ch = max(cw, multiple), max(ch, multiple)
    left, top = (w - cw) // 2, (h - ch) // 2
    pil = pil.crop((left, top, left + cw, top + ch))
    return np.asarray(pil, dtype="float32").transpose(2, 0, 1) / 255.0


def _architecture():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False)

    def conv1x1(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         bias=False)

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
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out

    class NPRNet(nn.Module):
        def __init__(self, num_classes=1):
            super().__init__()
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
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

        @staticmethod
        def _interpolate(img, factor):
            return F.interpolate(
                F.interpolate(img, scale_factor=factor, mode='nearest',
                              recompute_scale_factor=True),
                scale_factor=1 / factor, mode='nearest',
                recompute_scale_factor=True)

        @staticmethod
        def preprocess(pil):
            """Full frame, native resolution, ImageNet-normalised (1, 3, H, W)."""
            import numpy as np

            arr = _native_crop(pil)
            arr = (arr - np.asarray(MEAN, dtype="float32")[:, None, None]) / \
                np.asarray(STD, dtype="float32")[:, None, None]
            return torch.from_numpy(arr).unsqueeze(0)

        def forward_detailed(self, x):
            npr = x - self._interpolate(x, 0.5)
            x = self.conv1(npr * 2.0 / 3.0)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            feats = self.layer2(x)
            pooled = self.avgpool(feats)
            logit = self.fc1(pooled.view(pooled.size(0), -1))
            logits = torch.cat([torch.zeros_like(logit), logit], dim=1)
            return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                    "feat": feats}

        def forward(self, x):
            return self.forward_detailed(x)["logits"]

    return NPRNet


def build_model(weights_path, device):
    import torch

    NPRNet = _architecture()
    model = NPRNet(num_classes=1)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "net"):
                if isinstance(state.get(key), dict):
                    state = state[key]
                    break
            if all(k.startswith("module.") for k in state):
                state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
