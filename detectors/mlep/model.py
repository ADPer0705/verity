"""MLEP detector (NeurIPS 2025): Multi-granularity Local Entropy Patterns for
universal AI-generated image detection (Li et al., fkeufss/MLEP).

Ported from the official repo ``networks/resnet.py``.  The core pipeline:
split the input into 2x2 blocks, randomly shuffle the blocks (fixed seed 99,
one permutation per image), bilinearly resample at scales 1/2 and 1/4 (down
and back up), then per RGB channel compute a local entropy map over
overlapping 2x2 windows quantized to {0.0, 0.8, 1.0, 1.5, 2.0}.  The three
entropy maps are concatenated into 9 channels and classified by a truncated
ResNet50 (conv1 + layer1 + layer2 -> 512-d) with a single logit trained with
BCE (label 1 = fake).  The official checkpoint stores the raw state dict
(keys ``conv1.*``, ``layer1.*``, ``layer2.*``, ``fc1.*``); layer3/layer4 are
not part of the architecture.

Preprocessing (``input_mode: full_frame``): MLEP reuses NPR's code base; its
``test.py`` sets ``no_crop=True`` everywhere and ``no_resize=True`` for
GAN-set-2 (native resolution), with ``Resize((256, 256))`` on the other sets
only for batching ("due to the different shapes of images"), where images are
~256 px anyway.  The local-entropy patterns are computed on 2x2 pixel windows,
which resampling would alter, so ``preprocess`` follows the
``no_resize + no_crop`` path: whole frame at native resolution, ToTensor
([0, 1]), sides floored to even numbers (2x2 block shuffle) and centre-cropped
to at most ``MAX_SIDE`` (1024) px per side as a memory guard.  The yaml
mean/std are neutral (0,0,0)/(1,1,1); the model's own ImageNet normalization
is applied inside ``forward_detailed`` (``data/datasets.py`` Normalize).  The
2-class logits are ``[-s/2, +s/2]`` so that ``softmax(logits)[:, 1]`` equals
``sigmoid(s)``, the reference fake score.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MAX_SIDE = 1024


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class MLEPDetector(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _patch_shuffle(self, x, block_size=2):
        """Split into ``block_size`` blocks and shuffle them per image.

        Matches the reference: ``torch.manual_seed(99)`` once, then one
        ``torch.randperm`` per batch image (CPU generator, indexing a
        device tensor).  The gather is out-of-place so gradients survive
        Grad-CAM's ``enable_grad`` forward.
        """
        batch, channels, H, W = x.size()
        H_cropped = (H // block_size) * block_size
        W_cropped = (W // block_size) * block_size
        x = x[:, :, :H_cropped, :W_cropped]
        blocks = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        num_blocks_H = blocks.size(2)
        num_blocks_W = blocks.size(3)
        blocks = blocks.contiguous().view(batch, channels, num_blocks_H * num_blocks_W, block_size, block_size)
        torch.manual_seed(99)
        shuffled = []
        for i in range(batch):
            permuted_indices = torch.randperm(num_blocks_H * num_blocks_W)
            shuffled.append(blocks[i, :, permuted_indices, :, :])
        blocks = torch.stack(shuffled, dim=0)
        blocks = blocks.view(batch, channels, num_blocks_H, num_blocks_W, block_size, block_size)
        blocks = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
        return blocks.view(batch, channels, H_cropped, W_cropped)

    def _local_entropy(self, x):
        """Shannon entropy of every overlapping 2x2 window (stride 1), per channel.

        Quantized exactly as in the reference: all-same 0.0, three-same 0.8,
        two equal pairs 1.0, two same + two different 1.5, all-different 2.0.
        """
        batch, channels, H, W = x.size()
        patches = x.unfold(2, 2, 1).unfold(3, 2, 1).contiguous()
        patches = patches.view(batch, channels, (H - 1) * (W - 1), 4)
        p0 = patches[:, :, :, 0]
        p1 = patches[:, :, :, 1]
        p2 = patches[:, :, :, 2]
        p3 = patches[:, :, :, 3]
        all_same = (p0 == p1) & (p1 == p2) & (p2 == p3)
        three_same = ((p0 == p1) & (p1 == p2) & (p2 != p3)) | \
                     ((p0 == p1) & (p1 == p3) & (p2 != p3)) | \
                     ((p0 == p2) & (p2 == p3) & (p1 != p2)) | \
                     ((p1 == p2) & (p2 == p3) & (p0 != p1))
        two_pairs = ((p0 == p1) & (p2 == p3) & (p1 != p2)) | \
                    ((p0 == p2) & (p1 == p3) & (p0 != p1)) | \
                    ((p0 == p3) & (p1 == p2) & (p0 != p1))
        two_same_two_diff = ((p0 == p1) & (p2 != p3) & (p1 != p2) & (p1 != p3)) | \
                            ((p0 == p2) & (p1 != p3) & (p0 != p1) & (p0 != p3)) | \
                            ((p0 == p3) & (p1 != p2) & (p0 != p1) & (p0 != p2)) | \
                            ((p1 == p2) & (p0 != p3) & (p1 != p0) & (p1 != p3)) | \
                            ((p1 == p3) & (p0 != p2) & (p1 != p0) & (p1 != p2)) | \
                            ((p2 == p3) & (p0 != p1) & (p2 != p0) & (p2 != p1))
        all_diff = (p0 != p1) & (p0 != p2) & (p0 != p3) & \
                   (p1 != p2) & (p1 != p3) & (p2 != p3)
        device = x.device
        entropy_image = torch.zeros_like(p0)
        entropy_image = torch.where(three_same, torch.tensor(0.8, device=device), entropy_image)
        entropy_image = torch.where(two_pairs, torch.tensor(1.0, device=device), entropy_image)
        entropy_image = torch.where(two_same_two_diff, torch.tensor(1.5, device=device), entropy_image)
        entropy_image = torch.where(all_diff, torch.tensor(2.0, device=device), entropy_image)
        return entropy_image.view(batch, channels, H - 1, W - 1)

    def _entropy_stack(self, x):
        """Shuffled 2x2 blocks -> multi-scale resampling -> 9-channel entropy stack."""
        x_rearrange = self._patch_shuffle(x, 2)
        x_down_2 = F.interpolate(x_rearrange, scale_factor=0.5, mode="bilinear", align_corners=False)
        x_up_2 = F.interpolate(x_down_2, size=x_rearrange.shape[2:], mode="bilinear", align_corners=False)
        x_down_4 = F.interpolate(x_rearrange, scale_factor=0.25, mode="bilinear", align_corners=False)
        x_up_4 = F.interpolate(x_down_4, size=x_rearrange.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([
            self._local_entropy(x_rearrange),
            self._local_entropy(x_up_2),
            self._local_entropy(x_up_4),
        ], dim=1)

    @staticmethod
    def preprocess(pil):
        """Full frame, native resolution, float [0, 1] (1, 3, H, W); even
        sides, centre crop to at most MAX_SIDE px."""
        import numpy as np

        pil = pil.convert("RGB")
        w, h = pil.size
        cw, ch = max(2, (min(w, MAX_SIDE) // 2) * 2), max(2, (min(h, MAX_SIDE) // 2) * 2)
        left, top = (w - cw) // 2, (h - ch) // 2
        pil = pil.crop((left, top, left + cw, top + ch))
        arr = np.asarray(pil, dtype="float32").transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def forward(self, x):
        return self.forward_detailed(x)["logits"]

    def forward_detailed(self, x):
        x = (x - self.mean) / self.std
        entropy = self._entropy_stack(x)
        h = self.maxpool(self.relu(self.bn1(self.conv1(entropy))))
        h = self.layer1(h)
        feat = self.layer2(h)
        s = self.fc1(self.avgpool(feat).view(feat.size(0), -1))
        logits = torch.cat([-s, s], dim=1) * 0.5
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feat}


def build_model(weights_path, device):
    model = MLEPDetector(num_classes=1)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        state.pop("_metadata", None)
        state.pop("version", None)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
