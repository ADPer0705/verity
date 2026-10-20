"""FFD detector (Dang et al. 2020): Xception + regression attention map.

Ported from DeepfakeBench ``training/detectors/ffd_detector.py``.  The
regression map (a 1x1 separable conv over the block7 features, followed by a
sigmoid) is the model's native attention map; we expose it for explainability.
The checkpoint (``ffd_best.pth``) is the full detector state dict (keys
``backbone.*`` and ``map.c.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception import Xception, SeparableConv2d  # noqa: E402


class RegressionMap(nn.Module):
    def __init__(self, c_in=728):
        super().__init__()
        self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
        self.s = nn.Sigmoid()

    def forward(self, x):
        mask = self.s(self.c(x))
        return mask, None


class FFDDetector(nn.Module):
    def __init__(self, num_classes=2, mode="original", inc=3, dropout=False):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes, mode=mode, inc=inc, dropout=dropout)
        self.map = RegressionMap(728)

    def features(self, x):
        x = self.backbone.fea_part1(x)
        x = self.backbone.fea_part2(x)
        x = self.backbone.fea_part3(x)
        mask, vec = self.map(x)
        x = x * mask
        x = self.backbone.fea_part4(x)
        x = self.backbone.fea_part5(x)
        return x, mask, vec

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward(self, x):
        feats, _, _ = self.features(x)
        return self.classifier(feats)

    def forward_detailed(self, x):
        feats, mask, _ = self.features(x)
        logits = self.classifier(feats)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": feats,
            "attention_map": mask,
        }


def build_model(weights_path, device):
    model = FFDDetector(num_classes=2, mode="original", inc=3, dropout=False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
