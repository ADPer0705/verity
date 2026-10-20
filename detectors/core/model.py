"""Core detector (Yan et al. 2023), a plain Xception classifier.

Ported from DeepfakeBench ``training/detectors/core_detector.py``; the
checkpoint (``core_best.pth``) is the full detector state dict (keys
``backbone.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


class CoreDetector(nn.Module):
    def __init__(self, num_classes=2, mode="original", inc=3, dropout=False):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes, mode=mode, inc=inc, dropout=dropout)

    def features(self, x):
        return self.backbone.features(x)

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = CoreDetector(num_classes=2, mode="original", inc=3, dropout=False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
