"""SLADD detector (Chen et al. 2022): Xception-SLADD backbone with a
self-supervised adversarial synthesizer.

Ported from DeepfakeBench ``training/detectors/sladd_detector.py``.  The
synthesizer (``utils/sladd_api.py``), its optimizer and the mask/type/mag
auxiliary losses are training-only; at inference only the backbone's
``features -> classifier`` chain runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception_sladd import Xception_SLADD  # noqa: E402


class SLADDXceptionDetector(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False):
        super().__init__()
        self.backbone = Xception_SLADD(num_classes=num_classes, inc=inc, dropout=dropout)

    def features(self, imgs):
        return self.backbone.features(imgs)

    def classifier(self, features):
        return self.backbone.classifier(features)

    def forward_detailed(self, x):
        feats, map_fea = self.features(x)
        logits, pooled = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = SLADDXceptionDetector(num_classes=2, inc=3, dropout=False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
