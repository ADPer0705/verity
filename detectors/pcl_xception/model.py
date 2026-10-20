"""PCL-I2G detector (Zhao et al. 2021): Xception-SLADD backbone with pair-wise
consistency learning.

Ported from DeepfakeBench ``training/detectors/pcl_xception_detector.py``.
The yaml selects the ``xception_sladd`` backbone; at inference the PCL
non-local block, mask transform and BCE loss are unused, so the forward is a
plain ``backbone.features -> backbone.classifier`` chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception_sladd import Xception_SLADD  # noqa: E402


class PCLXceptionDetector(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False):
        super().__init__()
        self.backbone = Xception_SLADD(num_classes=num_classes, inc=inc, dropout=dropout)

    def features(self, x):
        return self.backbone.features(x)

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward_detailed(self, x):
        feats, x3 = self.features(x)
        logits, pooled = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = PCLXceptionDetector(num_classes=2, inc=3, dropout=False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
