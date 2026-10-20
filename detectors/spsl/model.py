"""SPSL detector (Liu et al. 2021): phase-spectrum input + Xception.

Ported from DeepfakeBench ``training/detectors/spsl_detector.py``.  The
grayscale phase (reconstructed from the DFT phase spectrum) is concatenated
with the RGB input into a 4-channel tensor; the backbone's first conv is
widened to 4 input channels.  The checkpoint (``spsl_best.pth``) is the full
detector state dict (keys ``backbone.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


class SPSLDetector(nn.Module):
    def __init__(self, num_classes=2, mode="original", inc=4, dropout=False):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes, mode=mode, inc=inc, dropout=dropout)

    @staticmethod
    def phase_without_amplitude(img):
        gray_img = torch.mean(img, dim=1, keepdim=True)
        X = torch.fft.fftn(gray_img, dim=(-1, -2))
        phase_spectrum = torch.angle(X)
        reconstructed_X = torch.exp(1j * phase_spectrum)
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        return reconstructed_x

    def preprocess(self, x):
        phase = self.phase_without_amplitude(x)
        return torch.cat([x, phase], dim=1)

    def features(self, x):
        x = self.preprocess(x)
        return self.backbone.features(x)

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        x4 = self.preprocess(x)
        feats = self.backbone.features(x4)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = SPSLDetector(num_classes=2, mode="original", inc=4, dropout=False)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
