"""F3Net detector (Qian et al. 2020): FAD (frequency-aware) head + Xception.

Ported from DeepfakeBench ``training/detectors/f3net_detector.py``.  The FAD
head splits the input into 4 DCT-filtered frequency bands (12 channels total);
the backbone's first conv is widened to 12 input channels (initialized by
splitting the 3-channel ImageNet weights).  The checkpoint (``f3net_best.pth``)
is the full detector state dict (keys ``FAD_head.*`` and ``backbone.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


def _dct_mat(size):
    return [[(np.sqrt(1.0 / size) if i == 0 else np.sqrt(2.0 / size))
             * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)]
            for i in range(size)]


def _generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)]
            for i in range(size)]


def _norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True):
        super().__init__()
        self.use_learnable = use_learnable
        self.base = nn.Parameter(
            torch.tensor(_generate_filter(band_start, band_end, size)),
            requires_grad=False,
        )
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size))
            self.learnable.data.normal_(0., 0.1)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + _norm_sigma(self.learnable)
        else:
            filt = self.base
        return x * filt


class FADHead(nn.Module):
    def __init__(self, size):
        super().__init__()
        dct = torch.tensor(_dct_mat(size)).float()
        self._DCT_all = nn.Parameter(dct, requires_grad=False)
        self._DCT_all_T = nn.Parameter(dct.transpose(0, 1), requires_grad=False)
        self.filters = nn.ModuleList([
            Filter(size, 0, size / 2.82),
            Filter(size, size / 2.82, size / 2),
            Filter(size, size / 2, size * 2),
            Filter(size, 0, size * 2),
        ])

    def forward(self, x):
        x_freq = self._DCT_all @ x @ self._DCT_all_T
        y_list = []
        for f in self.filters:
            x_pass = f(x_freq)
            y_list.append(self._DCT_all_T @ x_pass @ self._DCT_all)
        return torch.cat(y_list, dim=1)


class F3netDetector(nn.Module):
    def __init__(self, num_classes=2, mode="original", inc=3, dropout=0.5,
                 resolution=256):
        super().__init__()
        self.backbone = Xception(num_classes=num_classes, mode=mode, inc=inc,
                                 dropout=dropout)
        self.FAD_head = FADHead(resolution)

    def features(self, x):
        fea_FAD = self.FAD_head(x)
        return self.backbone.features(fea_FAD)

    def classifier(self, feats):
        return self.backbone.classifier(feats)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = F3netDetector(num_classes=2, mode="original", inc=12, dropout=0.5,
                          resolution=256)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
