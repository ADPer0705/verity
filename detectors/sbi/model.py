"""SBI detector (Shiohara & Yamasaki 2022): EfficientNet-B4 classifier.

Ported from DeepfakeBench ``training/detectors/sbi_detector.py``.  The
self-blended-image synthesizer only runs during training; at inference the path
is plain backbone features + classifier with a 3-class head (per
``config/detector/sbi.yaml`` ``num_classes: 3``), and the fake probability is
``softmax(logits)[:, 1]`` exactly as in the source.  The checkpoint is the full
detector state dict (keys ``backbone.efficientnet.*``, ``backbone.last_layer.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=3, inc=3, dropout=False, mode="original"):
        super().__init__()
        self.dropout = dropout
        self.mode = mode
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.last_layer = nn.Linear(1792, num_classes)
        if mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        x = self.efficientnet.extract_features(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, feats):
        x = F.adaptive_avg_pool2d(feats, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout:
            x = self.dropout_layer(x)
        self.last_emb = x
        return self.last_layer(x)


class SBIDetector(nn.Module):
    def __init__(self, num_classes=3, inc=3, dropout=False, mode="original"):
        super().__init__()
        self.backbone = EfficientNetB4(num_classes=num_classes, inc=inc, dropout=dropout, mode=mode)

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
    model = SBIDetector(num_classes=3, inc=3, dropout=False, mode="original")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
