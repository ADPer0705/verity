"""LSDA detector (Yan et al. 2023): EfficientNet-B4 student encoder with
latent-space domain augmentation.

Ported from DeepfakeBench ``training/detectors/lsda_detector.py``.  At
inference the generator's latent augmentation (encoders_f, encoder_c
/iresnet100, fc_weights, mlp, mixup/augment transforms) is unused; only the
student encoder + binary classifier run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class efficientnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_name("efficientnet-b4")
        self.conv = nn.Conv2d(1792, 512, 1)

    def features(self, x):
        x = self.model.extract_features(x)
        x = self.conv(x)
        return x


class LSDADetector(nn.Module):
    def __init__(self, num_classes=2, encoder_feat_dim=512):
        super().__init__()
        self.encoder_feat_dim = encoder_feat_dim
        self.student_encoder = efficientnet()
        self.binary_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.encoder_feat_dim, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def features(self, x):
        return self.student_encoder.features(x)

    def classifier(self, fea):
        return self.binary_classifier(fea)

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = LSDADetector(num_classes=2, encoder_feat_dim=512)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
