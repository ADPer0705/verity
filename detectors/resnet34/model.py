"""ResNet-34 detector (the CNN-Augmentation baseline, Bonettini et al. 2021).

Ported from DeepfakeBench ``training/detectors/resnet34_detector.py``.  The
backbone is a torchvision ResNet-34 body (up to ``layer4``) with a fresh
512-to-2 classifier.  The checkpoint (``cnnaug_best.pth``) is the full detector
state dict (keys ``backbone.resnet.*``, ``backbone.avgpool``,
``backbone.fc``); no ImageNet download is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision.models import resnet34  # noqa: E402


class ResnetDetector(nn.Module):
    def __init__(self, num_classes=2, inc=3, mode="Original"):
        super().__init__()
        self.mode = mode
        resnet = resnet34(weights=None)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        x = self.resnet(x)
        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, feats):
        x = self.avgpool(feats)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = ResnetDetector(num_classes=2, inc=3, mode="Original")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        # cnnaug_best.pth is the DeepfakeBench detector state dict, whose
        # modules live under ``backbone.``; strip it so the keys match
        # ``resnet.*`` / ``fc.*`` here (otherwise strict=False loads nothing).
        if all(k.startswith("backbone.") for k in state.keys()):
            state = {k[len("backbone."):]: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[resnet34] state dict mismatch: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected (e.g. {list(missing)[:2]} / "
                  f"{list(unexpected)[:2]})")
    model.to(device)
    model.eval()
    return model
