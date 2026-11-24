"""ProDet detector (Cheng, Yan, Zhang, Luo, Wang, Li, NeurIPS 2024, "Can We
Leave Deepfake Data Behind in Training Deepfake Detector?", arXiv 2408.17052).

Ported from ``demo/prodet_inference.py`` (``ProDet_infer``) of
github.com/beautyremain/ProDet, the standalone inference path of the
DeepfakeBench-based training code.  Backbone: DeepfakeBench's EfficientNet-B4
(efficientnet_pytorch ``from_name``, stem replaced by a plain
``Conv2d(3, 48, 3, 2)`` as in every DeepfakeBench port) whose 1792-channel map
is reduced to 512 channels (``adjust_feature``).  Three auxiliary classifiers
(deepfake / blend / identity-inconsistency, each AvgPool -> Linear(512, 256)
-> LeakyReLU -> Linear(256, 2)) feed a ``FeatureAttentionBlock`` that inverts
their 2-logit outputs back to 512x8x8 maps (Linear 2->256->512 +
ConvTranspose2d 8x8), concatenates them with the feature map and produces a
sigmoid attention mask; the masked features go to ``final_classifier``
(AvgPool -> 512 -> 256 -> 2).  P(fake) = ``softmax(final)[:, 1]``.

Checkpoint ``ProDet_best.pth`` (Google Drive): flat state dict (782 tensors)
holding the backbone, the heads and the training-only latent-space
mix-up generators (``real2sbi``, ``bi2fake``), which are dropped; everything
else loads strictly.  Preprocessing follows
``training/config/prodet.yaml``: 256x256 face crop, mean/std 0.5.
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
    """DeepfakeBench ``networks/efficientnetb4.py`` (mode "original")."""

    def __init__(self, num_classes=2, inc=3):
        super().__init__()
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, num_classes)

    def features(self, x):
        return self.efficientnet.extract_features(x)


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def _inverse_conv():
    return nn.Sequential(
        nn.Linear(2, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 512),
        Lambda(lambda x: x.view(-1, 512, 1, 1)),
        nn.ConvTranspose2d(512, 512, kernel_size=(8, 8)),
    )


class FeatureAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inverse_conv1 = _inverse_conv()
        self.inverse_conv2 = _inverse_conv()
        self.inverse_conv3 = _inverse_conv()
        self.attention = nn.Sequential(
            nn.Conv2d(512 * 4, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2, x3, feature):
        x1 = self.inverse_conv1(x1)
        x2 = self.inverse_conv2(x2)
        x3 = self.inverse_conv3(x3)
        att = self.attention(torch.cat((x1, x2, x3, feature), dim=1))
        return feature * att, att


def _head():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.LeakyReLU(inplace=True),
        nn.Linear(256, 2),
    )


class ProDetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNetB4(num_classes=2, inc=3)
        self.adjust_feature = nn.Conv2d(1792, 512, 1)
        self.blend_classifier = _head()
        self.fea_att = FeatureAttentionBlock()
        self.ID_inconsistency_classifier = _head()
        self.deepfake_classifier = _head()
        self.final_classifier = _head()

    def features(self, x):
        return self.adjust_feature(self.backbone.features(x))

    def forward_detailed(self, x):
        features = self.features(x)
        df_pred = self.deepfake_classifier(features)
        bld_pred = self.blend_classifier(features)
        bi_pred = self.ID_inconsistency_classifier(features)
        att_features, att = self.fea_att(df_pred, bld_pred, bi_pred, features)
        logits = self.final_classifier(att_features)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": att_features,
            "attention_map": att.mean(dim=1),
            "aux": {
                "deepfake": torch.softmax(df_pred, dim=1)[:, 1],
                "blend": torch.softmax(bld_pred, dim=1)[:, 1],
                "identity_inconsistency": torch.softmax(bi_pred, dim=1)[:, 1],
            },
        }

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = ProDetDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        # real2sbi.* / bi2fake.* are the training-time feature mix-up generators.
        state = {k: v for k, v in state.items() if not k.startswith(("real2sbi.", "bi2fake.", "sbi2bi."))}
        own = model.state_dict()
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise RuntimeError(f"ProDet checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
