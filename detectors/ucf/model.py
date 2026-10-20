"""UCF detector (Yan et al. 2023): dual-encoder Xception with fingerprint
separation into shared (forgery) and specific features.

Ported from DeepfakeBench ``training/detectors/ucf_detector.py``.  Inference
uses only the shared branch (``head_sha``); the ``Con2d1x1`` fingerprint maps
and the shared head's pooled embedding are exposed for explainability.  The
checkpoint (``ucf_best.pth``) is the full detector state dict; the training-only
``Conditional_UNet`` keys are ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.backbones.xception import Xception  # noqa: E402


class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_f, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_f, 1, 1),
        )

    def forward(self, x):
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super().__init__()
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_f, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_f),
        )

    def forward(self, x):
        bs = x.size()[0]
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat


class UCFDetector(nn.Module):
    def __init__(self, num_classes=2, encoder_feat_dim=512,
                 specific_task_number=5):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_feat_dim = encoder_feat_dim
        self.half_fingerprint_dim = encoder_feat_dim // 2

        backbone_kwargs = dict(num_classes=num_classes, mode="adjust_channel",
                               inc=3, dropout=False)
        self.encoder_f = Xception(**backbone_kwargs)
        self.encoder_c = Xception(**backbone_kwargs)

        self.head_spe = Head(in_f=self.half_fingerprint_dim,
                             hidden_dim=self.encoder_feat_dim,
                             out_f=specific_task_number)
        self.head_sha = Head(in_f=self.half_fingerprint_dim,
                             hidden_dim=self.encoder_feat_dim,
                             out_f=self.num_classes)
        self.block_spe = Conv2d1x1(in_f=self.encoder_feat_dim,
                                   hidden_dim=self.half_fingerprint_dim,
                                   out_f=self.half_fingerprint_dim)
        self.block_sha = Conv2d1x1(in_f=self.encoder_feat_dim,
                                   hidden_dim=self.half_fingerprint_dim,
                                   out_f=self.half_fingerprint_dim)

    def features(self, x):
        f_all = self.encoder_f.features(x)
        c_all = self.encoder_c.features(x)
        return {"forgery": f_all, "content": c_all}

    def classifier(self, feats):
        f_spe = self.block_spe(feats)
        f_share = self.block_sha(feats)
        return f_spe, f_share

    def forward_detailed(self, x):
        feats = self.features(x)
        forgery = feats["forgery"]
        f_spe, f_share = self.classifier(forgery)
        out_sha, sha_feat = self.head_sha(f_share)
        return {
            "logits": out_sha,
            "probs": torch.softmax(out_sha, dim=1),
            "feat": sha_feat,
            "attention_map": f_share,
        }

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = UCFDetector(num_classes=2, encoder_feat_dim=512,
                        specific_task_number=5)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
