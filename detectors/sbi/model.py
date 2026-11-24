"""SBI detector (Shiohara & Yamasaki, CVPR 2022, "Detecting Deepfakes with
Self-Blended Images"): the *official* inference architecture.

Ported from ``src/model.py`` + ``src/inference/inference_image.py`` of
github.com/mapooon/SelfBlendedImages.  ``Detector.net`` is
``EfficientNet.from_pretrained("efficientnet-b4", advprop=True, num_classes=2)``
(efficientnet_pytorch), i.e. a plain EfficientNet-B4 with a 2-way ``_fc``
head.  The released checkpoints (``FFraw.tar`` / ``FFc23.tar``) are
``{"model": state_dict, "optimizer": ..., "epoch": ...}`` with keys ``net.*``
and are loaded with ``strict=True`` onto ``self.net``.

Preprocessing (``inference_image.py`` / ``preprocess.py::crop_face``): the
RetinaFace box is enlarged by w/8 on each side (``w/4 * 0.5``, i.e. 25 %
overall; ``detector.yaml`` ``face_margin: 0.25``), resized to 380x380 with
``cv2.resize`` (no aspect preservation) and fed as float RGB in [0, 1] with
**no mean/std normalisation**: the yaml mean 0 / std 1 are neutral.  The
self-blending synthesiser is training-only.  ``softmax(logits)[:, 1]`` is
P(fake) upstream, so no reordering is needed.  Grad-CAM uses the 1792-channel
``extract_features`` map.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class SBIDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)

    def features(self, x):
        return self.net.extract_features(x)

    def classifier(self, feats):
        x = self.net._avg_pooling(feats).flatten(start_dim=1)
        x = self.net._dropout(x)
        return self.net._fc(x)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = SBIDetector()
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
