"""X-CLIP detector (Yan et al. 2023), X-CLIP ViT-B/32 vision tower + head.

Ported from DeepfakeBench ``training/detectors/xclip_detector.py``; the
detector is the ``XCLIPVisionModel`` of ``microsoft/xclip-base-patch32``
(768-d), run per-frame, mean-pooled across frames, passed through an extra
LayerNorm, and fed to a 2-way linear head. No detector checkpoint is released,
so without a ``weights_path`` the head is randomly initialized.

Input is a (B, 3, T, H, W) clip; frames are reordered to the (B, T, 3, H, W)
layout the DeepfakeBench source assumes and flattened to (B*T, 3, H, W) for
the per-frame vision model. Frames are processed independently, so T is not
constrained by the checkpoint (the config uses T=32), but each frame must be
224x224 (patch_size 32).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class XCLIPDetector(nn.Module):
    def __init__(self, num_classes=2, model_name="microsoft/xclip-base-patch32"):
        super().__init__()
        from transformers import XCLIPVisionModel

        self.backbone = XCLIPVisionModel.from_pretrained(model_name)
        self.fc_norm = nn.LayerNorm(768)
        self.head = nn.Linear(768, num_classes)

    def features(self, x):
        batch_size, _, num_frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, 3, height, width)  # (B*T,3,H,W)
        outputs = self.backbone(x, output_hidden_states=True)
        sequence_output = outputs["pooler_output"].reshape(batch_size, num_frames, -1)
        return self.fc_norm(sequence_output.mean(1))

    def classifier(self, feats):
        return self.head(feats)

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = XCLIPDetector(num_classes=2, model_name="microsoft/xclip-base-patch32")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
