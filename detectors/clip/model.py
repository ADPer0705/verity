"""CLIP detector (Yan et al. 2023), CLIP ViT-B/16 vision tower + linear head.

Ported from DeepfakeBench ``training/detectors/clip_detector.py``; the
detector is the ``vision_model`` of ``openai/clip-vit-base-patch16`` (768-d)
followed by a 2-way linear head. No detector checkpoint is released, so
without a ``weights_path`` the head is randomly initialized.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CLIPDetector(nn.Module):
    def __init__(self, num_classes=2, model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        from transformers import CLIPModel

        self.backbone = CLIPModel.from_pretrained(model_name).vision_model
        self.head = nn.Linear(768, num_classes)

    def features(self, x):
        return self.backbone(x)["pooler_output"]

    def classifier(self, feats):
        return self.head(feats)

    def forward_detailed(self, x):
        captured = {}

        def _pre_hook(_module, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return None
            hs = args[0]
            if hs.ndim != 3 or hs.shape[1] <= 1:
                return None
            cls_tok = hs[:, :1]
            patches = hs[:, 1:]
            side = int(round(patches.shape[1] ** 0.5))
            if side * side != patches.shape[1]:
                return None
            feat_map = patches.permute(0, 2, 1).reshape(
                patches.shape[0], patches.shape[2], side, side
            )
            captured["feat"] = feat_map
            hs_new = torch.cat(
                [cls_tok, feat_map.flatten(2).permute(0, 2, 1)], dim=1
            )
            return (hs_new,) + args[1:]

        handle = None
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
            handle = self.backbone.encoder.layers[-1].register_forward_pre_hook(_pre_hook)
        try:
            feats = self.features(x)
        finally:
            if handle is not None:
                handle.remove()

        logits = self.classifier(feats)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": captured.get("feat", feats),
        }


def build_model(weights_path, device):
    model = CLIPDetector(num_classes=2, model_name="openai/clip-vit-base-patch16")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
