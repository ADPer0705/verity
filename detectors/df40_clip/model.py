"""DF40 CLIP detector (NeurIPS 2024 D&B, Yan et al., arXiv 2406.13495).

Ported from ``DeepfakeBench_DF40/training/detectors/clip_detector.py`` in
YZY-stack/DF40: the ``vision_model`` of ``openai/clip-vit-large-patch14``
(ViT-L/14, 224 px) fully fine-tuned end to end, followed by ``nn.Linear(1024, 2)``
on the pooler output. Upstream normalises with mean/std 0.5 (``clip.yaml``),
not the CLIP statistics; the engine applies that from detector.yaml.

The released ``clip_large.pth`` checkpoints carry the whole fine-tuned tower
plus the head; the HF backbone is only used to build the architecture and is
then overwritten by a strict ``load_state_dict``.

Attention map: the mean over heads of the last layer's CLS-token attention
over the 16x16 patch grid is returned as ``attention_map``.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

CLIP_MODEL = "openai/clip-vit-large-patch14"


class DF40CLIPDetector(nn.Module):
    def __init__(self, num_classes=2, model_name=CLIP_MODEL):
        super().__init__()
        from transformers import CLIPModel

        self.backbone = CLIPModel.from_pretrained(model_name, attn_implementation="eager").vision_model
        self.head = nn.Linear(1024, num_classes)

    def features(self, x):
        return self.backbone(x)["pooler_output"]

    def classifier(self, feats):
        return self.head(feats)

    def forward_detailed(self, x):
        out = self.backbone(x, output_attentions=True)
        feats = out.pooler_output
        logits = self.head(feats)
        result = {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}
        attn = out.attentions[-1] if out.attentions else None
        if attn is not None:
            cls_attn = attn[:, :, 0, 1:].mean(dim=1)  # (B, n_patches), mean over heads
            side = int(round(cls_attn.shape[1] ** 0.5))
            if side * side == cls_attn.shape[1]:
                result["attention_map"] = cls_attn.reshape(-1, side, side)
        return result

    def forward(self, x):
        return self.head(self.features(x))


def _load_checkpoint(model: nn.Module, weights_path: str) -> None:
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    shape_mismatch = [k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or shape_mismatch:
        print(f"[df40_clip] checkpoint mismatch: missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
              f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''} "
              f"shape={shape_mismatch[:10]}", file=sys.stderr)
        raise RuntimeError(f"DF40 CLIP checkpoint {weights_path} does not match the model "
                           f"({len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{len(shape_mismatch)} shape mismatches)")
    model.load_state_dict(state, strict=True)


def build_model(weights_path, device):
    model = DF40CLIPDetector(num_classes=2)
    if weights_path is not None and os.path.isfile(weights_path):
        _load_checkpoint(model, weights_path)
    elif weights_path is not None:
        print(f"[df40_clip] WARNING: checkpoint {weights_path} not found; running with the "
              f"pretrained CLIP tower and a RANDOM head (not evidence-grade). See weights/README.md.",
              file=sys.stderr)
    model.to(device)
    model.eval()
    return model
