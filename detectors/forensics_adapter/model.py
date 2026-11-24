"""Forensics Adapter detector (Cui, Li, Luo, Zhou, Dong, CVPR 2025,
"Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection").

Ported from ``model/ds.py`` (``DS``) + ``model/adapters/adapter.py`` +
``model/attn.py`` + ``model/layer.py`` + the modified ``model/clip`` of
github.com/OUC-VAS/ForensicsAdapter (CC BY-NC 4.0); shared modules live in
``detectors/common/backbones`` (``clip_openai``, ``forensics_adapter``).
The released checkpoint (Google Drive) is the whole ``DS`` state dict
including the frozen CLIP ViT-L/14 (fp16), loaded strictly.

Inference path (``DS.forward``):

1. CLIP ViT-L/14 features of the 224-resized face at layers 0/1/8/15.
2. ``Adapter``: timm ViT-tiny trunk with 128 query tokens over the 256x256
   face; CLIP features are *fused* into the trunk after blocks 1-3 through a
   ``Fusion`` module that upstream re-instantiates with random weights at
   every forward (``x += 0.01 * proj(LN(clip_x))``) - reproduced with a
   fixed seed; the ``Mask_Decoder`` yields the blending-boundary prediction
   and a per-head attention bias.
3. ``RecAttnClip``: the frozen CLIP blocks re-run on ``[queries, CLS,
   patches]`` with the bias as attention mask (intra block after layer 6),
   giving projected, normalised query tokens.
4. ``PostClipProcess`` (Conv1d over queries + MLP) -> 2 logits, P(fake) =
   ``softmax[:, 1]``.

The boundary map (``xray_pred`` through ``MaskPostXrayProcess``; upstream
additionally masks it with ground-truth boundary patches at test time,
which no investigator has) is returned as the native attention map.  Input:
256x256 face crop with CLIP mean/std (``config/test.yaml``); trained on
FaceForensics++ c23 (DeepfakeBench crops).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.backbones.clip_openai import build_clip  # noqa: E402
from common.backbones.forensics_adapter import (Adapter, MaskPostXrayProcess, PostClipProcess,  # noqa: E402
                                                RecAttnClip)

NUM_QUERIES = 128
FUSION_MAP = {0: 0, 1: 1, 2: 8, 3: 15}


class ForensicsAdapterDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = build_clip()
        self.adapter = Adapter("vit_tiny_patch16_224", NUM_QUERIES, FUSION_MAP, 256, 128, 16,
                               fuse=True, fuse_scale=0.01)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, NUM_QUERIES, intra_layer=6, return_patches=False)
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=NUM_QUERIES)
        self.clip_post_process = PostClipProcess(NUM_QUERIES, 768)

    def forward_detailed(self, x):
        clip_images = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        clip_features = self.clip_model.extract_features(clip_images, FUSION_MAP.values())
        attn_biases, xray_preds = self.adapter(x, clip_features)
        clip_output = self.rec_attn_clip(clip_features, attn_biases[-1], normalize=True)
        logits = self.clip_post_process(clip_output.float())
        boundary = self.masked_xray_post_process(xray_preds[-1])
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": clip_output,
                "attention_map": torch.sigmoid(boundary[:, 0])}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = ForensicsAdapterDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
        own = model.state_dict()
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise RuntimeError(f"Forensics Adapter checkpoint mismatch: missing={missing[:8]} "
                               f"unexpected={unexpected[:8]}")
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
