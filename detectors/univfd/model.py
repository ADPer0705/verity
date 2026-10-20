"""UnivFD detector (CVPR 2023): frozen CLIP ViT-L/14 + linear probe.

Ported from WisconsinAIVision/UniversalFakeDetect. The official repo feeds the
CLIP *projected* image embedding into the probe: ``CLIP.encode_image`` returns
the 768-d projection of the 1024-d ViT-L/14 pooler output (``pooler @
visual_projection``, projection has no bias). This port therefore uses
``CLIPModel.get_image_features`` (768-d), not the raw 1024-d pooler output;
the probe state dict is ``{"weight": (1, 768), "bias": (1,)}``.

Preprocessing (``input_mode: full_frame``): the probe was trained
(``data/datasets.py``, ``options/base_options.py``: ``loadSize 256``,
``cropSize 224``) on ``TF.resize(img, 256)`` (shorter side, bilinear) ->
``RandomCrop(224)`` -> CLIP mean/std, and the official ``validate.py``
evaluates with ``CenterCrop(224)`` + CLIP mean/std on ~256 px benchmark
images (no resize).  Both amount to a 224 crop of a 256-short-side image, so
``preprocess`` resizes the shorter side to 256 (bilinear), centre-crops 224
and normalises with the CLIP statistics (also in ``detector.yaml``:
``resize_short: 256``, ``center_crop: true``).  Whole frames, no face
detection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

RESIZE_SHORT = 256
CROP = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class UnivFDDetector(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super().__init__()
        from transformers import CLIPModel

        self.backbone = CLIPModel.from_pretrained(model_name)
        self.head = nn.Linear(768, 1)

    @staticmethod
    def preprocess(pil):
        """resize shorter side 256 (bilinear) -> CenterCrop 224 -> CLIP mean/std."""
        pil = pil.convert("RGB")
        w, h = pil.size
        scale = RESIZE_SHORT / float(min(w, h))
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        left, top = int(round((new_w - CROP) / 2.0)), int(round((new_h - CROP) / 2.0))
        pil = pil.crop((left, top, left + CROP, top + CROP))
        arr = np.asarray(pil, dtype="float32").transpose(2, 0, 1) / 255.0
        arr = (arr - np.asarray(CLIP_MEAN, dtype="float32")[:, None, None]) / \
            np.asarray(CLIP_STD, dtype="float32")[:, None, None]
        return torch.from_numpy(arr).unsqueeze(0)

    def features(self, x):
        feats = self.backbone.get_image_features(pixel_values=x)
        # transformers >= 5 returns BaseModelOutputWithPooling whose
        # pooler_output already holds the 768-d projected embedding;
        # older versions return the projected tensor directly.
        if not torch.is_tensor(feats):
            embeds = getattr(feats, "image_embeds", None)
            feats = embeds if torch.is_tensor(embeds) else feats.pooler_output
        return feats

    def forward_detailed(self, x):
        feats = self.features(x)
        fake_logit = self.head(feats)
        # Official probe is a single BCE logit (fake = 1) with P(fake) = sigmoid(s).
        # Stacking [0, s] makes softmax[:, 1] == sigmoid(s) exactly, preserving
        # the original score semantics with index 1 = fake.
        logits = torch.cat([torch.zeros_like(fake_logit), fake_logit], dim=1)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = UnivFDDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location=device)
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.head.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
