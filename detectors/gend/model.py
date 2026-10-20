"""GenD detector (WACV 2026, arXiv 2508.06248), ported from the official
GenD repo using the HuggingFace transformers implementation.

Model: ``yermandy/GenD_CLIP_L_14`` — a CLIP ViT-L/14 vision tower with only
LayerNorm parameters fine-tuned, followed by an L2-normalized linear probe
(config head "LinearNorm"). The transformers model class is vendored in
``modeling_gend.py`` (verbatim copy of the official ``src/hf/modeling_gend.py``)
and is imported lazily so this module loads with only the stdlib.

Preprocessing: the official pipeline feeds PIL images through
``CLIPProcessor`` (resize shortest edge to 224, center crop 224, normalize
with CLIP mean/std). The engine reproduces it on tensors: face crops are
resized to 224x224 and normalized with the same CLIP mean/std from
detector.yaml, so the engine's input is already what ``feature_extractor``
expects and no un-normalization round-trip is needed (resize+center-crop
is an identity on the square 224x224 crops).

Grad-CAM: the head consumes the pooled CLS token (not spatial), so ``feat``
returns the ViT patch-token features reshaped to (B, 1024, 16, 16) instead
of the official ``pooler_output``; gradients reach them through the
self-attention path, yielding valid Grad-CAM maps.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

HUB_MODEL_ID = "yermandy/GenD_CLIP_L_14"


class GenDDetector:
    """Wraps the transformers GenD model behind the Verity detector contract."""

    def __init__(self, model):
        self.model = model

    def forward_detailed(self, x):
        import torch

        model = self.model
        hidden = model.feature_extractor.vision_model(x)
        pooled = hidden.pooler_output
        logits = model.model(pooled)
        patches = hidden.last_hidden_state[:, 1:]
        side = round(patches.shape[1] ** 0.5)
        feat = patches.permute(0, 2, 1).reshape(
            patches.shape[0], patches.shape[2], side, side
        )
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": feat,
        }


def build_model(weights_path, device):
    import torch

    from modeling_gend import GenD

    if weights_path is not None and Path(weights_path).exists():
        if (Path(weights_path) / "config.json").is_file():
            model = GenD.from_pretrained(str(weights_path))
        else:
            model = GenD.from_pretrained(HUB_MODEL_ID)
            state = torch.load(weights_path, map_location="cpu")
            if all(k.startswith("module.") for k in state.keys()):
                state = {k[7:]: v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
    else:
        model = GenD.from_pretrained(HUB_MODEL_ID)
    model.to(device)
    model.eval()
    return GenDDetector(model)
