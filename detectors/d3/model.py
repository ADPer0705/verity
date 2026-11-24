"""D³ detector (Yang, Qian, Zhu, Russakovsky, Wu, CVPR 2025, "D³: Scaling Up
Deepfake Detection by Learning from Discrepancy", arXiv 2404.04584).

Ported from ``models/clip_models.py::CLIPModelShuffleAttentionPenultimateLayer``
and ``models/transformer_attention.py::TransformerAttention`` of
github.com/BigAandSmallq/D3 (no license file), checkpoint ``ckpt/classifier.pth``
committed in the repository.

Architecture: a frozen CLIP ViT-L/14 image tower (OpenAI weights; here the
HuggingFace ``openai/clip-vit-large-patch14`` vision model, whose
``pooler_output`` = post-LayerNorm CLS token is exactly the upstream
``ln_post`` hook output, 1024-d) is run twice per image: once on the image
with its 14x14 patches randomly shuffled (``F.unfold`` -> ``randperm`` ->
``F.fold``) and once on the original image.  The two 1024-d features are
stacked as a 2-token sequence and passed to a single-head attention block
(q/k/v Linear(1024, 1024), softmax over **dim=1** as upstream, output
flattened to 2048) with ``fc = Linear(2048, 1)``; P(fake) = sigmoid.
``shuffle_times = original_times = 1`` and ``patch_size = 14`` are the
training defaults (``options/base_options.py``) and are confirmed by the
head's ``fc.weight`` shape (1, 2048).  The shuffle uses a seeded generator so
the score of a frame is reproducible.

Preprocessing (``validate_for_robustness.py``): ``Resize((224, 224))`` (both
sides, no crop) + CLIP mean/std, whole image (``input_mode: full_frame``).
Trained on the UFD (ProGAN) + GenImage multi-generator composite.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MODEL = "openai/clip-vit-large-patch14"
PATCH = 14
SHUFFLE_TIMES = 1
ORIGINAL_TIMES = 1
SHUFFLE_SEED = 0xD3


class TransformerAttention(nn.Module):
    """Verbatim semantics of the upstream head (including softmax over dim=1)."""

    def __init__(self, input_dim: int, output_dim: int, last_dim: int = 1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * output_dim, last_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q, k, v = self.query(x), self.key(x), self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.tensor(k.size(-1), dtype=attention.dtype, device=attention.device))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        output = output.reshape(output.shape[0], -1)
        return self.fc(output)


class D3Detector(nn.Module):
    def __init__(self, model_name: str = CLIP_MODEL):
        super().__init__()
        from transformers import CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.attention_head = TransformerAttention(1024, SHUFFLE_TIMES + ORIGINAL_TIMES, last_dim=1)

    @staticmethod
    def shuffle_patches(x: torch.Tensor, patch_size: int, generator: torch.Generator) -> torch.Tensor:
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
        perm = torch.randperm(patches.size(-1), generator=generator).to(x.device)
        shuffled = patches[:, :, perm]
        return F.fold(shuffled, output_size=(h, w), kernel_size=patch_size, stride=patch_size)

    def penultimate(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return out.pooler_output  # post-LN CLS token == upstream ln_post hook, (B, 1024)

    def forward_detailed(self, x):
        gen = torch.Generator(device="cpu").manual_seed(SHUFFLE_SEED)
        feats = []
        with torch.no_grad():
            for _ in range(SHUFFLE_TIMES):
                feats.append(self.penultimate(self.shuffle_patches(x, PATCH, gen)))
            original = self.penultimate(x)
            for _ in range(ORIGINAL_TIMES):
                feats.append(original.clone())
        tokens = torch.stack(feats, dim=-2)              # (B, 2, 1024)
        fake_logit = self.attention_head(tokens)         # (B, 1), sigmoid = P(fake)
        logits = torch.cat([torch.zeros_like(fake_logit), fake_logit], dim=1)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": original}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = D3Detector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        state = {k[len("attention_head."):] if k.startswith("attention_head.") else k: v
                 for k, v in state.items()}
        model.attention_head.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
