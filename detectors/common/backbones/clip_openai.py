"""OpenAI-layout CLIP (ViT-L/14) with the feature-extraction and
attention-bias hooks added by Forensics Adapter / VLAForge.

This is a re-implementation of ``model/clip/model.py`` as modified in
github.com/OUC-VAS/ForensicsAdapter (CVPR 2025, CC BY-NC 4.0) and
github.com/mala-lab/VLAForge (CVPR 2026, MIT); the base code is OpenAI's
CLIP (MIT).  Parameter names follow the OpenAI checkpoint layout
(``visual.transformer.resblocks.N.attn.in_proj_weight`` ...) so the released
detector checkpoints, which embed the frozen CLIP under ``clip_model.*``, load
strictly.  Differences from stock CLIP that the detectors were trained with
and that are reproduced here on purpose:

* ``ResidualAttentionBlock.forward(x, attn_mask=None)`` takes the mask per
  call; the text ``Transformer`` passes **no** causal mask (the upstream
  modification dropped it).
* ``VisionTransformer.forward(x, extract=[...])`` returns the CLS / patch
  tokens after the listed layers (``layer_0_cls`` / ``0`` are the tokens
  after ``ln_pre``) and stops at the deepest requested layer.
* ``encode_text(text, id_emb=None)`` (VLAForge) replaces the 7th token
  embedding by an identity embedding before the positional embedding.

``hf_clip_to_openai`` converts a transformers ``openai/clip-vit-large-patch14``
state dict into this layout for checkpoints that do not carry the tower.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

VIT_L14 = dict(embed_dim=768, image_resolution=224, vision_layers=24, vision_width=1024,
               vision_patch_size=14, context_length=77, vocab_size=49408,
               transformer_width=768, transformer_heads=12, transformer_layers=12)


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        orig = x.dtype
        return super().forward(x.type(torch.float32)).type(orig)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x, attn_mask):
        attn_mask = attn_mask.to(dtype=x.dtype, device=x.device) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x, attn_mask=None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class VTransformer(nn.Module):
    """Vision transformer trunk with intermediate-feature extraction."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x, extract=None):
        features = {}
        if extract is not None:
            features["layer_0_cls"] = x[0]                   # (N, D)
            features[0] = x[1:].permute(1, 0, 2)              # (N, L, D)
        for idx, layer in enumerate(self.resblocks.children(), start=1):
            x = layer(x)
            if extract is not None and idx in extract:
                features[f"layer_{idx}_cls"] = x[0]
                features[idx] = x[1:].permute(1, 0, 2)
                if idx == max(extract):
                    return features
        return x


class Transformer(nn.Module):
    """Text transformer; the upstream modification applies no causal mask."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = VTransformer(width, layers, heads)
        self.width = width
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, extract=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                         dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                       # NLD -> LND
        if extract is not None:
            return self.transformer(x, extract)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers):
        super().__init__()
        self.context_length = context_length
        self.visual = VisionTransformer(image_resolution, vision_patch_size, vision_width, vision_layers,
                                        vision_width // 64, embed_dim)
        self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads)
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def extract_features(self, image, extract):
        return self.visual(image.type(self.dtype), list(extract))

    def encode_text(self, text, id_emb=None):
        x = self.token_embedding(text).type(self.dtype)
        if id_emb is not None:
            # VLAForge identity prior: the token at position 6 is replaced by the
            # (normalised) CLIP image embedding of the face being examined.
            id_emb = id_emb.type(self.dtype)
            x = torch.cat([x[:, :6, :], id_emb.unsqueeze(0), x[:, 7:, :]], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def build_clip(config=VIT_L14) -> CLIP:
    return CLIP(**config)


def hf_clip_to_openai(hf: dict) -> dict:
    """transformers CLIPModel state dict (openai/clip-vit-large-patch14) ->
    OpenAI layout used by this module."""
    out = {}
    v = "vision_model."
    out["visual.conv1.weight"] = hf[v + "embeddings.patch_embedding.weight"]
    out["visual.class_embedding"] = hf[v + "embeddings.class_embedding"]
    out["visual.positional_embedding"] = hf[v + "embeddings.position_embedding.weight"]
    out["visual.ln_pre.weight"] = hf[v + "pre_layrnorm.weight"]
    out["visual.ln_pre.bias"] = hf[v + "pre_layrnorm.bias"]
    out["visual.ln_post.weight"] = hf[v + "post_layernorm.weight"]
    out["visual.ln_post.bias"] = hf[v + "post_layernorm.bias"]
    out["visual.proj"] = hf["visual_projection.weight"].t().contiguous()
    t = "text_model."
    out["token_embedding.weight"] = hf[t + "embeddings.token_embedding.weight"]
    out["positional_embedding"] = hf[t + "embeddings.position_embedding.weight"]
    out["ln_final.weight"] = hf[t + "final_layer_norm.weight"]
    out["ln_final.bias"] = hf[t + "final_layer_norm.bias"]
    out["text_projection"] = hf["text_projection.weight"].t().contiguous()
    out["logit_scale"] = hf["logit_scale"]

    def blocks(prefix_hf, prefix_out):
        i = 0
        while f"{prefix_hf}{i}.layer_norm1.weight" in hf:
            p = f"{prefix_hf}{i}."
            o = f"{prefix_out}{i}."
            out[o + "attn.in_proj_weight"] = torch.cat([hf[p + "self_attn.q_proj.weight"],
                                                       hf[p + "self_attn.k_proj.weight"],
                                                       hf[p + "self_attn.v_proj.weight"]], 0)
            out[o + "attn.in_proj_bias"] = torch.cat([hf[p + "self_attn.q_proj.bias"],
                                                     hf[p + "self_attn.k_proj.bias"],
                                                     hf[p + "self_attn.v_proj.bias"]], 0)
            out[o + "attn.out_proj.weight"] = hf[p + "self_attn.out_proj.weight"]
            out[o + "attn.out_proj.bias"] = hf[p + "self_attn.out_proj.bias"]
            out[o + "ln_1.weight"] = hf[p + "layer_norm1.weight"]
            out[o + "ln_1.bias"] = hf[p + "layer_norm1.bias"]
            out[o + "ln_2.weight"] = hf[p + "layer_norm2.weight"]
            out[o + "ln_2.bias"] = hf[p + "layer_norm2.bias"]
            out[o + "mlp.c_fc.weight"] = hf[p + "mlp.fc1.weight"]
            out[o + "mlp.c_fc.bias"] = hf[p + "mlp.fc1.bias"]
            out[o + "mlp.c_proj.weight"] = hf[p + "mlp.fc2.weight"]
            out[o + "mlp.c_proj.bias"] = hf[p + "mlp.fc2.bias"]
            i += 1

    blocks(v + "encoder.layers.", "visual.transformer.resblocks.")
    blocks(t + "encoder.layers.", "transformer.resblocks.")
    return out


def tokenize_openai(tokenizer, texts, context_length=77):
    """open_clip-style token ids: <sot> tokens <eot>, zero-padded to 77 (the
    HF tokenizer pads with <eot>, which matters here because the text tower
    runs without a causal mask)."""
    ids = []
    for t in texts:
        enc = tokenizer(t, add_special_tokens=True, truncation=True, max_length=context_length)["input_ids"]
        enc = enc + [0] * (context_length - len(enc))
        ids.append(enc)
    return torch.tensor(ids, dtype=torch.long)
