"""M2F2-Det stage-1 detector (Guo, Song, Zhang, Liu, Liu, CVPR 2025 oral,
"Rethinking Vision-Language Model in Face Forensics: Multi-Modal
Interpretable Forged Face Detector", arXiv 2503.20188).

Ported from ``sequence/models/M2F2_Det/models/model.py`` (``M2F2Det``) +
``vision_encoder.py`` of github.com/CHELSEA234/M2F2_Det (MIT), i.e. the
*binary detector* of stage 1 (the LLaVA-based explanation stages 2/3 need the
7B ``llava-v1.5-7b-M2F2-Det`` model and are not part of this port).

Architecture:

* **CLIP vision branch** - frozen ``openai/clip-vit-large-patch14-336``
  vision tower (transformers ``CLIPVisionModel``; the LLaVA-1.5 tower the
  authors ship is byte-identical to it): penultimate hidden state (577 x
  1024); the CLS token through ``vision_proj`` (1024 -> 1792, LayerNorm)
  scaled by the learned ``clip_vision_alpha``.
* **Deepfake branch** - torchvision EfficientNet-B4 ``features`` (1792 x 11
  x 11 at 336 px), average pooled, ``deepfake_proj`` (1792 -> 1792, LayerNorm).
* **Bridge adapter** - intermediate EfficientNet maps (stages 4/5/6: 112,
  160, 272 channels) and CLIP hidden states 6/10/14 (without their last
  token, as upstream) are projected to 64-d tokens, concatenated
  progressively (441 + 576, +441 + 576, +121 + 576 = 2731 tokens) and passed
  through three tiny Transformer encoder blocks (4-head attention, FFN 32);
  ``bridge_adapter_proj`` squeezes the 2731 x 64 output to 128-d, scaled by
  ``clip_text_alpha``.
* ``output = Linear(3712, 2)`` on ``[vision, bridge, deepfake]``; upstream
  takes ``softmax[:, 0]`` as the fake probability (label 0 = fake in its
  split files), so the columns are swapped here (index 1 = fake).

Upstream quirk that limits this port: the three bridge-adapter blocks are
kept in a plain Python list, so they are **not part of the released
``model_state_dict``** (and used flash-attention's ``MHA`` with attention
taken over the batch axis, which is the identity for batch size 1).  Their
weights are re-initialised deterministically (seed 100, upstream's
``_seed_id``) and the detector carries a ``caveat`` in every report: 128 of
the 3,712 fused features come from an untrained module.  The CLIP text
encoder with the learned prompt tokens is also in the checkpoint but its
output is not used by the detection head, so it is not instantiated.

Preprocessing: the engine's 336x336 face crop with CLIP mean/std equals
``CLIPImageProcessor`` on the square crop, which upstream feeds to *both*
branches.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MODEL = "openai/clip-vit-large-patch14-336"
HIDDEN = 1792
CLIP_HIDDEN = 1024
BRIDGE_DIM = 128
BRIDGE_SEED = 100


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MHA(nn.Module):
    """flash_attn.modules.mha.MHA parameter layout (Wqkv, out_proj), eager math."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):                       # (B, S, D), attention along S
        b, s, d = x.shape
        qkv = self.Wqkv(x).view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B, H, S, hd)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.head_dim ** 0.5, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, d)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MHA(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(ff_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class BridgeAdapterProj(nn.Module):
    def __init__(self, hidden_size, bridge_adapter_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.bridge_adapter_proj = nn.Sequential(View(-1, hidden_size), nn.Linear(hidden_size, 1), nn.LayerNorm(1))
        self.bridge_adapter_reduction = nn.Sequential(nn.Linear(2731, bridge_adapter_dim),
                                                      nn.LayerNorm(bridge_adapter_dim))

    def forward(self, x, batch_size):
        x = self.bridge_adapter_proj(x).view(batch_size, -1)
        return self.bridge_adapter_reduction(x)


class M2F2Det(nn.Module):
    def __init__(self, model_name=CLIP_MODEL):
        super().__init__()
        from torchvision.models import efficientnet
        from transformers import CLIPVisionModel

        self.clip_vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        for p in self.clip_vision_encoder.parameters():
            p.requires_grad = False
        self.deepfake_encoder = efficientnet.efficientnet_b4(weights=None).features
        self.deepfake_proj = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN))
        self.vision_proj = nn.Sequential(nn.Linear(CLIP_HIDDEN, HIDDEN), nn.LayerNorm(HIDDEN))
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.clip_text_alpha = nn.Parameter(torch.tensor(4.0))
        self.linear_1 = nn.Linear(112, CLIP_HIDDEN // 16)
        self.linear_2 = nn.Linear(160, CLIP_HIDDEN // 16)
        self.linear_3 = nn.Linear(272, CLIP_HIDDEN // 16)
        self.clip_reduction = nn.Linear(CLIP_HIDDEN, CLIP_HIDDEN // 16)
        self.bridge_adapter_proj = BridgeAdapterProj(CLIP_HIDDEN // 16, BRIDGE_DIM)
        self.output = nn.Linear(2 * HIDDEN + BRIDGE_DIM, 2)
        # Not in the released checkpoint (see module docstring): deterministic init.
        state = torch.random.get_rng_state()
        torch.manual_seed(BRIDGE_SEED)
        self.bridge_adapter = nn.ModuleList([TransformerEncoderBlock(CLIP_HIDDEN // 16, 4, 32) for _ in range(3)])
        torch.random.set_rng_state(state)
        self._outputs = {}
        for i, idx in enumerate((4, 5, 6)):
            self.deepfake_encoder[idx].register_forward_hook(self._hook(f"b_{i + 1}"))

    def _hook(self, name):
        def hook(_m, _i, out):
            self._outputs[name] = out
        return hook

    def forward_detailed(self, x):
        b = x.shape[0]
        hs = self.clip_vision_encoder(pixel_values=x, output_hidden_states=True).hidden_states
        clip_0, clip_1, clip_2, clip_vision_features = hs[6][:, :-1, :], hs[10][:, :-1, :], hs[14][:, :-1, :], hs[-2]
        clip_vision_features = self.vision_proj(clip_vision_features)
        clip_vision_cls = self.clip_vision_alpha * clip_vision_features[:, 0, :]

        deepfake_features = self.deepfake_encoder(x)
        feats = [self._outputs["b_1"], self._outputs["b_2"], self._outputs["b_3"]]
        linears = [self.linear_1, self.linear_2, self.linear_3]
        bridge_out = None
        for i, (df, cf) in enumerate(zip(feats, (clip_0, clip_1, clip_2))):
            cf = self.clip_reduction(cf)
            spatial = df.shape[-2] * df.shape[-1]
            df = linears[i](df.reshape(b, df.shape[1], spatial).permute(0, 2, 1))
            combined = torch.cat((df, cf), dim=1) if bridge_out is None else torch.cat((bridge_out, df, cf), dim=1)
            # upstream hands (S, B, D) to a batch-first attention: attention runs over the batch axis
            bridge_out = self.bridge_adapter[i](combined.permute(1, 0, 2)).permute(1, 0, 2)
        clip_adapt_embed = self.clip_text_alpha * self.bridge_adapter_proj(bridge_out.permute(1, 0, 2), b).view(b, -1)

        deepfake_features = self.deepfake_proj(F.adaptive_avg_pool2d(deepfake_features, 1).view(b, -1))
        features = torch.cat([clip_vision_cls, clip_adapt_embed, deepfake_features], dim=-1)
        logits_upstream = self.output(features)          # [fake, real]
        logits = logits_upstream[:, [1, 0]]              # Verity: index 1 = fake
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": self._outputs["b_3"],
                "upstream_logits": logits_upstream}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = M2F2Det()
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        # The text tower + prompt tokens + text_proj are saved but never reach the head.
        state = {k: v for k, v in state.items() if not k.startswith(("clip_text_encoder.", "text_proj."))}
        own = model.state_dict()
        # Upstream wraps CLIPVisionModel as ``clip_vision_encoder.model``; transformers
        # versions differ in whether the ``vision_model.`` level appears in the keys.
        tower_prefix = "clip_vision_encoder.model."
        own_tower = [k for k in own if k.startswith("clip_vision_encoder.")]
        keep_level = any(k.startswith("clip_vision_encoder.vision_model.") for k in own_tower)
        remapped = {}
        for k, v in state.items():
            if k.startswith(tower_prefix):
                rest = k[len(tower_prefix):]
                if not keep_level and rest.startswith("vision_model."):
                    rest = rest[len("vision_model."):]
                k = "clip_vision_encoder." + rest
            remapped[k] = v
        state = remapped
        missing = sorted(k for k in set(own) - set(state) if not k.startswith("bridge_adapter."))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise RuntimeError(f"M2F2-Det checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
