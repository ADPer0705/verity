"""AntiDeepfake MMS-300M (NII Yamagishi Lab, ASRU 2025; Ge, Wang, Liu,
Yamagishi, arXiv 2506.21090): post-trained wav2vec 2.0 / MMS-300M speech
foundation model + mean-pooling + one FC layer for deepfake-speech detection.

Upstream (github.com/nii-yamagishilab/AntiDeepfake, models/W2V.py) builds the
SSL tower with fairseq ``Wav2Vec2Model`` and stores the detector as a flat
safetensors state dict (``m_ssl.model.*`` = fairseq wav2vec2 keys,
``proj_fc.*`` = the 1024->2 classifier).  This port rebuilds the same
architecture with ``transformers.Wav2Vec2Model`` (do_stable_layer_norm,
feat_extract_norm="layer", conv_bias) and remaps the fairseq keys 1:1 (the
same mapping transformers itself uses to convert fairseq wav2vec2 checkpoints),
so no fairseq install is needed.  Pre-training-only tensors (quantizer,
project_q, final_proj) are dropped.

Input: mono 16 kHz waveform, arbitrary length.  Upstream normalises each
utterance with ``F.layer_norm(wav, wav.shape)`` (zero mean / unit variance,
dataio/dataio.py and the HF model card), reproduced here before the tower.

Output: upstream logits are ordered [fake, real] (dataio labels: real=1,
fake=0; the card prints ``real prob = prob[1], fake prob = prob[0]``).  Verity
wants softmax index 1 = fake, so ``forward_detailed`` returns the columns
swapped: ``logits[:, [1, 0]]``.

Weights are fetched from the Hugging Face hub (``nii-yamagishilab/
mms-300m-anti-deepfake``, CC BY-NC-SA 4.0) via ``huggingface_hub`` at first
run; set ``VERITY_ANTIDEEPFAKE_REPO`` to switch to a sibling checkpoint with
the same layout (e.g. ``nii-yamagishilab/xls-r-1b-anti-deepfake``: 48 layers,
1280-d; the architecture is read from the repo's config.json so the code is
unchanged).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HUB_MODEL_ID = os.environ.get("VERITY_ANTIDEEPFAKE_REPO", "nii-yamagishilab/mms-300m-anti-deepfake")


def _map_fairseq_key(key: str) -> str | None:
    """fairseq wav2vec2 state-dict key -> transformers Wav2Vec2Model key.
    Returns None for pre-training-only tensors that the encoder does not use."""
    if key.startswith(("quantizer.", "project_q.", "final_proj.")):
        return None
    if key == "mask_emb":
        return "masked_spec_embed"
    if key.startswith("feature_extractor.conv_layers."):
        # conv_layers.{i}.0.* = conv, conv_layers.{i}.2.1.* = Fp32LayerNorm
        parts = key.split(".")
        i, sub, rest = parts[2], parts[3], parts[-1]
        if sub == "0":
            return f"feature_extractor.conv_layers.{i}.conv.{rest}"
        if sub == "2":
            return f"feature_extractor.conv_layers.{i}.layer_norm.{rest}"
        raise KeyError(key)
    if key.startswith("layer_norm."):
        return "feature_projection." + key
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    if key.startswith("encoder.pos_conv.0."):
        rest = key.split(".")[-1]
        rest = {"weight_g": "parametrizations.weight.original0",
                "weight_v": "parametrizations.weight.original1"}.get(rest, rest)
        return f"encoder.pos_conv_embed.conv.{rest}"
    if key.startswith("encoder.layers."):
        key = key.replace(".self_attn.", ".attention.")
        key = key.replace(".self_attn_layer_norm.", ".layer_norm.")
        key = key.replace(".fc1.", ".feed_forward.intermediate_dense.")
        key = key.replace(".fc2.", ".feed_forward.output_dense.")
        return key
    if key.startswith("encoder.layer_norm."):
        return key
    raise KeyError(key)


class AntiDeepfakeDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.ssl = Wav2Vec2Model(config)
        self.proj_fc = nn.Linear(config.hidden_size, 2)

    def forward_detailed(self, x):
        # x: (B, T) float32 waveform in [-1, 1] at 16 kHz
        if x.ndim == 3:
            x = x[:, :, 0]
        x = x.float()
        # Per-utterance layer_norm over all samples (upstream dataio.py / card).
        x = F.layer_norm(x, x.shape[1:])
        emb = self.ssl(x).last_hidden_state            # (B, frames, 1024)
        pooled = emb.mean(dim=1)                        # AdaptiveAvgPool1d(1)
        logits_upstream = self.proj_fc(pooled)          # [fake, real]
        logits = logits_upstream[:, [1, 0]]             # Verity: index 1 = fake
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "upstream_logits": logits_upstream, "feat": pooled}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def _load_state(model: AntiDeepfakeDetector, path: str) -> None:
    from safetensors.torch import load_file

    raw = load_file(path)
    ssl_state, head_state = {}, {}
    for k, v in raw.items():
        if k.startswith("m_ssl.model."):
            mk = _map_fairseq_key(k[len("m_ssl.model."):])
            if mk is not None:
                ssl_state[mk] = v
        elif k.startswith("proj_fc."):
            head_state[k[len("proj_fc."):]] = v
        else:
            raise KeyError(f"unexpected checkpoint key {k}")
    missing, unexpected = model.ssl.load_state_dict(ssl_state, strict=False)
    missing = [m for m in missing if m != "masked_spec_embed"]
    if missing or unexpected:
        raise RuntimeError(f"AntiDeepfake key mapping failed: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]}")
    model.proj_fc.load_state_dict(head_state, strict=True)


def build_model(weights_path, device):
    import json

    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2Config

    cfg_path = hf_hub_download(HUB_MODEL_ID, "config.json")
    if weights_path and os.path.exists(weights_path):
        st_path = weights_path
    else:
        st_path = hf_hub_download(HUB_MODEL_ID, "model.safetensors")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # Inference-only: no SpecAugment masking, no LayerDrop, no dropout.
    cfg.update({"apply_spec_augment": False, "layerdrop": 0.0, "mask_time_prob": 0.0,
                "hidden_dropout": 0.0, "attention_dropout": 0.0, "feat_proj_dropout": 0.0,
                "activation_dropout": 0.0, "final_dropout": 0.0})
    config = Wav2Vec2Config(**{k: v for k, v in cfg.items() if k != "architectures"})
    model = AntiDeepfakeDetector(config)
    _load_state(model, st_path)
    model.to(device)
    model.eval()
    return model
