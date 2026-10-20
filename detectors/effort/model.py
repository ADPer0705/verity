"""Effort detector (ICML 2025, Yan et al., arXiv 2411.15633).

Ported from the official ``DeepfakeBench/training/detectors/effort_detector.py``
in YZY-stack/Effort-AIGI-Detection. The detector is the ``vision_model`` of
``openai/clip-vit-large-patch14`` (ViT-L/14, 224 px, CLIP mean/std) followed by
a 2-way linear head on the 1024-d pooler output.

Effort's "orthogonal subspace decomposition": every ``nn.Linear`` inside each
``self_attn`` block (q/k/v/out projections, 24 layers x 4 = 96 linears) is
replaced by ``SVDResidualLinear``. The pretrained weight ``W = U S V^T`` is split
into a frozen principal part (top ``r = 1023`` singular components,
``weight_main``) and a trainable residual part (the remaining ``n - r = 1``
component: ``U_residual``, ``S_residual``, ``V_residual``). The effective weight
is ``weight_main + U_residual @ diag(S_residual) @ V_residual``; only the
residual (plus the head) is learned, with orthogonality / singular-value
regularisers at training time (not needed at inference).

The released checkpoints (Google Drive, CC BY-NC 4.0) carry the *whole* vision
tower (frozen CLIP params, ``weight_main`` and the learned residuals; the
principal factors ``U_r``/``S_r``/``V_r`` are *not* in the file, they are only
used by training regularisers) plus the head, so with a checkpoint the HF backbone is only
used to build the architecture: every tensor is then overwritten by a strict
``load_state_dict``. Without a checkpoint the SVD split is computed from the
CLIP weights (lossless; the head stays random -> untrained).

Attention map: the mean over heads of the last layer's CLS-token attention
over the 16x16 patch grid is returned as ``attention_map`` (requires the eager
attention implementation so transformers returns the attention weights).
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MODEL = "openai/clip-vit-large-patch14"
SVD_RANK = 1024 - 1  # upstream: r = 1024 - 1 for ViT-L/14


class SVDResidualLinear(nn.Module):
    """Linear layer with a frozen top-``r`` SVD part and a trainable residual.

    Verbatim semantics of the upstream module; ``init_weight=None`` and
    ``decompose=False`` allocate correctly-shaped parameters without running
    the SVD (they are filled by ``load_state_dict``).
    """

    def __init__(self, in_features, out_features, r, bias=True, init_weight=None,
                 decompose=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.weight_main = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        n = min(out_features, in_features)
        r = min(r, n)
        self.r = r
        n_res = n - r
        if n_res > 0:
            self.S_residual = nn.Parameter(torch.empty(n_res))
            self.U_residual = nn.Parameter(torch.empty(out_features, n_res))
            self.V_residual = nn.Parameter(torch.empty(n_res, in_features))
            # Upstream keeps the principal factors as frozen nn.Parameters, but
            # the released checkpoints do not contain them (they only feed the
            # training-time regularisers; inference uses weight_main). They are
            # therefore non-persistent buffers here, so strict loading matches
            # the checkpoint key set exactly (681 keys for ViT-L/14).
            self.register_buffer("S_r", torch.empty(r), persistent=False)
            self.register_buffer("U_r", torch.empty(out_features, r), persistent=False)
            self.register_buffer("V_r", torch.empty(r, in_features), persistent=False)
        else:
            self.S_residual = self.U_residual = self.V_residual = None
            self.S_r = self.U_r = self.V_r = None

        if init_weight is not None:
            if decompose:
                self.decompose_from(init_weight)
            else:
                self.weight_main.data.copy_(init_weight)
        elif decompose:
            nn.init.kaiming_uniform_(self.weight_main, a=math.sqrt(5))

    @torch.no_grad()
    def decompose_from(self, weight: torch.Tensor) -> None:
        """Upstream ``replace_with_svd_residual``: split ``weight`` by SVD."""
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        r = self.r
        self.weight_main.data.copy_((U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).to(weight.dtype))
        if self.S_residual is not None:
            self.U_residual.data.copy_(U[:, r:])
            self.S_residual.data.copy_(S[r:])
            self.V_residual.data.copy_(Vh[r:, :])
            self.U_r.copy_(U[:, :r])
            self.S_r.copy_(S[:r])
            self.V_r.copy_(Vh[:r, :])

    def compute_current_weight(self):
        if self.S_residual is not None:
            return self.weight_main + self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        return self.weight_main

    def forward(self, x):
        return F.linear(x, self.compute_current_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, r={self.r}"


def apply_svd_residual_to_self_attn(model: nn.Module, r: int, decompose: bool = True) -> nn.Module:
    """Replace every ``nn.Linear`` inside modules named ``*self_attn*`` (upstream logic)."""
    for name, module in model.named_children():
        if "self_attn" in name:
            for sub_name, sub_module in list(module.named_modules()):
                if isinstance(sub_module, nn.Linear):
                    parent = module
                    parts = sub_name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    new = SVDResidualLinear(sub_module.in_features, sub_module.out_features, r,
                                            bias=sub_module.bias is not None,
                                            init_weight=sub_module.weight.data.clone(),
                                            decompose=decompose)
                    if sub_module.bias is not None:
                        new.bias.data.copy_(sub_module.bias.data)
                    setattr(parent, parts[-1], new)
        else:
            apply_svd_residual_to_self_attn(module, r, decompose)
    for pname, param in model.named_parameters():
        param.requires_grad = any(k in pname for k in ("S_residual", "U_residual", "V_residual"))
    return model


class EffortDetector(nn.Module):
    def __init__(self, num_classes=2, model_name=CLIP_MODEL, decompose=True):
        super().__init__()
        from transformers import CLIPModel

        clip_model = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
        self.backbone = apply_svd_residual_to_self_attn(clip_model.vision_model, r=SVD_RANK,
                                                        decompose=decompose)
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
    # Non-parameter tensor attributes the upstream module stored on the
    # instance (``weight_original_fnorm``/``weight_main_fnorm``) are training
    # regulariser constants; drop them if a checkpoint ever contains them.
    state = {k: v for k, v in state.items()
             if not k.endswith(("weight_original_fnorm", "weight_main_fnorm"))}
    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    shape_mismatch = [k for k in set(own) & set(state) if tuple(own[k].shape) != tuple(state[k].shape)]
    if missing or unexpected or shape_mismatch:
        print(f"[effort] checkpoint mismatch: missing={missing[:10]}{'...' if len(missing) > 10 else ''} "
              f"unexpected={unexpected[:10]}{'...' if len(unexpected) > 10 else ''} "
              f"shape={shape_mismatch[:10]}", file=sys.stderr)
        raise RuntimeError(f"Effort checkpoint {weights_path} does not match the model "
                           f"({len(missing)} missing, {len(unexpected)} unexpected, "
                           f"{len(shape_mismatch)} shape mismatches)")
    model.load_state_dict(state, strict=True)


def build_model(weights_path, device):
    import os

    have_ckpt = weights_path is not None and os.path.isfile(weights_path)
    if weights_path is not None and not have_ckpt:
        print(f"[effort] WARNING: checkpoint {weights_path} not found; running with the "
              f"pretrained CLIP tower and a RANDOM head (not evidence-grade). See weights/README.md.",
              file=sys.stderr)
    # With a checkpoint, skip the 96 SVDs: every tensor is loaded from the file.
    model = EffortDetector(num_classes=2, decompose=not have_ckpt)
    if have_ckpt:
        _load_checkpoint(model, weights_path)
    model.to(device)
    model.eval()
    return model
