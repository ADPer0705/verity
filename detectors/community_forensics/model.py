"""Community Forensics detector (Park & Owens, CVPR 2025, arXiv 2411.04125).

Ported from ``JeongsooP/Community-Forensics`` (MIT).  The released model is
``models.ViTClassifier``: a timm ``vit_small_patch16_384`` (ViT-S/16 at 384 px,
577 tokens, 21.8 M params) whose 1000-way ImageNet head is replaced by
``nn.Linear(384, 1)``; it is trained end-to-end on the Community Forensics
dataset (thousands of generators) with ``BCEWithLogitsLoss`` where the label
is 1 = fake (``--additional_data_label_format "real:0,fake:1"``), and
``eval.py`` reports ``sigmoid(logit)`` as the fake score.

Weights: ``OwensLab/commfor-model-384`` (HF hub, MIT, not gated).  The file
``model.safetensors`` is the ``PyTorchModelHubMixin`` export of
``ViTClassifier`` (all 152 keys prefixed ``vit.``, config.json =
``{model_size: small, input_size: 384, patch_size: 16}``); it holds the entire
network, so no timm pretrained download happens here.  The ``timm`` init in
the official class (``pretrained=True`` ImageNet-21k weights) is only the
*starting point* of training and is fully overwritten by the checkpoint, which
is why this port instantiates the architecture with ``pretrained=False`` and
loads the state dict strictly.

Preprocessing (``dataloader.py::get_transform`` mode "test" with
``determine_resize_crop_sizes``: input_size 384 -> resize_size 440,
crop_size 384): ``transforms.Resize(440)`` (shorter side, bilinear PIL) ->
``CenterCrop(384)`` -> ToTensor [0, 1] -> ``Normalize`` with **ImageNet**
mean/std ``[0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]`` (NOT CLIP stats; the
official repo normalises with ImageNet statistics, matching the timm
``augreg_in21k_ft_in1k`` backbone the model was initialised from).  The
engine's ``full_frame`` path reproduces this from ``detector.yaml``
(``resize_short: 440``, ``center_crop: true``, ``resolution: 384``), but
``preprocess`` below is also provided so the resize interpolation is the
bilinear one used upstream rather than the engine's default bicubic.

Explainability: the single logit is read from the CLS token, so Grad-CAM on
the *last* block's patch tokens would receive zero gradient.  ``feat`` is
therefore the token grid entering the last transformer block (B, 384, 24, 24):
the final block's attention routes those patches into the CLS token, so the
engine's Grad-CAM yields a meaningful 24x24 map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

RESIZE_SHORT = 440
CROP = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TIMM_ARCH = "vit_small_patch16_384"


class CommunityForensicsDetector(nn.Module):
    def __init__(self):
        super().__init__()
        import timm

        # Architecture only; the checkpoint carries every parameter.
        self.vit = timm.create_model(TIMM_ARCH, pretrained=False, num_classes=1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    # -- preprocessing (mirrors dataloader.get_transform(mode="test")) -------
    def preprocess(self, pil: Image.Image) -> torch.Tensor:
        pil = pil.convert("RGB")
        w, h = pil.size
        scale = RESIZE_SHORT / float(min(w, h))
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        # torchvision Resize(int) on a PIL image resizes the shorter side with
        # bilinear interpolation (PIL antialiased).
        pil = pil.resize((new_w, new_h), Image.BILINEAR)
        left, top = (new_w - CROP) // 2, (new_h - CROP) // 2
        pil = pil.crop((left, top, left + CROP, top + CROP))
        arr = np.asarray(pil, dtype="float32").transpose(2, 0, 1) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        return (x - self.mean.cpu()) / self.std.cpu()

    # -- forward --------------------------------------------------------------
    def forward_detailed(self, x):
        vit = self.vit
        t = vit.patch_embed(x)
        t = vit._pos_embed(t)
        t = vit.patch_drop(t)
        t = vit.norm_pre(t)
        for blk in vit.blocks[:-1]:
            t = blk(t)
        n_prefix = vit.num_prefix_tokens
        prefix, patches = t[:, :n_prefix], t[:, n_prefix:]
        side = int(round(patches.shape[1] ** 0.5))
        feat = patches.permute(0, 2, 1).reshape(patches.shape[0], patches.shape[2], side, side)
        # Re-assemble the sequence *from* feat so the logit is a function of
        # it (autograd needs feat on the path to the output for Grad-CAM).
        t = torch.cat([prefix, feat.flatten(2).permute(0, 2, 1)], dim=1)
        t = vit.blocks[-1](t)
        t = vit.norm(t)
        logit = vit.forward_head(t)  # (B, 1): CLS token -> Linear(384, 1)
        # Official score is sigmoid(logit) with 1 = fake. [0, z] keeps
        # softmax[:, 1] == sigmoid(z) exactly.
        logits = torch.cat([torch.zeros_like(logit), logit], dim=1)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feat}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = CommunityForensicsDetector()
    if weights_path is not None:
        from safetensors.torch import load_file

        state = load_file(weights_path)
        # Keys are the PyTorchModelHubMixin export of ViTClassifier ("vit.*");
        # mean/std are non-persistent buffers, so strict loading is exact.
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
