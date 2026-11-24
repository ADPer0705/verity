"""Shared modules of Forensics Adapter (Cui, Li, Luo, Zhou, Dong, CVPR 2025)
and VLAForge (Zhu, Miao, Zhang, Deng, Pang, CVPR 2026).

Re-implemented from ``model/adapters/adapter.py`` / ``model/ForgePerceiver.py``,
``model/attn.py`` and ``model/layer.py`` of the two official repositories
(OUC-VAS/ForensicsAdapter, CC BY-NC 4.0; mala-lab/VLAForge, MIT), inference
path only (the intra/inter contrastive losses are training-time).  Module and
parameter names match the released checkpoints.

* ``Adapter`` - a timm ViT-tiny trunk (first 8 blocks) over 16x16 patches of
  the 256x256 face with ``num_queries`` learnable query tokens; its
  ``Mask_Decoder`` turns the queries and patch map into a blending-boundary
  prediction (``xray_pred``, N x Q x 16 x 16) and a per-head attention bias
  (N x 16 x Q x 16 x 16).  Forensics Adapter additionally *fuses* CLIP layer
  features into the trunk through a ``Fusion`` module that upstream
  instantiates with fresh random weights on every forward (x += 0.01 * proj);
  ``fuse=True`` reproduces that with a fixed seed.  VLAForge defines but never
  calls ``fuse``.
* ``RecAttnClip`` - runs the frozen CLIP visual blocks again on
  ``[Q query copies of CLS, CLS, patch tokens]`` with the adapter's attention
  bias as an additive attention mask, injects a small residual from
  ``ClipIntraBlock`` after block ``intra_layer`` (6 for FA, 10 for VLAForge),
  and returns the projected, normalised query tokens (and, for VLAForge, the
  raw patch tokens).
* ``PostClipProcess`` - Conv1d over the queries + MLP -> 2 logits.
* ``MaskPostXrayProcess`` - the boundary-map decoder (Q x 16 x 16 -> 1 x 256 x 256).
"""
from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, affine_func=nn.Linear):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(affine_func(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Fusion(nn.Module):
    def __init__(self, clip_dim, adapter_dim, scale):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.scale = scale
        self.proj = nn.Sequential(LayerNorm2d(clip_dim), nn.Conv2d(clip_dim, adapter_dim, kernel_size=1))

    def forward(self, x, clip_x, spatial_shape):
        h, w = spatial_shape
        n, l, d = clip_x.shape
        if l == h * w:
            clip_x = clip_x.permute(0, 2, 1).reshape(n, d, h, w)
        else:
            clip_x = clip_x.permute(0, 2, 1).reshape(n, d, 14, 14)
            clip_x = F.interpolate(clip_x.contiguous(), size=(16, 16), mode="bilinear", align_corners=False)
        clip_x = self.proj(clip_x).reshape(n, self.adapter_dim, h * w).permute(0, 2, 1)
        return x + self.scale * clip_x


class VT_LN(nn.LayerNorm):
    def forward(self, x):
        orig = x.dtype
        return super().forward(x.type(torch.float32)).type(orig)


class PatchEmbed(nn.Module):
    """Upstream custom embed layer handed to timm (its ``norm`` is unused)."""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=192, norm_layer=None, bias=False, **kwargs):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.norm = VT_LN(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)


class Mask_Decoder(nn.Module):
    def __init__(self, in_dim, mlp_dim=512, out_dim=256, mlp_num_layers=3, head_num=16):
        super().__init__()
        self.head_num = head_num
        dense = partial(nn.Conv2d, kernel_size=1)
        self.query_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers)
        self.xray_mlp = MLP(in_dim, mlp_dim, out_dim, mlp_num_layers, affine_func=dense)
        self.attn_mlp = MLP(in_dim, mlp_dim, out_dim * head_num, mlp_num_layers, affine_func=dense)
        self.bias_scaling = nn.Linear(1, 1)

    def forward(self, query, x):
        query = self.query_mlp(query)
        xray = self.xray_mlp(x)
        attn = self.attn_mlp(x)
        xray_pred = torch.einsum("NQD,NDhw->NQhw", query, xray)
        n, d, h, w = xray.shape
        attn = attn.reshape(n, self.head_num, d, h, w)
        attn_bias = torch.einsum("NQD,NHDhw->NHQhw", query, attn)
        attn_bias = self.bias_scaling(attn_bias[..., None]).squeeze(-1)
        return xray_pred, attn_bias


class Adapter(nn.Module):
    """Forensics Adapter ``Adapter`` / VLAForge ``ForgePerceiver``."""

    def __init__(self, vit_name="vit_tiny_patch16_224", num_queries=128, fusion_map=None, mlp_dim=256,
                 mlp_out_dim=128, head_num=16, fuse=False, fuse_scale=0.01, fuse_seed=0xFA):
        super().__init__()
        from timm import create_model

        self.vit_model = create_model(vit_name, pretrained=False, fc_norm=False, num_classes=0,
                                      embed_layer=PatchEmbed)
        self.vit_model.pos_embed = nn.Parameter(self.vit_model.pos_embed[:, 1:, ...])
        del self.vit_model.cls_token
        self.vit_model.cls_token = None
        del self.vit_model.norm
        self.vit_model.norm = nn.Identity()
        self.num_queries = num_queries
        self.num_features = self.vit_model.num_features
        self.query_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))
        self.query_pos_embed = nn.Parameter(torch.zeros(1, num_queries, self.num_features))
        self.fusion_map = dict(fusion_map or {0: 0, 1: 1, 2: 8, 3: 15})
        self.mask_decoder = Mask_Decoder(self.num_features, mlp_dim, mlp_out_dim, 3, head_num)
        self.ln_pre = VT_LN(self.num_features)
        self.patch_conv = nn.Conv2d(3, self.num_features, kernel_size=16, stride=16, bias=True)
        self.fuse = fuse
        self.fuse_scale = fuse_scale
        self.fuse_seed = fuse_seed
        self._fusions = {}

    def _fusion(self, block_idx, clip_dim, device):
        # Upstream builds ``Fusion(clip_dim, D)`` with fresh random weights on
        # every call; a fixed seed per block makes the perturbation reproducible.
        key = (block_idx, str(device))
        if key not in self._fusions:
            gen_state = torch.random.get_rng_state()
            torch.manual_seed(self.fuse_seed + block_idx)
            self._fusions[key] = Fusion(clip_dim, self.num_features, self.fuse_scale).to(device)
            torch.random.set_rng_state(gen_state)
        return self._fusions[key]

    def forward(self, image, clip_features):
        x = self.patch_conv(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        pos = self.vit_model.pos_embed.permute(0, 2, 1)
        pos = F.interpolate(pos.reshape(pos.shape[0], pos.shape[1], 14, 14), size=(16, 16), mode="bilinear",
                            align_corners=False).reshape(pos.shape[0], pos.shape[1], 256).permute(0, 2, 1)
        pos = torch.cat([self.query_pos_embed.expand(pos.shape[0], -1, -1), pos], dim=1)
        v_l = x.shape[1]
        h, w = 16, 16
        x = torch.cat([self.query_embed.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        outs = []
        out_layers = [8]
        for i, block in enumerate(self.vit_model.blocks, start=1):
            x = block(x)
            if self.fuse and i in self.fusion_map:
                clip_layer = self.fusion_map[i]
                fusion = self._fusion(i, clip_features[clip_layer].shape[2], x.device)
                L = h * w
                x = torch.cat([x[:, :-L], fusion(x[:, -L:], clip_features[clip_layer], (h, w))], dim=1)
            if i in out_layers:
                n, _, d = x.shape
                outs.append({"query": x[:, :-v_l], "x": x[:, -v_l:].permute(0, 2, 1).reshape(n, d, h, w)})
            x = x + pos
            if i == max(out_layers):
                break
        xray_preds, attn_biases = [], []
        for feat in outs:
            xray_pred, attn_bias = self.mask_decoder(feat["query"], feat["x"])
            xray_preds.append(xray_pred)
            attn_biases.append(attn_bias)
        return attn_biases, xray_preds


class ClipIntraBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.conv_first = nn.Conv1d(num_features, 192, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv_second = nn.Conv1d(192, num_features, kernel_size=1)

    def forward(self, x, clip_l):
        intra = x.permute(1, 2, 0)[:, :, -clip_l:].float()   # LND -> NDL
        intra = self.relu(self.conv_first(intra))
        intra = self.conv_second(intra)
        return intra.permute(2, 0, 1)                        # NDL -> LND


class RecAttnClip(nn.Module):
    def __init__(self, vit, num_queries, intra_layer=6, return_patches=False):
        super().__init__()
        self.vit = vit
        self.resblocks = self.vit.transformer.resblocks
        self.first_layer = 0
        self.clss_nums = num_queries
        self.ln_post = self.vit.ln_post
        self.proj = self.vit.proj
        self.num_features = self.vit.width
        self.intra_scale = nn.Parameter(torch.zeros(1))
        self.intra_layer = intra_layer
        self.return_patches = return_patches
        self.clip_intra_blocks = nn.ModuleList([ClipIntraBlock(self.num_features)])

    def build_attn_mask(self, attn_bias):
        num_heads = self.resblocks[0].attn.num_heads
        n, head, q, h, w = attn_bias.shape
        assert head == num_heads, f"num_head={head} is not supported. Modify to {num_heads}"
        attn_bias = attn_bias.reshape(n * head, q, -1)
        l = attn_bias.shape[-1]
        attn_mask = attn_bias.new_zeros(q + 1 + l, q + 1 + l)
        attn_mask[:, :q] = -100
        attn_mask[torch.arange(q), torch.arange(q)] = 0
        attn_mask[:q, q] = -100
        attn_mask = attn_mask[None, ...].expand(n * head, -1, -1).clone()
        attn_mask[:, :q, -l:] = attn_bias
        return attn_mask

    def forward(self, clip_features, attn_bias, normalize=True):
        cls_token = clip_features[f"layer_{self.first_layer}_cls"].unsqueeze(1).permute(1, 0, 2).clone()
        vision_tokens = clip_features[self.first_layer].permute(1, 0, 2).clone()
        clss_token = cls_token.repeat(self.clss_nums, 1, 1)
        x = torch.cat([clss_token, cls_token, vision_tokens], dim=0)
        clip_l = vision_tokens.shape[0]
        mask = self.build_attn_mask(attn_bias)
        for i, block in enumerate(self.resblocks.children()):
            x = block(x, mask)
            if i == self.intra_layer:
                intra = self.clip_intra_blocks[0](x, clip_l)
                x = torch.cat([x[:-clip_l], intra * 0.05 + x[-clip_l:]], dim=0)
        x = x.permute(1, 0, 2)
        clss = self.ln_post(x[:, :self.clss_nums, :])
        if self.proj is not None:
            clss = clss @ self.proj
        if normalize:
            clss = F.normalize(clss, dim=-1)
        if self.return_patches:
            return x[:, -clip_l:, :], clss
        return clss


class MaskPostXrayProcess(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c // 2), nn.ReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c // 4), nn.ReLU(),
            nn.Conv2d(in_c // 4, 1, kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(1, 1, kernel_size=16, stride=16),
        )

    def forward(self, x, if_boundaries=None):
        n, q = x.shape[0], x.shape[1]
        x = x.reshape(n, q, -1).permute(0, 2, 1)
        if if_boundaries is not None:
            x = x * if_boundaries.unsqueeze(-1)
        x = x.permute(0, 2, 1).reshape(n, q, 16, 16)
        return self.process(x)


class PostClipProcess(nn.Module):
    def __init__(self, num_queries, embed_dim):
        super().__init__()
        self.first_process = nn.Sequential(
            nn.Conv1d(num_queries, num_queries // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_queries // 2), nn.ReLU(),
            nn.Conv1d(num_queries // 2, num_queries // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_queries // 4), nn.ReLU(),
            nn.Conv1d(num_queries // 4, 1, kernel_size=3, stride=1, padding=1),
        )
        self.second_process = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4), nn.ReLU(),
            nn.Linear(embed_dim // 4, 2),
        )

    def forward(self, x):
        x = self.first_process(x).squeeze(1)      # NQD -> N1D -> ND
        return self.second_process(x)
