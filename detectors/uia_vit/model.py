"""UIA-ViT detector (Zhuang et al. 2022): ViT-B/16 with an unsupervised
inconsistency-aware head that mixes the [CLS] token with patch features
localized by the cross-attention maps of the last blocks.

Ported from DeepfakeBench ``training/detectors/uia_vit_detector.py``.  Input is
a ``(B, 3, 224, 224)`` image.  Inference runs with ``step=1`` (progressive
schedule fully blended): the localization map is the mean CLS->patch attention
of blocks 7..11 (kept as the source's list-branch ``forward_features``,
including its off-by-one block indexing), and the head consumes
``[cls, weighted-mean patch]`` (768+768 -> 2).  The ``head`` is 1536->2 so the
ImageNet jx_vit weights cannot load the head (skipped by ``strict=False``);
``c_real/c_fake/c_cross`` are kept for checkpoint compatibility.  Training-only
code (patch consistency loss, MVG feature stats, LedoitWolf) is dropped.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # noqa: E402


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_qk = (q @ k.transpose(-2, -1)) * self.scale
        attn_s = attn_qk.softmax(dim=-1)
        attn = self.attn_drop(attn_s)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_qk


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_attn, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.c_real = nn.Parameter(torch.tensor(0.6))
        self.c_fake = nn.Parameter(torch.tensor(0.6))
        self.c_cross = nn.Parameter(torch.tensor(0.2))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.norm_middle = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, attn_blk, feat_blk=False):
        if not feat_blk:
            feat_blk = attn_blk - 1
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_list = []
        for i, blk in enumerate(self.blocks):
            if i == feat_blk:
                x_block = self.norm_middle(x)
            if i in attn_blk:
                attn_list.append(attn)
            x, attn = blk(x)
        x = self.norm(x)
        if (i + 1) in attn_blk:
            attn_list.append(attn)
        if i == feat_blk - 1:
            x_block = x
        attn_block = torch.cat(attn_list, dim=1)

        x_block = x_block[:, 1:].reshape(
            (x_block.size(0), int(x_block.size(1) ** 0.5), int(x_block.size(1) ** 0.5), x_block.size(2)))
        return x, x_block, attn_block

    def forward(self, x, step=1, attn_blk=[8, 9, 10, 11, 12], feat_blk=6,
                k=12, thr=0.7, is_progressive=1):
        x, feat_block, attn_block = self.forward_features(x, attn_blk, feat_blk)

        x_cls, x_patch = x[:, 0], x[:, 1:]
        B, PP, C = x_patch.shape
        localization_map = torch.sigmoid(torch.mean(attn_block[:, :, 0, 1:], dim=1))

        if is_progressive:
            if step < 1 / 8.:
                localization_map = (torch.ones(B, 1, PP) / PP).to(x_patch.device)
            else:
                w = torch.sigmoid(torch.tensor(-k * (step - thr))).to(x_patch.device)
                localization_map = (w * torch.ones(B, 1, PP).to(x_patch.device)
                                    + (1 - w) * localization_map.reshape(B, 1, PP)) / PP
        else:
            localization_map = localization_map.reshape(B, 1, PP) / PP
        x = torch.cat([x_cls, torch.bmm(localization_map, x_patch).squeeze(1)], -1)
        x = self.head(x)
        return x, feat_block, attn_block


class UIAViTDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = VisionTransformer(patch_size=16, embed_dim=768, depth=12,
                                       num_heads=12, mlp_ratio=4, qkv_bias=True,
                                       norm_layer=nn.LayerNorm, num_classes=num_classes)

    def forward_detailed(self, x):
        pred, feature_patch, attn_block = self.model(x, step=1)
        attention_map = torch.sigmoid(torch.mean(attn_block[:, :, 0, 1:], dim=1))
        attention_map = attention_map.reshape(-1, int(attention_map.size(1) ** 0.5),
                                              int(attention_map.size(1) ** 0.5))
        feat = feature_patch.permute(0, 3, 1, 2)
        return {"logits": pred, "probs": torch.softmax(pred, dim=1),
                "feat": feat, "attention_map": attention_map}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = UIAViTDetector(num_classes=2)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
