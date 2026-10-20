"""FTCN detector (DeepfakeBench port): I3D R50 backbone with temporal-only
convs and a TimeTransformer head.

Ported from DeepfakeBench ``training/detectors/ftcn_detector.py`` and
``networks/time_transformer.py``; the checkpoint (``I3D_8x8_R50.pth``) is the
Kinetics-pretrained backbone. Only the inference path is reproduced (the
``temporal_only_conv`` rewire and the stop_point=5 head swap are kept; the
original loss/optimizer are dropped). The raw network output is a single
sigmoid value, converted to 2-class logits.
"""
from __future__ import annotations

import sys
from inspect import signature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from torch import einsum  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import rearrange, repeat  # noqa: E402

from common.slowfast.config.defaults import get_cfg  # noqa: E402
from common.slowfast.models.video_model_builder import ResNet as ResNetOri  # noqa: E402

config_text = """
TRAIN:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 16
  SAMPLING_RATE: 8
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
SOLVER:
  BASE_LR: 0.1
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TimeTransformer(nn.Module):
    def __init__(self, num_patches, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.num_patches = num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask=None)

        x = x.mean(dim=1) if self.pool == 'cls' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class TransformerHead(nn.Module):
    def __init__(self, spatial_size=7, time_size=16, in_channels=2048):
        super().__init__()
        patch_type = "time"
        assert patch_type == "time"
        self.pool = nn.AvgPool3d((1, spatial_size, spatial_size))
        self.num_patches = time_size

        self.dim = -1
        if self.dim == -1:
            self.dim = in_channels

        self.in_channels = in_channels

        if self.dim != self.in_channels:
            self.fc = nn.Linear(self.in_channels, self.dim)

        default_params = dict(
            dim=self.dim, depth=1, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
        )
        self.time_T = TimeTransformer(
            num_patches=self.num_patches, num_classes=1, **default_params
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feat = self.pool(x[0])
        x = feat.reshape(-1, self.in_channels, self.num_patches)
        x = x.permute(0, 2, 1)
        if self.dim != self.in_channels:
            x = self.fc(x.reshape(-1, self.in_channels))
            x = x.reshape(-1, self.num_patches, self.dim)
        x = self.time_T(x)
        x = self.sigmoid(x)
        return x, feat


parameters = [parameter for parameter in signature(nn.Conv3d).parameters]

spatial_count = 0
keep_stride_count = 0
no_time_pool = True


def temporal_only_conv(module, name, removed, stride_removed=0):
    """Recursively rewire 3D convs to temporal-only (1x1x1 spatial) kernels."""
    for attr_str in dir(module):
        sub_module = getattr(module, attr_str)
        if type(sub_module) == nn.Conv3d:
            target_spatial_size = 1
            predefine_padding = {1: 0, 3: 1, 5: 2, 7: 3}
            kernel_size = list(sub_module.kernel_size)
            assert kernel_size[1] == kernel_size[2]
            stride = sub_module.stride
            extra = None
            if stride[1] == stride[2] == 2:
                stride_removed += 1
                if stride_removed > keep_stride_count:
                    stride = [1, 1, 1]
                    extra = nn.MaxPool3d((1, 2, 2))

            if kernel_size[1] == 1 and extra is None:
                continue
            padding = list(sub_module.padding)

            kernel_size[1] = kernel_size[2] = target_spatial_size
            padding[1] = padding[2] = predefine_padding[target_spatial_size]

            param_dict = {key: getattr(sub_module, key) for key in parameters if key not in ['device', 'dtype']}

            param_dict.update(kernel_size=kernel_size, padding=padding, stride=stride)

            conv = nn.Conv3d(**param_dict)

            new_module = conv

            removed += 1
            if removed > spatial_count:
                setattr(module, attr_str, new_module)
                if extra is not None:
                    if attr_str == "conv":
                        bn_str = "bn"
                    else:
                        bn_str = f"{attr_str}_bn"
                    if hasattr(module, bn_str):
                        bn_module = getattr(module, bn_str)
                        assert isinstance(bn_module, nn.BatchNorm3d)
                        new_bn_module = nn.Sequential(bn_module, extra)
                        setattr(module, bn_str, new_bn_module)
        elif type(sub_module) == nn.Dropout:
            setattr(module, attr_str, nn.Dropout(p=0.5))
        if no_time_pool:
            if type(sub_module) == nn.MaxPool3d:
                kernel_size = list(sub_module.kernel_size)
                if kernel_size[0] == 2:
                    kernel_size[0] = 1
                    setattr(module, attr_str, nn.MaxPool3d(kernel_size))
            elif type(sub_module) == nn.AvgPool3d:
                kernel_size = list(sub_module.kernel_size)
                kernel_size[0] = 2 * kernel_size[0]
                setattr(module, attr_str, nn.AvgPool3d(kernel_size))

    old_name = name
    for name, immediate_child_module in module.named_children():
        removed, stride_removed = temporal_only_conv(
            immediate_child_module, old_name + "." + name, removed, stride_removed
        )
    return removed, stride_removed


def _prob_to_logits(p):
    p_flat = p.reshape(-1).clamp(1e-7, 1 - 1e-7)
    logits = torch.zeros(p_flat.shape[0], 2, device=p.device, dtype=p.dtype)
    logits[:, 1] = torch.log(p_flat / (1.0 - p_flat))
    return logits


class FTCNDetector(nn.Module):
    def __init__(self, clip_size=16):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = clip_size
        self.resnet = ResNetOri(cfg)

    def forward_detailed(self, x):
        inputs = [x]
        pred, feat = self.resnet(inputs)
        logits = _prob_to_logits(pred)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feat}


def build_model(weights_path, device, clip_size=16):
    model = FTCNDetector(clip_size=clip_size)
    if weights_path is not None:
        pretrained_weights = torch.load(weights_path, map_location="cpu", encoding="latin1")
        modified_weights = {k.replace("resnet.", ""): v for k, v in pretrained_weights.items()}
        modified_weights["head.projection.weight"] = modified_weights["head.projection.weight"][:1, :]
        modified_weights["head.projection.bias"] = modified_weights["head.projection.bias"][:1]
        model.resnet.load_state_dict(modified_weights, strict=True)

    temporal_only_conv(model.resnet, "model", 0)

    stop_point = 5
    for i in [5, 4, 3]:
        if stop_point <= i:
            setattr(model.resnet, f"s{i}", nn.Identity())
            if stop_point == 3:
                setattr(model.resnet, "pathway0_pool", nn.Identity())

    params = {
        6: dict(spatial_size=7, time_size=clip_size, in_channels=2048),
        5: dict(spatial_size=14, time_size=clip_size, in_channels=1024),
        4: dict(spatial_size=28, time_size=clip_size, in_channels=512),
        3: dict(spatial_size=56, time_size=clip_size * 2, in_channels=256),
    }[stop_point]

    model.resnet.head = TransformerHead(**params)

    model.to(device)
    model.eval()
    return model
