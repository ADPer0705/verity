"""M2TR detector (Wang, Wu, Chen, Jiang, ICMR 2022, "M2TR: Multi-modal
Multi-scale Transformers for Deepfake Detection", arXiv 2104.09770).

Ported from ``M2TR/models/m2tr.py`` + ``modules/*`` + the vendored
``efficientnet.py`` of github.com/wdrink/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection
(MIT), configuration ``configs/m2tr.yaml``: EfficientNet-B4 backbone, texture
layer ``b2`` (32 channels at 80x80 for a 320 px input), ``DEPTH 4``,
``NUM_CLASSES 2``, no decoder.  Each of the 4 stages applies (1) a
multi-scale patch self-attention block over the texture map (patch sizes
80/40/20/10, one channel group per scale, 3x3 output conv), (2) a
frequency block (learned global filter in the rFFT domain of the 80x80 map,
``complex_weight`` 80x41x32) and (3) cross-modality attention fusing the
frequency branch into the RGB branch.  The fused texture map goes through
the remaining EfficientNet blocks and head (1792-d), average pooling and
``Classifier2D`` (dropout 0.5, ``Linear(1792, 2)``).

Upstream trains with a per-class focal BCE on the raw 2 logits
(``LOSS: FocalLoss, logits: True``, one-hot labels, index 1 = fake) and
evaluates with ``softmax(sigmoid(logits))[:, 1]`` because ``Classifier2D``
applies its sigmoid in eval mode; this port returns the raw logits so
``softmax[:, 1]`` is the natural fake probability of the trained pair
(same ranking, sharper calibration).  Checkpoint ``FF-C23`` from the model
zoo: ``{"model_state": ...}`` with ``model.*``, ``layers.*``,
``classifier.projection.*`` keys, loaded strictly.  Preprocessing
(``TEST_AUGMENTATIONS``): resize to 320x320 + ImageNet mean/std.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.fft  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from efficientnet_m2tr import EfficientNetM2TR  # noqa: E402

IMG_SIZE = 320
DEPTH = 4
NUM_CLASSES = 2
DROP_RATIO = 0.5


class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=80, w=41):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        b, _, a, c = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        dtype = x.dtype
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.irfft2(x, s=(a, c), dim=(1, 2), norm="ortho")
        return x.to(dtype).permute(0, 3, 1, 2).contiguous()


class FreqBlock(nn.Module):
    def __init__(self, dim, h=80, w=41):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.feed_forward = FeedForward2D(in_channel=dim, out_channel=dim)

    def forward(self, x):
        return x + self.feed_forward(self.filter(x))


def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1), nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(
                self.patchsize, torch.chunk(_query, len(self.patchsize), dim=1),
                torch.chunk(_key, len(self.patchsize), dim=1), torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            query = query.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5).contiguous() \
                .view(b, out_h * out_w, d_k * height * width)
            key = key.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5).contiguous() \
                .view(b, out_h * out_w, d_k * height * width)
            value = value.view(b, d_k, out_h, height, out_w, width).permute(0, 2, 4, 1, 3, 5).contiguous() \
                .view(b, out_h * out_w, d_k * height * width)
            y, _ = attention(query, key, value)
            y = y.view(b, out_h, out_w, d_k, height, width).permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)
            output.append(y)
        return self.output_linear(torch.cat(output, 1))


class TransformerBlock(nn.Module):
    def __init__(self, patchsize, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(in_channel=in_channel, out_channel=in_channel)

    def forward(self, rgb):
        output = rgb + self.attention(rgb)
        return output + self.feed_forward(output)


class CMA_Block(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0)
        self.scale = hidden_channel ** -0.5
        self.conv4 = nn.Sequential(nn.Conv2d(hidden_channel, out_channel, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, rgb, freq):
        _, _, h, w = rgb.size()
        q = self.conv1(rgb)
        k = self.conv2(freq)
        v = self.conv3(freq)
        q = q.view(q.size(0), q.size(1), -1).transpose(-2, -1)
        k = k.view(k.size(0), k.size(1), -1)
        m = (torch.matmul(q, k) * self.scale).softmax(dim=-1)
        v = v.view(v.size(0), v.size(1), -1).transpose(-2, -1)
        z = torch.matmul(m, v).view(q.size(0), h, w, -1).permute(0, 3, 1, 2).contiguous()
        return rgb + self.conv4(z)


class PatchTrans(nn.Module):
    def __init__(self, in_channel, in_size):
        super().__init__()
        patchsize = [(in_size, in_size), (in_size // 2, in_size // 2), (in_size // 4, in_size // 4),
                     (in_size // 8, in_size // 8)]
        self.t = TransformerBlock(patchsize, in_channel=in_channel)

    def forward(self, enc_feat):
        return self.t(enc_feat)


class Classifier2D(nn.Module):
    def __init__(self, dim_in, num_classes, dropout_rate=0.0):
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return self.projection(x)


class M2TRDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNetM2TR("efficientnet-b4", NUM_CLASSES)
        freq_h = IMG_SIZE // 4
        freq_w = freq_h // 2 + 1
        texture_dim, feature_dim = 32, 1792
        self.layers = nn.ModuleList([])
        for _ in range(DEPTH):
            self.layers.append(nn.ModuleList([
                PatchTrans(in_channel=texture_dim, in_size=freq_h),
                FreqBlock(dim=texture_dim, h=freq_h, w=freq_w),
                CMA_Block(in_channel=texture_dim, hidden_channel=texture_dim, out_channel=texture_dim),
            ]))
        self.classifier = Classifier2D(feature_dim, NUM_CLASSES, DROP_RATIO)

    def forward_detailed(self, x):
        layers = {}
        rgb = self.model.extract_textures(x, layers)
        for attn, filt, cma in self.layers:
            rgb = attn(rgb)
            freq = filt(rgb)
            rgb = cma(rgb, freq)
        features = self.model.extract_features(rgb, layers)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        logits = self.classifier(pooled)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": features}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = M2TRDetector()
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
