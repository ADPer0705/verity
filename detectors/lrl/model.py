"""LRL detector (Chen et al. 2021): dual EfficientNet-B4 encoders (RGB + DCT).

Ported from DeepfakeBench ``training/detectors/lrl_detector.py``.  The engine
only supplies the normalized RGB ``image``, so the 1-channel DCT input for
``encoder_idct`` is recomputed inside ``forward_detailed`` with
``image_to_idct``: a torch port of the transform in ``training/dataset/lrl_dataset.py``
(grayscale -> orthonormal DCT-II, mathematically equivalent to ``cv2.dct`` ->
high-frequency triangular mask ``k*x + y >= 0.33*W`` -> IDCT -> normalize to
[-1, 1]).  The only difference is that OpenCV rounds the grayscale conversion to
uint8, while this port keeps float precision.  The checkpoint is the full
detector state dict (keys ``encoder_rgb.*``, ``encoder_idct.*``, ``rfam1..3.*``,
``feature_adjust1..2``, ``decoder.*``, ``final.*``, ``overall_classifier.*``).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class EfficientNetB4(nn.Module):
    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super().__init__()
        self.dropout = dropout
        self.mode = mode
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        if dropout:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.last_layer = nn.Linear(1792, num_classes)

    def block_part1(self, x):
        x = self.efficientnet._swish(self.efficientnet._bn0(self.efficientnet._conv_stem(x)))
        for block in self.efficientnet._blocks[:10]:
            x = block(x)
        return x

    def block_part2(self, x):
        for block in self.efficientnet._blocks[10:22]:
            x = block(x)
        return x

    def block_part3(self, x):
        for block in self.efficientnet._blocks[22:]:
            x = block(x)
        return self.efficientnet._swish(self.efficientnet._bn1(self.efficientnet._conv_head(x)))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class ConcatBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, dec, enc=None):
        return self.seq(dec)


class Decoder(nn.Module):
    def __init__(self, decoder_filters, filters, decoder_block=DecoderBlock,
                 bottleneck=ConcatBottleneck):
        super().__init__()
        self.decoder_filters = decoder_filters
        self.filters = filters
        self.decoder_block = decoder_block
        self.decoder_stages = nn.ModuleList([self._get_decoder(idx) for idx in range(0, len(decoder_filters))])
        self.bottlenecks = nn.ModuleList([bottleneck(f, f) for f in reversed(decoder_filters)])
        self.last_block = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, encoder_results):
        x = encoder_results[0]
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x)
        x = self.last_block(x)
        return x

    def _get_decoder(self, layer):
        idx = layer + 1
        if idx == len(self.decoder_filters):
            in_channels = self.filters[idx]
        else:
            in_channels = self.decoder_filters[idx]
        return self.decoder_block(in_channels, self.decoder_filters[max(layer, 0)])


class RFAM(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(features * 2, features, 1),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, 2, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        U = torch.cat((x1, x2), dim=1)
        A = self.attention(U)
        A1 = A[:, 0, ...].unsqueeze(1).contiguous()
        A2 = A[:, 1, ...].unsqueeze(1).contiguous()
        x1 *= A1
        x2 *= A2
        return x1, x2


class LRLDetector(nn.Module):
    _dct_cache: dict = {}

    def __init__(self, num_classes=2, inc=3, dropout=False, mode="Original"):
        super().__init__()
        self.encoder_rgb = EfficientNetB4(num_classes=num_classes, inc=inc, dropout=dropout, mode=mode)
        self.encoder_idct = EfficientNetB4(num_classes=num_classes, inc=1, dropout=dropout, mode=mode)
        self.encoder_idct.efficientnet._conv_stem = nn.Conv2d(1, 48, kernel_size=3, stride=2, bias=False)

        self.feature_adjust1 = nn.Upsample(scale_factor=0.25)
        self.feature_adjust2 = nn.Upsample(scale_factor=0.5)
        self.decoder = Decoder(decoder_filters=[64, 128, 256, 256],
                               filters=[48, 40, 64, 176, 2008])
        self.rfam1 = RFAM(56)
        self.rfam2 = RFAM(160)
        self.rfam3 = RFAM(1792)

        self.final = nn.Conv2d(64, out_channels=1, kernel_size=1, bias=False)

        self.overall_classifier = nn.Sequential(
            nn.Linear(240, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def features(self, rgb, idct):
        rgb1 = self.encoder_rgb.block_part1(rgb)
        idct1 = self.encoder_idct.block_part1(idct)
        rgb1, idct1 = self.rfam1(rgb1, idct1)
        featuremap_low = rgb1 + idct1

        rgb2 = self.encoder_rgb.block_part2(rgb1)
        idct2 = self.encoder_idct.block_part2(idct1)
        rgb2, idct2 = self.rfam2(rgb2, idct2)
        featuremap_mid = rgb2 + idct2

        rgb3 = self.encoder_rgb.block_part3(rgb2)
        idct3 = self.encoder_idct.block_part3(idct2)
        rgb3, idct3 = self.rfam3(rgb3, idct3)
        featuremap_high = rgb3 + idct3

        f1 = self.feature_adjust1(featuremap_low)
        f2 = self.feature_adjust2(featuremap_mid)
        f3 = featuremap_high
        return torch.cat((f1, f2, f3), dim=1)

    def feature_process(self, feature):
        w = F.unfold(feature, kernel_size=2, stride=2, padding=0).permute(0, 2, 1)
        w_normed = w / (w * w).sum(dim=2, keepdim=True).sqrt()
        B, K = w.shape[:2]
        sim = torch.einsum('bij,bjk->bik', w_normed, w_normed.permute(0, 2, 1))
        sim = (sim + 1) / 2
        mask = (torch.eye(K, device=feature.device) != 1).repeat(B, 1).view(B, K, K)
        sim_mask = torch.masked_select(sim, mask).view(B, K, -1)
        return sim_mask.view(B, -1), sim

    def _dct_matrix(self, n, device):
        cached = self._dct_cache.get((n, device))
        if cached is None:
            i = torch.arange(n, dtype=torch.float32, device=device)
            k = torch.arange(n, dtype=torch.float32, device=device)
            c = torch.cos((math.pi / n) * (i[None, :] + 0.5) * k[:, None])
            c *= math.sqrt(2.0 / n)
            c[0, :] *= math.sqrt(0.5)
            self._dct_cache[(n, device)] = c
            return c
        return cached

    def image_to_idct(self, x):
        """Port of ``LRLDataset.image2dct`` for a normalized (mean/std 0.5) input."""
        h, w = x.shape[-2], x.shape[-1]
        device = x.device
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        gray = (gray * 0.5 + 0.5) * 255.0
        c_h = self._dct_matrix(h, device)
        c_w = self._dct_matrix(w, device)
        dct = torch.einsum("ij,bjk->bik", c_h, gray)
        dct = torch.einsum("bik,lk->bil", dct, c_w)
        rows = torch.arange(h, device=device)[:, None]
        cols = torch.arange(w, device=device)[None, :]
        high = (w / h * rows + cols) >= 0.33 * w
        dct = dct * high.to(dtype=x.dtype)
        idct = torch.einsum("ij,bjk->bik", c_h.t(), dct)
        idct = torch.einsum("bik,lk->bil", idct, c_w.t())
        return ((idct / 255.0 - 0.5) / 0.5).unsqueeze(1)

    def forward_detailed(self, x):
        idct = self.image_to_idct(x)
        features = self.features(x, idct)
        features_processed, _ = self.feature_process(features)
        logits = self.overall_classifier(features_processed)
        mask = torch.sigmoid(self.final(self.decoder([features])))
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "feat": features, "mask": mask}


def build_model(weights_path, device):
    model = LRLDetector(num_classes=2, inc=3, dropout=False, mode="Original")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
