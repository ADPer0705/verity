"""LipFD detector (Liu, She, Liu, Li, Yao, Liang, Wang, NeurIPS 2024, "Lips
Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in
Lip-syncing DeepFakes", arXiv 2401.15668).

Ported from ``models/LipFD.py`` + ``models/region_awareness.py`` +
``preprocess.py`` + ``data/datasets.py`` of github.com/AaronComo/LipFD (no
license file), checkpoint ``ckpt.pth`` (Google Drive; trained on AVLips).
Shared OpenAI-layout CLIP from ``detectors/common/backbones/clip_openai``
(the checkpoint embeds the full ViT-L/14 under ``encoder.*``).

LipFD scores 5-frame windows of a talking-head video together with the
audio of the same span, as one composite image:

* ``preprocess.py``: the 5 frames are resized to 500x500 and tiled
  horizontally (500 x 2500); the mel spectrogram of the whole soundtrack
  (librosa, 22.05 kHz, ``power_to_db(ref=min)``) is rendered with
  matplotlib's viridis colormap, the columns covering the window are
  resized to 500 x 2500 and stacked **on top** -> 1000 x 2500 RGB image.
* ``data/datasets.py``: the image is read back with OpenCV (**BGR**) as a
  0-255 float tensor; the composite is resized to 1120x1120 and encoded by
  ``conv1`` (5x5 stride 5 -> 224) + CLIP ViT-L/14 (768-d global feature);
  three sets of five region crops are taken from the frame row
  (``img[:, 500:, i:i+500]`` for i in 0..4 - note that upstream slides the
  window by one *pixel*, so the five crops are near-identical views of the
  first frame; reproduced verbatim), each resized to 224 and also cropped
  at 0.65x and 0.45x scale.  No mean/std normalisation is applied to the
  tensors that reach the networks.
* ``region_awareness.py``: a ResNet-50 encodes every crop; each crop
  feature is concatenated with the global CLIP feature (2048 + 768) and
  gated by a learned sigmoid weight (softmax over the 5 views of a scale),
  the three scales are averaged and ``fc`` gives one logit; P(fake) =
  sigmoid (label 1 = fake).

Engine contract (``input_mode: av_clip``, ``clip_size: 5``,
``audio_sample_rate: 22050``): ``forward_detailed(window)`` receives the 5
native frames, the full 22.05 kHz waveform, ``total_frames`` and the window's
first frame index, and reproduces the composite exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.backbones.clip_openai import build_clip  # noqa: E402

WINDOW_LEN = 5
FRAME_SIZE = 500
COMPOSITE = 1120
CROP_IDX = [(28, 196), (61, 163)]


# -- region-awareness ResNet-50 (verbatim module names) -------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class RegionAwareResNet(nn.Module):
    def __init__(self, layers=(3, 4, 6, 3), global_dim=768):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.get_weight = nn.Sequential(nn.Linear(2048 + global_dim, 1), nn.Sigmoid())
        self.fc = nn.Linear(2048 + global_dim, 1)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def encode(self, f):
        f = self.maxpool(self.relu(self.bn1(self.conv1(f))))
        f = self.layer4(self.layer3(self.layer2(self.layer1(f))))
        return torch.flatten(self.avgpool(f), 1)

    def forward(self, crops, feature):
        """crops: list (scales) of lists (views) of (B, 3, 224, 224) tensors."""
        parts, weights_max, weights_org = [], [], []
        for i in range(len(crops[0])):
            feats, ws = [], []
            for j in range(len(crops)):
                f = torch.cat([self.encode(crops[j][i]), feature], dim=1)
                feats.append(f)
                ws.append(self.get_weight(f))
            feats = torch.stack(feats, dim=2)
            ws = F.softmax(torch.stack(ws, dim=2), dim=2)
            weights_max.append(ws[:, :, :len(crops)].max(dim=2)[0])
            weights_org.append(ws[:, :, 0])
            parts.append(feats.mul(ws).sum(2).div(ws.sum(2)))
        out = torch.stack(parts, dim=0).sum(0).div(len(parts))
        return self.fc(out), weights_max, weights_org


class LipFDDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5, stride=5)
        self.encoder = build_clip()
        self.backbone = RegionAwareResNet()

    # -- composite image exactly as preprocess.py + datasets.py ------------------
    @staticmethod
    def _mel_image(audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa
        from matplotlib import colormaps
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio.astype("float32"), sr=sr), ref=np.min)
        lo, hi = float(mel.min()), float(mel.max())
        norm = (mel - lo) / (hi - lo) if hi > lo else np.zeros_like(mel)
        rgba = colormaps["viridis"](norm)                       # plt.imsave default colormap
        return (rgba * 255).astype(np.uint8)                     # (128, T, 4), as plt.imread * 255

    def _composite(self, window) -> np.ndarray:
        import cv2
        frames = window["frames"][:WINDOW_LEN]
        while len(frames) < WINDOW_LEN:
            frames = frames + [frames[-1]]
        tiles = []
        for fr in frames:
            rgba = cv2.cvtColor(fr, cv2.COLOR_BGR2RGBA)
            tiles.append(cv2.resize(rgba, (FRAME_SIZE, FRAME_SIZE)))
        x = np.concatenate(tiles, axis=1)                        # (500, 2500, 4)
        mel = self._mel_image(window["audio_full"], int(window["sample_rate"]))
        frame_count = max(1, int(window["total_frames"]))
        start = int(window["frame_indices"][0])
        mapping = mel.shape[1] / frame_count
        begin = int(np.round(start * mapping))
        end = int(np.round((start + WINDOW_LEN) * mapping))
        end = max(end, begin + 1)
        begin = min(begin, mel.shape[1] - 1)
        end = min(end, mel.shape[1])
        sub_mel = cv2.resize(mel[:, begin:end], (FRAME_SIZE * WINDOW_LEN, FRAME_SIZE))
        comp = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)   # (1000, 2500, 3) RGB
        return comp[:, :, ::-1].copy()                           # datasets.py reads it back with cv2 -> BGR

    def _tensors(self, composite_bgr: np.ndarray, device):
        img = torch.from_numpy(composite_bgr).to(device).permute(2, 0, 1).float()   # (3, 1000, 2500), 0-255
        crops0 = [F.interpolate(img[None, :, FRAME_SIZE:, i:i + FRAME_SIZE], size=(224, 224), mode="bilinear",
                                align_corners=False, antialias=True) for i in range(WINDOW_LEN)]
        crops1 = [F.interpolate(c[:, :, CROP_IDX[0][0]:CROP_IDX[0][1], CROP_IDX[0][0]:CROP_IDX[0][1]],
                                size=(224, 224), mode="bilinear", align_corners=False, antialias=True) for c in crops0]
        crops2 = [F.interpolate(c[:, :, CROP_IDX[1][0]:CROP_IDX[1][1], CROP_IDX[1][0]:CROP_IDX[1][1]],
                                size=(224, 224), mode="bilinear", align_corners=False, antialias=True) for c in crops0]
        full = F.interpolate(img[None], size=(COMPOSITE, COMPOSITE), mode="bilinear", align_corners=False,
                             antialias=True)
        return full, [crops0, crops1, crops2]

    def get_features(self, x):
        return self.encoder.encode_image(self.conv1(x))

    def forward_detailed(self, window):
        device = next(self.parameters()).device
        comp = self._composite(window)
        full, crops = self._tensors(comp, device)
        features = self.get_features(full)
        score, weights_max, weights_org = self.backbone(crops, features)
        logits = torch.cat([torch.zeros_like(score), score], dim=1)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "info": {"region_weight_max": [round(float(w[0]), 4) for w in weights_max],
                         "window_frames": WINDOW_LEN}}

    def forward(self, window):
        return self.forward_detailed(window)["logits"]


def build_model(weights_path, device):
    model = LipFDDetector()
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
