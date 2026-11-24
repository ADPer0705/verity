"""LAA-Net detector (Nguyen, Mejri, Singh, Kuleshova, Astrid, Kacem, Ghorbel,
Aouada, CVPR 2024, "LAA-Net: Localized Artifact Attention Network for
Quality-Agnostic and Generalizable Deepfake Detection").

Ported from ``models/networks/pose_efficientNet.py`` (``PoseEfficientNet``)
and ``models/networks/common.py`` of github.com/10Ring/LAA-Net (SnT academic
license, University of Luxembourg), with the ``configs/efn4_fpn_sbi_adv.yaml``
settings: EfficientNet-B4 trunk (``efficientnet_pytorch``; identical block
and key names), E-FPN decoder (``deconv_1..4``, C5'/C4/C3 skip features
gated by ``(1 - sigmoid(x))``, Inception block before the last up-sampling)
and three heads on the 112-channel 96x96 map: ``hm`` (1-channel localized
artifact heatmap), ``cls`` (global pool + Linear(64, 1)) and ``cstency``
(256-d self-consistency, unused at inference).

Scoring follows ``scripts/test.py`` with ``flip_test: True``: the image and
its horizontal flip are both scored and the ``cls`` outputs averaged; the
training loop applies ``sigmoid`` to ``cls`` before the BCE
(``lib/core_function.py``), so P(fake) = sigmoid(mean logit).  The sigmoid
heatmap (flip-averaged, un-flipped) is the native attention map.

Preprocessing: ``package_utils/images_crop.py`` enlarges the RetinaFace box
by 25 % of its size on every side (``PADDING = 0.25`` -> ``face_margin:
0.5``) and resizes to 256; the test transform resizes to 384 (affine) and
normalises with mean/std 0.5.  Verity crops with the same margin and resizes
straight to 384 (``resolution: 384``).

Checkpoint: the SBI-trained ``..._8SBI_..._model_best.pth`` (``state_dict``
with 803 tensors; ``deconv_4.0`` and ``inception_block`` are the same
module, as upstream).  The BI-trained alternate shares the layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402

BN_MOMENTUM = 0.1
DECODER_FILTERS = [1792, 448, 160, 56]
HEAD_CONV = 64
HEADS = {"hm": 1, "cls": 1, "cstency": 256}
BACKBONE_PREFIXES = ("_conv_stem", "_bn0", "_blocks", "_conv_head", "_bn1")


def point_wise_block(inplanes, outplanes):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )


def conv_block(inplanes, outplanes, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM),
        nn.ReLU(inplace=True),
    )


class InceptionBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, pool_size=3):
        super().__init__()
        self.pw_block = point_wise_block(inplanes, outplanes // 4)
        self.mp_layer = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=1)
        self.conv3_block = conv_block(outplanes // 4, outplanes // 4, kernel_size=3, stride=1, padding=1)
        self.conv5_block = conv_block(outplanes // 4, outplanes // 4, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x1 = self.pw_block(x)
        x2 = self.conv3_block(self.pw_block(x))
        x3 = self.conv5_block(self.pw_block(x))
        x4 = self.pw_block(self.mp_layer(x))
        return torch.cat((x1, x2, x3, x4), dim=1)


def _deconv(out_ch):
    return nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False)


class LAANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b4", include_top=False)
        f = DECODER_FILTERS
        self._dropout = nn.Dropout(0.4)
        self._sigmoid = nn.Sigmoid()
        self._relu = nn.ReLU(inplace=True)
        self.inception_block = InceptionBlock(112, 112, stride=1, pool_size=3)
        self.deconv_1 = nn.Sequential(conv_block(f[0], f[1], (3, 3), stride=1, padding=1))
        self.deconv_2 = nn.Sequential(conv_block(f[1] * 2, f[2], (3, 3), stride=1, padding=1), _deconv(f[2]),
                                      nn.BatchNorm2d(f[2], momentum=BN_MOMENTUM))
        self.deconv_3 = nn.Sequential(conv_block(f[2] * 2, f[3], (3, 3), stride=1, padding=1), _deconv(f[3]),
                                      nn.BatchNorm2d(f[3], momentum=BN_MOMENTUM))
        self.deconv_4 = nn.Sequential(self.inception_block, _deconv(f[3] * 2),
                                      nn.BatchNorm2d(f[3] * 2, momentum=BN_MOMENTUM))
        in_head = f[3] * 2
        self.hm = nn.Sequential(
            nn.Conv2d(in_head, HEAD_CONV, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(HEAD_CONV),
            nn.ReLU(inplace=True), nn.Conv2d(HEAD_CONV, HEADS["hm"], kernel_size=1, stride=1, padding=0))
        self.cls = nn.Sequential(
            nn.Conv2d(in_head, HEAD_CONV, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(HEAD_CONV, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(HEAD_CONV, HEADS["cls"], bias=True))
        self.cstency = nn.Sequential(
            nn.Conv2d(in_head, HEAD_CONV, kernel_size=3, padding=1, bias=True), nn.BatchNorm2d(HEAD_CONV),
            nn.ReLU(inplace=True), nn.Conv2d(HEAD_CONV, HEADS["cstency"], kernel_size=1, stride=1, padding=0))

    def _gate_cat(self, x, skip):
        return torch.cat([x, (1 - self._sigmoid(x)) * skip], dim=1)

    def forward_single(self, x):
        ep = self.backbone.extract_endpoints(x)
        x1, x2, x3, x4 = ep["reduction_6"], ep["reduction_5"], ep["reduction_4"], ep["reduction_3"]
        x1, x2, x3, x4 = self._dropout(x1), self._dropout(x2), self._dropout(x3), self._dropout(x4)
        y = self.deconv_1(x1)
        y = self.deconv_2(self._gate_cat(y, x2))
        y = self.deconv_3(self._gate_cat(y, x3))
        y = self.deconv_4(self._gate_cat(y, x4))
        y = self._relu(y)
        return {"hm": self.hm(y), "cls": self.cls(y), "feat": y}


class LAANetDetector(nn.Module):
    def __init__(self, flip_test=True):
        super().__init__()
        self.net = LAANet()
        self.flip_test = flip_test

    def forward_detailed(self, x):
        out = self.net.forward_single(x)
        hm, cls = out["hm"], out["cls"]
        if self.flip_test:
            out_f = self.net.forward_single(x.flip(dims=(3,)))
            hm = (hm + out_f["hm"].flip(dims=(3,))) / 2
            cls = (cls + out_f["cls"]) / 2
        fake_logit = cls                                      # (B, 1); upstream sigmoid(cls) is P(fake)
        logits = torch.cat([torch.zeros_like(fake_logit), fake_logit], dim=1)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": out["feat"],
                "attention_map": torch.sigmoid(hm[:, 0])}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = LAANetDetector()
    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        remapped = {("backbone." + k if k.startswith(BACKBONE_PREFIXES) else k): v for k, v in state.items()}
        own = model.net.state_dict()
        missing = sorted(set(own) - set(remapped))
        unexpected = sorted(set(remapped) - set(own))
        if missing or unexpected:
            raise RuntimeError(f"LAA-Net checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        model.net.load_state_dict(remapped, strict=True)
    model.to(device)
    model.eval()
    return model
