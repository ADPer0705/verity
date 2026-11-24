"""RealForensics detector (Haliassos, Mira, Petridis, Pantic, CVPR 2022,
"Leveraging Real Talking Faces via Self-Supervision for Robust Forgery
Detection").

Ported from ``stage2`` of github.com/ahaliassos/RealForensics (MIT).  The
stage-2 detector is a channel-separated 3D ResNet-101 (``csn_temporal_no_head``:
pytorchvideo CSN with a max-pool stem and no temporal striding, 2048-d output
averaged over space) followed by ``MeanLinear`` (``norm_linear``: temporal
mean, then cosine linear ``F.linear(normalize(x), normalize(W)) * 64``) with a
single logit, P(fake) = sigmoid.  The backbone was first trained
self-supervised on real talking faces (LRW, BYOL-AV) and then fine-tuned with
the forgery loss on FaceForensics++ c23.  The CSN is re-implemented here with
pytorchvideo's module/key names (``blocks.0`` stem, ``blocks.1-4`` stages of
``res_blocks[i].branch1_conv/branch1_norm/branch2.conv_a..norm_c``) so the
released ``realforensics_ff.pth`` loads strictly without the pytorchvideo
dependency.

Preprocessing reproduces ``preprocessing/extract_faces.py`` + the validation
transform in ``stage2/data/combined_dm.py`` on the engine's 25-frame face
clip: FAN 68-point landmarks per frame, landmarks smoothed over a 13-frame
window and re-centred, similarity transform (``cv2.estimateAffinePartial2D``,
LMEDS) of 8 stable points (28, 33, 36, 39, 42, 45, 48, 54) onto the 256x256
LRW mean face, 150x150 patch centred on landmarks 15:68, then /255,
CenterCrop(140), Resize(112) and ImageNet mean/std (``channel: rgb``).  The
clip is scored as ``(1, 3, 25, 112, 112)``.
"""
from __future__ import annotations

import os
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
MEAN_FACE_PATH = os.path.join(_MODEL_DIR, "20words_mean_face.npy")

_STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
_START_IDX, _STOP_IDX = 15, 68
_WINDOW_MARGIN = 12
_CROP = 150
_CANVAS = 256
_CENTER_CROP = 140
_RESIZE = 112
_INPUT_MEAN = (0.5, 0.5, 0.5)
_INPUT_STD = (0.5, 0.5, 0.5)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SCALE = 64.0

_ARCH = None


def _get_arch():
    global _ARCH
    if _ARCH is not None:
        return _ARCH
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResNetBasicStem(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv3d(in_channels, out_channels, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
            self.norm = nn.BatchNorm3d(out_channels)
            self.activation = nn.ReLU()
            self.pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        def forward(self, x):
            return self.pool(self.activation(self.norm(self.conv(x))))

    class BottleneckBlock(nn.Module):
        def __init__(self, dim_in, dim_inner, dim_out, stride):
            super().__init__()
            self.conv_a = nn.Conv3d(dim_in, dim_inner, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
            self.norm_a = nn.BatchNorm3d(dim_inner)
            self.act_a = nn.ReLU()
            # channel-separated (depthwise) 3x3x3 convolution
            self.conv_b = nn.Conv3d(dim_inner, dim_inner, (3, 3, 3), stride=stride, padding=(1, 1, 1),
                                    groups=dim_inner, bias=False)
            self.norm_b = nn.BatchNorm3d(dim_inner)
            self.act_b = nn.ReLU()
            self.conv_c = nn.Conv3d(dim_inner, dim_out, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
            self.norm_c = nn.BatchNorm3d(dim_out)

        def forward(self, x):
            x = self.act_a(self.norm_a(self.conv_a(x)))
            x = self.act_b(self.norm_b(self.conv_b(x)))
            return self.norm_c(self.conv_c(x))

    class ResBlock(nn.Module):
        def __init__(self, dim_in, dim_inner, dim_out, stride):
            super().__init__()
            if dim_in != dim_out or any(s > 1 for s in stride):
                self.branch1_conv = nn.Conv3d(dim_in, dim_out, (1, 1, 1), stride=stride, bias=False)
                self.branch1_norm = nn.BatchNorm3d(dim_out)
            else:
                self.branch1_conv = None
                self.branch1_norm = None
            self.branch2 = BottleneckBlock(dim_in, dim_inner, dim_out, stride)
            self.activation = nn.ReLU()

        def forward(self, x):
            shortcut = x if self.branch1_conv is None else self.branch1_norm(self.branch1_conv(x))
            return self.activation(shortcut + self.branch2(x))

    class ResStage(nn.Module):
        def __init__(self, depth, dim_in, dim_inner, dim_out, stride):
            super().__init__()
            blocks = []
            for i in range(depth):
                blocks.append(ResBlock(dim_in if i == 0 else dim_out, dim_inner, dim_out,
                                       stride if i == 0 else (1, 1, 1)))
            self.res_blocks = nn.ModuleList(blocks)

        def forward(self, x):
            for b in self.res_blocks:
                x = b(x)
            return x

    class CSN(nn.Module):
        """``csn_temporal_no_head(model_depth=101, input_channel=3)``."""

        def __init__(self, depth=101, in_channels=3):
            super().__init__()
            stage_depths = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}[depth]
            blocks = [ResNetBasicStem(in_channels, 64)]
            dim_in, dim_out = 64, 256
            spatial = (1, 2, 2, 2)
            for idx, d in enumerate(stage_depths):
                blocks.append(ResStage(d, dim_in, dim_out // 4, dim_out, (1, spatial[idx], spatial[idx])))
                dim_in, dim_out = dim_out, dim_out * 2
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x.mean(dim=(3, 4))          # (B, 2048, T)

    class MeanLinear(nn.Module):
        def __init__(self, in_dim=2048, out_dim=1, scale=_SCALE):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            self.scale = scale

        def forward(self, x):
            x = x.mean(-1)
            return F.linear(F.normalize(x), F.normalize(self.weight)) * self.scale

    class RealForensicsDetector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = CSN(101, 3)
            self.df_head = MeanLinear(2048, 1)

        def forward(self, x):
            return self.df_head(self.backbone(x))

        def forward_detailed(self, x):
            clip = _preprocess_clip(x)
            feat = self.backbone(clip)
            logit = self.df_head(feat)
            logits = torch.zeros(x.shape[0], 2, dtype=x.dtype, device=x.device)
            logits[:, 1] = logit.squeeze(1)
            return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feat}

    _ARCH = {"RealForensicsDetector": RealForensicsDetector}
    return _ARCH


# -- preprocessing (extract_faces.py + val transform) --------------------------

_MEAN_FACE = None
_FACE_ALIGNER = None


def _load_mean_face():
    global _MEAN_FACE
    if _MEAN_FACE is None:
        import numpy as np
        _MEAN_FACE = np.load(MEAN_FACE_PATH)
    return _MEAN_FACE


def _get_face_aligner():
    global _FACE_ALIGNER
    if _FACE_ALIGNER is None:
        import face_alignment
        import torch
        lm_type = getattr(face_alignment.LandmarksType, "TWO_D", None) \
            or getattr(face_alignment.LandmarksType, "_2D")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _FACE_ALIGNER = face_alignment.FaceAlignment(lm_type, face_detector="folder",
                                                     flip_input=False, device=device)
    return _FACE_ALIGNER


def _landmarks(frames_rgb):
    import numpy as np
    fa = _get_face_aligner()
    out = []
    for frame in frames_rgb:
        h, w = frame.shape[:2]
        det = fa.get_landmarks_from_image(frame, detected_faces=[(0, 0, w, h)])
        out.append(np.asarray(det[0], dtype=np.float64) if det else None)
    return out


def _affine_transform(frame, landmarks, reference):
    import cv2
    import numpy as np
    stable_reference = np.vstack([reference[x] for x in _STABLE_POINTS]).astype(np.float32)
    src = np.vstack([landmarks[x] for x in _STABLE_POINTS]).astype(np.float32)
    transform = cv2.estimateAffinePartial2D(src, stable_reference, method=cv2.LMEDS)[0]
    if transform is None:
        return None, None
    warped = cv2.warpAffine(frame, transform, dsize=(_CANVAS, _CANVAS), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    pts = np.matmul(landmarks, transform[:, :2].T) + transform[:, 2].T
    return warped, pts


def _cut_patch(img, landmarks, height, width, threshold=5):
    import numpy as np
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise ValueError("too much bias in height")
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise ValueError("too much bias in width")
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise ValueError("too much bias in height")
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise ValueError("too much bias in width")
    return np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                       int(round(center_x) - round(width)): int(round(center_x) + round(width))])


def _crop_patch(frames_rgb, landmarks, reference):
    """Port of ``extract_faces.crop_patch``: smoothed + re-centred landmarks,
    similarity warp to the mean face, 150x150 patch around landmarks 15:68."""
    import numpy as np
    seq = []
    n = min(len(landmarks), len(frames_rgb))
    for i in range(n):
        margin = min(_WINDOW_MARGIN // 2, i, len(landmarks) - 1 - i)
        smoothed = np.mean([landmarks[x] for x in range(i - margin, i + margin + 1)], axis=0)
        smoothed += landmarks[i].mean(axis=0) - smoothed.mean(axis=0)
        warped, pts = _affine_transform(frames_rgb[i], smoothed, reference)
        if warped is None:
            raise ValueError("degenerate similarity transform")
        seq.append(_cut_patch(warped, pts[_START_IDX:_STOP_IDX], _CROP // 2, _CROP // 2))
    return seq


def _center_fallback(frames_rgb):
    """No landmarks: centre 150x150 of the face crop (resized to 256 canvas)."""
    import cv2
    out = []
    for f in frames_rgb:
        f = cv2.resize(f, (_CANVAS, _CANVAS))
        o = (_CANVAS - _CROP) // 2
        out.append(f[o:o + _CROP, o:o + _CROP].copy())
    return out


def _preprocess_clip(x):
    """(B, 3, T, H, W) engine clip (mean/std 0.5) -> (B, 3, T, 112, 112) RealForensics input."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    std = torch.tensor(_INPUT_STD, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    mean = torch.tensor(_INPUT_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
    rgb = (x * std + mean).clamp(0, 1)
    reference = _load_mean_face()
    clips = []
    for b in range(rgb.shape[0]):
        frames = [(rgb[b, :, t].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                  for t in range(rgb.shape[2])]
        lms = _landmarks(frames)
        try:
            if any(lm is None for lm in lms):
                raise ValueError("missing landmarks")
            patches = _crop_patch(frames, lms, reference)
        except ValueError:
            patches = _center_fallback(frames)
        clip = torch.from_numpy(np.stack(patches)).to(x.device).permute(3, 0, 1, 2).float() / 255.0  # (3, T, 150, 150)
        o = (_CROP - _CENTER_CROP) // 2
        clip = clip[:, :, o:o + _CENTER_CROP, o:o + _CENTER_CROP]
        clip = F.interpolate(clip.permute(1, 0, 2, 3), size=(_RESIZE, _RESIZE), mode="bilinear",
                             align_corners=False, antialias=True).permute(1, 0, 2, 3)
        m = torch.tensor(_IMAGENET_MEAN, device=x.device).view(3, 1, 1, 1)
        s = torch.tensor(_IMAGENET_STD, device=x.device).view(3, 1, 1, 1)
        clips.append((clip - m) / s)
    return torch.stack(clips)


def build_model(weights_path, device):
    import torch
    arch = _get_arch()
    model = arch["RealForensicsDetector"]()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
