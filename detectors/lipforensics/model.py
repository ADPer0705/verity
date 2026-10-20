"""LipForensics detector (Haliassos et al., CVPR 2021, "Lips Don't Lie").

Full port of the official repository (ahaliassos/LipForensics).  The network
is the Lipreading model -- Conv3d frontend + ResNet-18 trunk (PReLU) +
multiscale multibranch temporal conv net -- with ``num_classes=1``; the
official FF++ checkpoint ``lipforensics_ff.pth`` holds the plain Lipreading
state dict under key ``model`` and loads with ``load_state_dict`` directly.

``forward_detailed`` consumes the engine's normalized face-crop clip
``(B, 3, T, H, W)`` (H=W=256; yaml mean/std are the round-trip 0.5 values),
un-normalizes it back to [0,1] RGB, runs FAN-68 landmark detection on every
frame, replicates the official preprocessing (``crop_mouths.py`` +
``data/transforms.py`` + ``evaluate.py``): 12-frame landmark smoothing,
similarity warp of frame and landmarks to the LRW mean face (256x256),
96x96 mouth crop, grayscale, center 88x88, normalize (0.421, 0.165).  The
resulting ``(B, 1, 25, 88, 88)`` clip is scored; the single logit is emitted
as 2-class logits ``[0, logit]`` so ``softmax[:, 1] == sigmoid(logit)``.

All heavy imports (torch, numpy, cv2, face_alignment) are deferred to
call time so this module imports cleanly without any of them installed.
"""
from __future__ import annotations

import os
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent
MEAN_FACE_PATH = os.path.join(_MODEL_DIR, "20words_mean_face.npy")

_CLIP_SIZE = 25
_STD_SIZE = 256
_MOUTH_SIZE = 96
_MOUTH_START = 48
_MOUTH_END = 68
_STABLE_POINTS = [33, 36, 39, 42, 45]
_SMOOTH_WINDOW = 12

_INPUT_MEAN = (0.5, 0.5, 0.5)
_INPUT_STD = (0.5, 0.5, 0.5)
_GRAY_MEAN = 0.421
_GRAY_STD = 0.165

_TCN_OPTIONS = {
    "kernel_size": [3, 5, 7],
    "num_layers": 4,
    "dropout": 0.2,
    "dwpw": False,
    "width_mult": 1,
}

_ARCH = None


def _get_arch():
    """Lazily define the torch Lipreading architecture (imports torch)."""
    global _ARCH
    if _ARCH is not None:
        return _ARCH
    import torch
    import torch.nn as nn

    def conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def downsample_basic_block(inplanes, outplanes, stride):
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type="relu"):
            super(BasicBlock, self).__init__()
            assert relu_type in ["relu", "prelu"]
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = nn.BatchNorm2d(planes)
            if relu_type == "relu":
                self.relu1 = nn.ReLU(inplace=True)
                self.relu2 = nn.ReLU(inplace=True)
            else:
                self.relu1 = nn.PReLU(num_parameters=planes)
                self.relu2 = nn.PReLU(num_parameters=planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            residual = x
            out = self.relu1(self.bn1(self.conv1(x)))
            out = self.conv2(out)
            out = self.bn2(out)
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
            out = self.relu2(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, layers, relu_type="relu"):
            self.inplanes = 64
            super(ResNet, self).__init__()
            self.layer1 = self._make_layer(block, 64, layers[0], relu_type=relu_type)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, relu_type=relu_type)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, relu_type=relu_type)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, relu_type=relu_type)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, (2.0 / n) ** 0.5)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def _make_layer(self, block, planes, blocks, stride=1, relu_type="relu"):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = downsample_basic_block(self.inplanes, planes * block.expansion, stride)
            layers = [block(self.inplanes, planes, stride, downsample, relu_type=relu_type)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, relu_type=relu_type))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return x

    class Chomp1d(nn.Module):
        def __init__(self, chomp_size, symm_chomp):
            super(Chomp1d, self).__init__()
            self.chomp_size = chomp_size
            self.symm_chomp = symm_chomp

        def forward(self, x):
            if self.chomp_size == 0:
                return x
            if self.symm_chomp:
                return x[:, :, self.chomp_size // 2 : -self.chomp_size // 2].contiguous()
            return x[:, :, : -self.chomp_size].contiguous()

    class ConvBatchChompRelu(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
            super(ConvBatchChompRelu, self).__init__()
            self.dwpw = dwpw
            if dwpw:
                self.conv = nn.Sequential(
                    nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=n_inputs, bias=False),
                    nn.BatchNorm1d(n_inputs),
                    Chomp1d(padding, True),
                    nn.PReLU(num_parameters=n_inputs) if relu_type == "prelu" else nn.ReLU(inplace=True),
                    nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                    nn.BatchNorm1d(n_outputs),
                    nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU(inplace=True),
                )
            else:
                self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation)
                self.batchnorm = nn.BatchNorm1d(n_outputs)
                self.chomp = Chomp1d(padding, True)
                self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU()

        def forward(self, x):
            if self.dwpw:
                return self.conv(x)
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)

    class MultibranchTemporalBlock(nn.Module):
        def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding,
                     dropout=0.2, relu_type="relu", dwpw=False):
            super(MultibranchTemporalBlock, self).__init__()
            self.kernel_sizes = kernel_sizes
            self.num_kernels = len(kernel_sizes)
            self.n_outputs_branch = n_outputs // self.num_kernels
            for k_idx, k in enumerate(self.kernel_sizes):
                cbcr = ConvBatchChompRelu(n_inputs, self.n_outputs_branch, k, stride,
                                          dilation, padding[k_idx], relu_type, dwpw=dwpw)
                setattr(self, "cbcr0_{}".format(k_idx), cbcr)
            self.dropout0 = nn.Dropout(dropout)
            for k_idx, k in enumerate(self.kernel_sizes):
                cbcr = ConvBatchChompRelu(n_outputs, self.n_outputs_branch, k, stride,
                                          dilation, padding[k_idx], relu_type, dwpw=dwpw)
                setattr(self, "cbcr1_{}".format(k_idx), cbcr)
            self.dropout1 = nn.Dropout(dropout)
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs // self.num_kernels) != n_outputs else None
            self.relu_final = nn.PReLU(num_parameters=n_outputs) if relu_type == "prelu" else nn.ReLU()

        def forward(self, x):
            outputs = [getattr(self, "cbcr0_{}".format(k_idx))(x) for k_idx in range(self.num_kernels)]
            out0 = self.dropout0(torch.cat(outputs, 1))
            outputs = [getattr(self, "cbcr1_{}".format(k_idx))(out0) for k_idx in range(self.num_kernels)]
            out1 = self.dropout1(torch.cat(outputs, 1))
            res = x if self.downsample is None else self.downsample(x)
            return self.relu_final(out1 + res)

    class MultibranchTemporalConvNet(nn.Module):
        def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type="relu", dwpw=False):
            super(MultibranchTemporalConvNet, self).__init__()
            self.ksizes = tcn_options["kernel_size"]
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = num_inputs if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                padding = [(s - 1) * dilation_size for s in self.ksizes]
                layers.append(MultibranchTemporalBlock(
                    in_channels, out_channels, self.ksizes, stride=1, dilation=dilation_size,
                    padding=padding, dropout=dropout, relu_type=relu_type, dwpw=dwpw))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    def reshape_tensor(x):
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        return x.reshape(n_batch * s_time, n_channels, sx, sy)

    def _average_batch(x, lengths):
        return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)

    class MultiscaleMultibranchTCN(nn.Module):
        def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
            super(MultiscaleMultibranchTCN, self).__init__()
            self.kernel_sizes = tcn_options["kernel_size"]
            self.num_kernels = len(self.kernel_sizes)
            self.mb_ms_tcn = MultibranchTemporalConvNet(
                input_size, num_channels, tcn_options, dropout=dropout, relu_type=relu_type, dwpw=dwpw)
            self.tcn_output = nn.Linear(num_channels[-1], num_classes)
            self.consensus_func = _average_batch

        def forward(self, x, lengths):
            x = x.transpose(1, 2)
            out = self.mb_ms_tcn(x)
            out = self.consensus_func(out, lengths)
            return self.tcn_output(out)

        def forward_with_features(self, x, lengths):
            x = x.transpose(1, 2)
            out = self.mb_ms_tcn(x)
            feat = self.consensus_func(out, lengths)
            return self.tcn_output(feat), feat

    class Lipreading(nn.Module):
        def __init__(self, hidden_dim=256, num_classes=500, relu_type="prelu", tcn_options=None):
            super(Lipreading, self).__init__()
            tcn_options = tcn_options or {}
            self.frontend_nout = 64
            self.backend_out = 512
            self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == "prelu" else nn.ReLU()
            self.frontend3D = nn.Sequential(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2),
                          padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(self.frontend_nout),
                frontend_relu,
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            )
            self.tcn = MultiscaleMultibranchTCN(
                input_size=self.backend_out,
                num_channels=[hidden_dim * len(tcn_options["kernel_size"]) * tcn_options["width_mult"]]
                * tcn_options["num_layers"],
                num_classes=num_classes,
                tcn_options=tcn_options,
                dropout=tcn_options["dropout"],
                relu_type=relu_type,
                dwpw=tcn_options["dwpw"],
            )

        def forward(self, x, lengths):
            x = self.frontend3D(x)
            t_new = x.shape[2]
            x = reshape_tensor(x)
            x = self.trunk(x)
            x = x.view(-1, t_new, x.size(1))
            return self.tcn(x, lengths)

        def forward_with_features(self, x, lengths):
            x = self.frontend3D(x)
            t_new = x.shape[2]
            x = reshape_tensor(x)
            x = self.trunk(x)
            x = x.view(-1, t_new, x.size(1))
            return self.tcn.forward_with_features(x, lengths)

    class LipForensicsDetector(nn.Module):
        def __init__(self, clip_size=_CLIP_SIZE):
            super(LipForensicsDetector, self).__init__()
            self.clip_size = clip_size
            self.network = Lipreading(num_classes=1, tcn_options=_TCN_OPTIONS, relu_type="prelu")

        def forward(self, x):
            return self.network(x, lengths=[self.clip_size] * x.shape[0])

        def forward_detailed(self, x):
            import numpy as np

            mean_face = _load_mean_face()
            std = torch.tensor(_INPUT_STD, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
            mean = torch.tensor(_INPUT_MEAN, dtype=x.dtype, device=x.device).view(3, 1, 1, 1)
            rgb = x * std + mean

            logits_list, feats_list = [], []
            for b in range(rgb.shape[0]):
                pil_frames = _tensor_to_pil_frames(rgb[b])
                landmarks = _extract_landmarks(pil_frames)
                if any(lm is None for lm in landmarks):
                    crops = _center_crop_fallback(pil_frames)
                else:
                    crops = _align_and_crop_clip(pil_frames, np.stack(landmarks), mean_face)
                clip = _crops_to_clip_tensor(crops, x.device)
                logit, feat = self.network.forward_with_features(clip, [self.clip_size])
                logits_list.append(logit)
                feats_list.append(feat)

            logits = torch.zeros(rgb.shape[0], 2, dtype=x.dtype, device=x.device)
            logits[:, 1] = torch.cat(logits_list).squeeze(1)
            return {
                "logits": logits,
                "probs": torch.softmax(logits, dim=1),
                "feat": torch.cat(feats_list),
            }

    _ARCH = {
        "LipForensicsDetector": LipForensicsDetector,
    }
    return _ARCH


_MEAN_FACE = None


def _load_mean_face():
    global _MEAN_FACE
    if _MEAN_FACE is None:
        import numpy as np
        _MEAN_FACE = np.load(MEAN_FACE_PATH)
    return _MEAN_FACE


_FACE_ALIGNER = None


def _get_face_aligner():
    """FAN 68-landmark model; weights (~150MB) auto-download on first use."""
    global _FACE_ALIGNER
    if _FACE_ALIGNER is None:
        import face_alignment
        import torch
        # face-alignment 1.4 renamed LandmarksType._2D -> TWO_D and dropped the
        # "none" face detector; "folder" is its lazy no-op equivalent (it loads
        # no weights and is never invoked because we always pass detected_faces).
        lm_type = getattr(face_alignment.LandmarksType, "TWO_D", None) \
            or getattr(face_alignment.LandmarksType, "_2D")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _FACE_ALIGNER = face_alignment.FaceAlignment(
            lm_type, face_detector="folder", flip_input=False, device=device)
    return _FACE_ALIGNER


def _tensor_to_pil_frames(clip):
    """(3, T, H, W) float [0,1] RGB -> list of T PIL RGB images."""
    import numpy as np
    from PIL import Image
    frames = []
    for t in range(clip.shape[1]):
        frame = clip[:, t].permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0)
        frames.append(Image.fromarray(frame.cpu().numpy().astype(np.uint8), "RGB"))
    return frames


def _extract_landmarks(pil_frames):
    """FAN 68-point landmarks per frame; the crop is the full detection box."""
    import numpy as np
    fa = _get_face_aligner()
    landmarks = []
    for frame in pil_frames:
        det = fa.get_landmarks_from_image(np.asarray(frame),
                                          detected_faces=[(0, 0, frame.width, frame.height)])
        landmarks.append(np.asarray(det[0], dtype=np.float64) if det else None)
    return landmarks


def _similarity_matrix(src, dst):
    """Least-squares similarity transform src -> dst (skimage umeyama, scale=True)."""
    import numpy as np
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = dst_demean.T @ src_demean / src.shape[0]
    u, s, vt = np.linalg.svd(a)
    d = np.ones(2)
    if np.linalg.det(a) < 0:
        d[1] = -1
    rot = u @ np.diag(d) @ vt
    scale = (s @ d) / src_demean.var(axis=0).sum()
    m = np.eye(3)
    m[:2, :2] = rot * scale
    m[:2, 2] = dst_mean - scale * (rot @ src_mean)
    return m


def _warp_frame(pil_frame, transform):
    """Affine-warp a frame onto the 256x256 mean-face canvas (cv2 warpAffine)."""
    import cv2
    import numpy as np
    from PIL import Image
    arr = np.asarray(pil_frame, dtype=np.uint8)
    warped = cv2.warpAffine(arr, transform[:2], (_STD_SIZE, _STD_SIZE), flags=cv2.INTER_LINEAR)
    return Image.fromarray(warped, "RGB")


def _transform_landmarks(landmarks, transform):
    import numpy as np
    pts = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
    return (transform @ pts.T).T[:, :2]


def _cut_patch(img, landmarks, half_h, half_w):
    """96x96 box centred on the mean mouth landmark, clamped to the frame."""
    import numpy as np
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - half_h < 0:
        center_y = half_h
    if center_x - half_w < 0:
        center_x = half_w
    if center_y + half_h > img.shape[0]:
        center_y = img.shape[0] - half_h
    if center_x + half_w > img.shape[1]:
        center_x = img.shape[1] - half_w
    y0 = int(round(center_y) - half_h)
    x0 = int(round(center_x) - half_w)
    return np.ascontiguousarray(img[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w])


def _align_and_crop_clip(pil_frames, landmarks, mean_face):
    """Port of ``crop_video_and_save``: 12-frame smoothed landmarks, warp to
    the LRW mean face, crop the mouth.  Frames beyond the last full window
    reuse the last window's transform and are cropped on their own
    (un-smoothed) transformed landmarks, exactly as the official queue does."""
    import numpy as np
    num_frames = len(pil_frames)
    window = _SMOOTH_WINDOW
    last_transform = None
    crops = []
    for i in range(num_frames):
        if i <= num_frames - window:
            smoothed = np.mean(landmarks[i:i + window], axis=0)
            transform = _similarity_matrix(smoothed[_STABLE_POINTS], mean_face[_STABLE_POINTS])
            last_transform = transform
        else:
            transform = last_transform
        frame = _warp_frame(pil_frames[i], transform)
        trans_lms = _transform_landmarks(landmarks[i], transform)
        crop = _cut_patch(np.asarray(frame), trans_lms[_MOUTH_START:_MOUTH_END],
                          _MOUTH_SIZE // 2, _MOUTH_SIZE // 2)
        crops.append(crop)
    return crops


def _center_crop_fallback(pil_frames):
    """Fallback when FAN returns no landmarks: centre 96x96 of the face crop."""
    import numpy as np
    crops = []
    for frame in pil_frames:
        arr = np.asarray(frame)
        h, w = arr.shape[:2]
        y0, x0 = (h - _MOUTH_SIZE) // 2, (w - _MOUTH_SIZE) // 2
        crops.append(np.ascontiguousarray(arr[y0:y0 + _MOUTH_SIZE, x0:x0 + _MOUTH_SIZE]))
    return crops


def _crops_to_clip_tensor(crops, device):
    """(T,96,96) uint8 crops -> (1, 1, T, 88, 88) float, grayscale-normalized
    with the official (0.421, 0.165) mean/std (CenterCrop(88) first)."""
    import numpy as np
    import torch
    from PIL import Image
    gray = np.stack([np.asarray(Image.fromarray(c).convert("L"), dtype=np.float32) for c in crops])
    clip = torch.from_numpy(gray).div_(255.0).sub_(_GRAY_MEAN).div_(_GRAY_STD)
    return clip[:, 4:92, 4:92].unsqueeze(0).unsqueeze(0).to(device)


def build_model(weights_path, device):
    import torch
    arch = _get_arch()
    model = arch["LipForensicsDetector"]()
    if weights_path is not None:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        model.network.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
