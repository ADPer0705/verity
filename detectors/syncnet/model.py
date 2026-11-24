"""SyncNet audio-visual synchronisation (Chung & Zisserman, "Out of time:
automated lip sync in the wild", ACCV 2016 workshop) as a lip-sync forensic
measurement.

Ported from ``SyncNetModel.py`` + ``SyncNetInstance.py`` + ``run_pipeline.py``
of github.com/joonson/syncnet_python (MIT); checkpoint ``syncnet_v2.model``
from the VGG group's download script.

The network embeds 5-frame mouth-region clips (3D conv tower on 224x224 BGR
face crops, raw 0-255 values as upstream) and 0.2 s MFCC windows (13 x 20,
python_speech_features at 16 kHz) into a joint 1024-d space.  Upstream
``evaluate`` slides the audio against the video by +-``vshift`` frames
(15) and reports the **AV offset** (argmin of the mean distance), the
**minimum distance** and the **confidence** (median distance minus minimum,
the LSE-C metric of the Wav2Lip literature).  Real talking-head footage has
high confidence (roughly 6-10 on LRS2/LRW) and an offset near 0; dubbed,
re-synthesised or badly lip-synced mouths have low confidence (below ~3-4)
and arbitrary offsets.

This detector is a *measurement*, not a trained real/fake classifier.
Verity converts the confidence of each window into a fake probability with
a fixed, documented logistic rule (``CONF_MID`` = 4.0, ``CONF_SCALE`` = 1.0:
p_fake = sigmoid((4 - confidence))), so confidence 4 maps to 0.5, 6 to 0.12
and 2 to 0.88.  The raw offset / min-distance / confidence / framewise
confidence are stored per window (``info``) so an investigator can read the
primary evidence instead of the derived score.

Per window (``input_mode: av_clip``) the model: detects the face with the
engine's MTCNN on every ``DETECT_EVERY``-th frame (linear interpolation in
between; upstream tracks with S3FD + IoU and median-filters the boxes),
crops it like ``run_pipeline.crop_video`` (half-size ``bs`` = max(w, h) / 2,
``crop_scale`` 0.4: rows ``cy - bs .. cy + bs * 1.8``, cols
``cx - bs * 1.4 .. cx + bs * 1.4``) and resizes to 224x224; re-samples the
frames to 25 fps by nearest timestamp (upstream re-encodes the video at
25 fps); computes MFCCs of the window's 16 kHz audio; runs the two towers in
batches and evaluates the offset/confidence exactly as upstream.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VSHIFT = 15
TARGET_FPS = 25.0
CROP_SCALE = 0.40
CROP_SIZE = 224
DETECT_EVERY = 5
CONF_MID = 4.0
CONF_SCALE = 1.0
BATCH = 20


class S(nn.Module):
    """Verbatim ``SyncNetModel.S``."""

    def __init__(self, num_layers_in_fc_layers=1024):
        super().__init__()
        self.netcnnaud = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),
            nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.netfcaud = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netfclip = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, num_layers_in_fc_layers),
        )
        self.netcnnlip = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        return self.netfcaud(mid.view(mid.size(0), -1))

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        return self.netfclip(mid.view(mid.size(0), -1))


def calc_pdist(feat1, feat2, vshift=VSHIFT):
    win_size = vshift * 2 + 1
    feat2p = F.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(F.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]))
    return dists


def _interp_boxes(boxes: dict, n: int) -> list[tuple[float, float, float]]:
    """(cx, cy, half-size) per frame from sparse detections (nearest /
    linear interpolation, as a cheap stand-in for upstream's tracker)."""
    idx = sorted(boxes)
    out = []
    for i in range(n):
        if i in boxes:
            out.append(boxes[i])
            continue
        prev = max((j for j in idx if j < i), default=None)
        nxt = min((j for j in idx if j > i), default=None)
        if prev is None:
            out.append(boxes[nxt])
        elif nxt is None:
            out.append(boxes[prev])
        else:
            t = (i - prev) / float(nxt - prev)
            out.append(tuple(a + (b - a) * t for a, b in zip(boxes[prev], boxes[nxt])))
    return out


class SyncNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = S(num_layers_in_fc_layers=1024)

    # -- preprocessing -------------------------------------------------------
    def _face_track(self, frames, face_detector):
        from PIL import Image
        det = {}
        for i in range(0, len(frames), DETECT_EVERY):
            boxes = face_detector.detect(Image.fromarray(frames[i][:, :, ::-1])) if face_detector else []
            if boxes:
                x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
                det[i] = (x + w / 2.0, y + h / 2.0, max(w, h) / 2.0)
        if not det:
            return None
        return _interp_boxes(det, len(frames))

    @staticmethod
    def _crop(frame, cx, cy, bs):
        import cv2
        bsi = int(bs * (1 + 2 * CROP_SCALE))
        padded = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
        my, mx = cy + bsi, cx + bsi
        face = padded[int(my - bs):int(my + bs * (1 + 2 * CROP_SCALE)),
                      int(mx - bs * (1 + CROP_SCALE)):int(mx + bs * (1 + CROP_SCALE))]
        return cv2.resize(face, (CROP_SIZE, CROP_SIZE))

    def forward_detailed(self, window):
        import python_speech_features

        frames = window["frames"]
        fps = float(window["fps"])
        sr = int(window["sample_rate"])
        audio = np.asarray(window["audio"], dtype="float32")
        device = next(self.parameters()).device

        # 25 fps re-sampling by nearest timestamp
        n_src = len(frames)
        duration = n_src / fps
        n_out = max(5, int(math.floor(duration * TARGET_FPS)))
        src_idx = [min(n_src - 1, int(round(k / TARGET_FPS * fps))) for k in range(n_out)]
        frames25 = [frames[i] for i in src_idx]

        track = self._face_track(frames25, window.get("face_detector"))
        h_im, w_im = frames25[0].shape[:2]
        if track is None:
            logits = torch.tensor([[0.0, 0.0]], device=device)
            return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                    "info": {"error": "no face found in the window", "confidence": None}}
        crops = np.stack([self._crop(f, *b) for f, b in zip(frames25, track)], axis=3)   # (224, 224, 3, T) BGR
        im = np.transpose(crops[None], (0, 3, 4, 1, 2))                                   # (1, 3, T, 224, 224)
        imtv = torch.from_numpy(im.astype("float32"))

        # MFCC (python_speech_features defaults: 25 ms window, 10 ms step, 13 coefficients)
        audio_i16 = np.clip(audio * 32768.0, -32768, 32767).astype("int16")
        mfcc = python_speech_features.mfcc(audio_i16, sr)                # (n, 13)
        cct = torch.from_numpy(np.ascontiguousarray(mfcc.T)[None, None].astype("float32"))  # (1, 1, 13, n)

        min_length = min(n_out, int(math.floor(len(audio) / (sr / TARGET_FPS))))
        lastframe = min_length - 5
        if lastframe < 1:
            logits = torch.tensor([[0.0, 0.0]], device=device)
            return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                    "info": {"error": "window too short for SyncNet", "confidence": None}}

        im_feat, cc_feat = [], []
        for i in range(0, lastframe, BATCH):
            im_batch = [imtv[:, :, v:v + 5, :, :] for v in range(i, min(lastframe, i + BATCH))]
            im_feat.append(self.net.forward_lip(torch.cat(im_batch, 0).to(device)).cpu())
            cc_batch = [cct[:, :, :, v * 4:v * 4 + 20] for v in range(i, min(lastframe, i + BATCH))]
            cc_feat.append(self.net.forward_aud(torch.cat(cc_batch, 0).to(device)).cpu())
        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = calc_pdist(im_feat, cc_feat, vshift=VSHIFT)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        minval, minidx = torch.min(mdist, 0)
        offset = int(VSHIFT - minidx.item())
        conf = float(torch.median(mdist) - minval)
        fdist = np.stack([d[minidx].numpy() for d in dists])
        fconf = (torch.median(mdist).numpy() - fdist).astype("float64")

        fake_logit = (CONF_MID - conf) / CONF_SCALE
        logits = torch.tensor([[0.0, fake_logit]], dtype=torch.float32, device=device)
        cx, cy, bs = track[0]
        bbox = [int(max(0, cx - bs)), int(max(0, cy - bs)), int(2 * bs), int(2 * bs)]
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "bbox": bbox,
            "info": {
                "av_offset_frames": offset,
                "av_offset_seconds": round(offset / TARGET_FPS, 3),
                "min_distance": round(float(minval), 4),
                "confidence": round(conf, 4),
                "framewise_confidence_mean": round(float(np.mean(fconf)), 4),
                "framewise_confidence_min": round(float(np.min(fconf)), 4),
                "frames_scored": int(lastframe),
                "calibration": f"p_fake = sigmoid(({CONF_MID} - confidence) / {CONF_SCALE})",
            },
        }

    def forward(self, window):
        return self.forward_detailed(window)["logits"]


def build_model(weights_path, device):
    model = SyncNetDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.net.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
