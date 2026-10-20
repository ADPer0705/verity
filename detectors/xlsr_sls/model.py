"""XLS-R + SLS audio deepfake detector (Zhang, Wen, Hu, ACM MM 2024:
"Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier").

wav2vec 2.0 XLS-R-300M front-end whose 24 transformer-layer outputs are
pooled, gated by a learned per-layer sigmoid attention (Sensitive Layer
Selection), summed, then BatchNorm + SELU + max-pool + FC(22847, 1024) +
FC(1024, 2) + LogSoftmax.  Upstream code: github.com/QiShanZhang/
SLSforASVspoof-2021-DF (fairseq-based; no license file).  The official
checkpoint (``model_15``, ASVspoof2019 LA trained) is re-hosted by
SpeechAntiSpoofingBenchmarks on the HF hub as ``xlsr-sls.onnx`` (opset 17,
input ``wav`` (batch, 64600) float32, output ``logits`` (batch, 2)), which this
port runs with onnxruntime -- no fairseq and no ``xlsr2_300m.pt`` needed.

Input: raw 16 kHz mono waveform, exactly 64,600 samples (the head's fc1 is
sized for that); upstream ``data_utils_SSL.pad`` takes the first 64,600
samples and tile-repeats shorter utterances (no amplitude normalisation),
which the engine's windowing (``audio_window_seconds: 4.0375``) reproduces.

Output: upstream log-softmax over [spoof, bonafide] (``main.py``:
``batch_score = batch_out[:, 1]`` = bonafide; labels bonafide = 1).  Verity
wants softmax index 1 = fake, so ``forward_detailed`` returns the columns
swapped: ``logits[:, [1, 0]]`` (softmax of a log-softmax is the same
distribution, so softmax[:, 1] is exactly the upstream P(spoof)).

The ONNX session is wrapped in a ``torch.nn.Module`` so the engine's
``.to(device)`` / ``.eval()`` calls are harmless; the execution provider is
chosen at build time (CUDA when onnxruntime-gpu can see a GPU, else CPU).
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

NB_SAMP = 64600
INPUT_NAME = "wav"
OUTPUT_NAME = "logits"


class XLSRSLSOnnx(nn.Module):
    def __init__(self, onnx_path: str, device):
        super().__init__()
        import onnxruntime as ort

        try:  # onnxruntime-gpu >= 1.21: load CUDA/cuDNN from the pip nvidia-* packages
            ort.preload_dlls()
        except Exception:
            pass
        want_cuda = str(device).startswith("cuda") and os.environ.get("VERITY_ORT_FORCE_CPU", "") != "1"
        available = ort.get_available_providers()
        providers = []
        if want_cuda and "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.provider = self.session.get_providers()[0]
        ins = [i.name for i in self.session.get_inputs()]
        outs = [o.name for o in self.session.get_outputs()]
        self.input_name = INPUT_NAME if INPUT_NAME in ins else ins[0]
        self.output_name = OUTPUT_NAME if OUTPUT_NAME in outs else outs[0]
        self._device = torch.device(device)
        # Dummy parameter so next(model.parameters()) and .to() behave.
        self._marker = nn.Parameter(torch.zeros(1), requires_grad=False)

    @staticmethod
    def _fix_length(x: np.ndarray) -> np.ndarray:
        n = x.shape[1]
        if n == NB_SAMP:
            return x
        if n > NB_SAMP:
            return x[:, :NB_SAMP]
        reps = NB_SAMP // n + 1
        return np.tile(x, (1, reps))[:, :NB_SAMP]

    def forward_detailed(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        arr = np.ascontiguousarray(x.detach().cpu().float().numpy(), dtype=np.float32)
        arr = self._fix_length(arr)
        out = self.session.run([self.output_name], {self.input_name: arr})[0]   # log-softmax [spoof, bonafide]
        upstream = torch.from_numpy(np.asarray(out, dtype=np.float32)).to(x.device)
        logits = upstream[:, [1, 0]]                                             # Verity: index 1 = fake
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "upstream_logits": upstream}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"XLSR-SLS ONNX checkpoint not found at {weights_path}; see weights/README.md")
    model = XLSRSLSOnnx(weights_path, device)
    model.eval()
    print(f"  xlsr_sls: onnxruntime provider = {model.provider}", flush=True)
    return model
