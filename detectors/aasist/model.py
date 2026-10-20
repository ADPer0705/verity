"""AASIST (Jung et al., ICASSP 2022): raw-waveform anti-spoofing with
sinc-conv front-end + residual encoder + heterogeneous stacking graph
attention (HS-GAL) and max graph operations.

``aasist_net.py`` is a verbatim copy of ``models/AASIST.py`` from
github.com/clovaai/aasist (MIT, NAVER Corp.); the config dict below is the
``model_config`` block of ``config/AASIST.conf``.  Checkpoint:
``models/weights/AASIST.pth`` from the same repo (ASVspoof2019 LA trained,
0.83 % EER on ASVspoof2019 LA eval).

Input: raw 16 kHz mono waveform, exactly 64,600 samples (~4.04 s); upstream
``data_utils.pad`` takes the first 64,600 samples and tile-repeats shorter
utterances, which the engine's audio windowing reproduces
(``audio_window_seconds: 4.0375``).  No amplitude normalisation upstream.

Output: upstream ``out_layer`` logits are ordered [spoof, bonafide]
(``data_utils.genSpoof_list``: label 1 = "bonafide", 0 = spoof; main.py
scores with ``batch_out[:, 1]`` = bonafide).  Verity wants index 1 = fake, so
``forward_detailed`` returns ``logits[:, [1, 0]]``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

NB_SAMP = 64600

# config/AASIST.conf -> model_config
AASIST_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": NB_SAMP,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


class AASISTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        from aasist_net import Model

        self.net = Model(AASIST_CONFIG)

    @staticmethod
    def _fix_length(x: torch.Tensor) -> torch.Tensor:
        """Upstream pad(): first NB_SAMP samples, tile-repeat if shorter."""
        n = x.shape[1]
        if n == NB_SAMP:
            return x
        if n > NB_SAMP:
            return x[:, :NB_SAMP]
        reps = NB_SAMP // n + 1
        return x.repeat(1, reps)[:, :NB_SAMP]

    def forward_detailed(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        x = self._fix_length(x.float())
        hidden, logits_upstream = self.net(x, Freq_aug=False)   # [spoof, bonafide]
        logits = logits_upstream[:, [1, 0]]                      # Verity: index 1 = fake
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "upstream_logits": logits_upstream, "feat": hidden}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = AASISTDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.net.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
