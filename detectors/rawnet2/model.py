"""RawNet2 anti-spoofing (Tak, Patino, Todisco, Nautsch, Evans, Larcher,
ICASSP 2021, "End-to-end anti-spoofing with RawNet2") trained with RawBoost
augmentation (Tak, Kamble, Patino, Todisco, Evans, ICASSP 2022).

Ported from ``model.py`` + ``model_config_RawNet.yaml`` of
github.com/TakHemlata/RawBoost-antispoofing (MIT, EURECOM), the ASVspoof 2021
RawNet2 baseline.  Checkpoint ``pre_trained_model/Best_model.pth`` from the
same repository: RawNet2 trained on ASVspoof2019 LA train with RawBoost
(algo 5: linear+non-linear convolutive noise + impulsive signal-dependent
noise, the LA-track recipe of the paper).

Architecture (``filts [20, [20, 20], [20, 128], [128, 128]]``): a fixed
mel-spaced sinc filterbank (20 filters, kernel 1024; the band-pass filters are
computed once here instead of at every forward as upstream) -> |x| ->
max-pool(3) -> BN -> SELU -> 6 residual blocks with feature-map scaling
(FMS: per-channel sigmoid gate, ``x * y + y``) -> BN -> SELU -> 3-layer GRU
(1024) -> FC(1024, 1024) -> FC(1024, 2).

Input: 16 kHz mono waveform, exactly 64,600 samples (``data_utils_rawboost.pad``:
first 64,600 samples, shorter signals tile-repeated), no amplitude
normalisation; the engine windowing (``audio_window_seconds: 4.0375``)
reproduces this.  Output: upstream logits are [spoof, bonafide] (``genSpoof_list``:
bonafide = 1; ``produce_evaluation_file`` scores ``batch_out[:, 1]``), so
``forward_detailed`` swaps the columns to make index 1 = fake.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NB_SAMP = 64600
SAMPLE_RATE = 16000
D_ARGS = {
    "nb_samp": NB_SAMP,
    "first_conv": 1024,
    "in_channels": 1,
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
}


class SincConv(nn.Module):
    """Mel-spaced fixed sinc band-pass filterbank (no learnable parameters)."""

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=SAMPLE_RATE):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        nfft = 512
        f = int(sample_rate / 2) * np.linspace(0, 1, int(nfft / 2) + 1)
        fmel = self.to_mel(f)
        filbandwidthsmel = np.linspace(np.min(fmel), np.max(fmel), out_channels + 1)
        mel = self.to_hz(filbandwidthsmel)
        hsupp = np.arange(-(kernel_size - 1) / 2, (kernel_size - 1) / 2 + 1)
        band_pass = np.zeros((out_channels, kernel_size), dtype="float64")
        window = np.hamming(kernel_size)
        for i in range(len(mel) - 1):
            fmin, fmax = mel[i], mel[i + 1]
            h_high = (2 * fmax / sample_rate) * np.sinc(2 * fmax * hsupp / sample_rate)
            h_low = (2 * fmin / sample_rate) * np.sinc(2 * fmin * hsupp / sample_rate)
            band_pass[i, :] = window * (h_high - h_low)
        self.register_buffer("filters", torch.tensor(band_pass, dtype=torch.float32)
                             .view(out_channels, 1, kernel_size), persistent=False)

    def forward(self, x):
        return F.conv1d(x, self.filters, stride=1, padding=0, dilation=1, bias=None, groups=1)


class ResidualBlock(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(nb_filts[0], nb_filts[1], kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(nb_filts[1], nb_filts[1], padding=1, kernel_size=3, stride=1)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv1d(nb_filts[0], nb_filts[1], padding=0, kernel_size=1, stride=1)
        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        identity = x
        # Upstream applies bn1/lrelu to ``out`` but then convolves ``x`` (the
        # un-normalised input); reproduced verbatim so the checkpoint matches.
        if not self.first:
            _ = self.lrelu(self.bn1(x))
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out = out + identity
        return self.mp(out)


class RawNet2(nn.Module):
    def __init__(self, d_args=D_ARGS):
        super().__init__()
        filts = [list(f) if isinstance(f, list) else f for f in d_args["filts"]]
        self.Sinc_conv = SincConv(out_channels=filts[0], kernel_size=d_args["first_conv"])
        self.first_bn = nn.BatchNorm1d(num_features=filts[0])
        self.selu = nn.SELU(inplace=True)
        self.block0 = nn.Sequential(ResidualBlock(nb_filts=filts[1], first=True))
        self.block1 = nn.Sequential(ResidualBlock(nb_filts=filts[1]))
        self.block2 = nn.Sequential(ResidualBlock(nb_filts=filts[2]))
        filts[2][0] = filts[2][1]
        self.block3 = nn.Sequential(ResidualBlock(nb_filts=filts[2]))
        self.block4 = nn.Sequential(ResidualBlock(nb_filts=filts[2]))
        self.block5 = nn.Sequential(ResidualBlock(nb_filts=filts[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_attention0 = nn.Sequential(nn.Linear(filts[1][-1], filts[1][-1]))
        self.fc_attention1 = nn.Sequential(nn.Linear(filts[1][-1], filts[1][-1]))
        self.fc_attention2 = nn.Sequential(nn.Linear(filts[2][-1], filts[2][-1]))
        self.fc_attention3 = nn.Sequential(nn.Linear(filts[2][-1], filts[2][-1]))
        self.fc_attention4 = nn.Sequential(nn.Linear(filts[2][-1], filts[2][-1]))
        self.fc_attention5 = nn.Sequential(nn.Linear(filts[2][-1], filts[2][-1]))
        self.bn_before_gru = nn.BatchNorm1d(num_features=filts[2][-1])
        self.gru = nn.GRU(input_size=filts[2][-1], hidden_size=d_args["gru_node"],
                          num_layers=d_args["nb_gru_layer"], batch_first=True)
        self.fc1_gru = nn.Linear(d_args["gru_node"], d_args["nb_fc_node"])
        self.fc2_gru = nn.Linear(d_args["nb_fc_node"], d_args["nb_classes"], bias=True)
        self.sig = nn.Sigmoid()

    def _fms(self, x, block, fc):
        x = block(x)
        y = self.avgpool(x).view(x.size(0), -1)
        y = self.sig(fc(y)).view(y.size(0), y.size(1), -1)
        return x * y + y

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.selu(self.first_bn(x))
        x = self._fms(x, self.block0, self.fc_attention0)
        x = self._fms(x, self.block1, self.fc_attention1)
        x = self._fms(x, self.block2, self.fc_attention2)
        x = self._fms(x, self.block3, self.fc_attention3)
        x = self._fms(x, self.block4, self.fc_attention4)
        x = self._fms(x, self.block5, self.fc_attention5)
        x = self.selu(self.bn_before_gru(x))
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1_gru(x)
        return self.fc2_gru(x)


class RawNet2Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = RawNet2()

    @staticmethod
    def _fix_length(x: torch.Tensor) -> torch.Tensor:
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
        logits_upstream = self.net(x)              # [spoof, bonafide]
        logits = logits_upstream[:, [1, 0]]        # Verity: index 1 = fake
        return {"logits": logits, "probs": torch.softmax(logits, dim=1),
                "upstream_logits": logits_upstream}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = RawNet2Detector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        # Upstream SincConv keeps no parameters; everything else must match.
        model.net.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
