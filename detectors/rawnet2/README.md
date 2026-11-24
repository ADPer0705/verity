# RawNet2 (RawBoost) Detector

End-to-end anti-spoofing with RawNet2 (Tak, Patino, Todisco, Nautsch, Evans,
Larcher, ICASSP 2021) trained with RawBoost raw-waveform augmentation (Tak,
Kamble, Patino, Todisco, Evans, ICASSP 2022). Vendored from
github.com/TakHemlata/RawBoost-antispoofing (MIT, EURECOM); this is also the
anti-spoofing form of the "RawNet2" architecture (the Jungjee/RawNet
repository holds the speaker-verification variants, which are not spoof
detectors).

## Setup (uv-managed)

```bash
cd detectors/rawnet2
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` may contain audio files or videos (the soundtrack is
extracted with ffmpeg). The 16 kHz mono waveform is scored in fixed
64,600-sample windows (4.0375 s, hop 2 s, short signals tile-padded as
upstream); `timeline.png` shows the per-window fake probability.

## Weights

`weights/Best_model.pth` (ASVspoof2019 LA + RawBoost, MIT), see
`weights/README.md`. Benchmark-era baseline: pair it with the modern
self-supervised audio detectors (`xlsr_sls`, `antideepfake_mms`).
