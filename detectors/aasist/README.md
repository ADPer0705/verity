# AASIST Detector

Raw-waveform audio anti-spoofing with spectro-temporal graph attention
(Jung et al., ICASSP 2022), vendored from github.com/clovaai/aasist (MIT).

## Setup (uv-managed)

```bash
cd detectors/aasist
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

`weights/AASIST.pth` (ASVspoof2019 LA trained, MIT) — see `weights/README.md`.
This is a 2019 baseline that generalises poorly to 2024 in-the-wild audio.
