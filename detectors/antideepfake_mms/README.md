# AntiDeepfake MMS-300M Detector

Post-trained MMS-300M (wav2vec 2.0) + FC head for deepfake-speech detection
(NII Yamagishi Lab, ASRU 2025, arXiv 2506.21090). Ported from
github.com/nii-yamagishilab/AntiDeepfake without fairseq: the fairseq
wav2vec2 state dict is remapped onto `transformers.Wav2Vec2Model`.

## Setup (uv-managed)

```bash
cd detectors/antideepfake_mms
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` may contain audio files or videos (the engine extracts the
soundtrack with ffmpeg). The 16 kHz mono waveform is scored in 4 s windows
(hop 2 s); `timeline.png` shows the per-window fake probability over a
spectrogram and `result.json` the aggregate.

## Weights

Fetched from HuggingFace (`nii-yamagishilab/mms-300m-anti-deepfake`,
CC BY-NC-SA 4.0) at first run; see `weights/README.md`.
