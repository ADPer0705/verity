# XLS-R + SLS Detector

wav2vec 2.0 XLS-R-300M + Sensitive Layer Selection classifier for audio
deepfake detection (Zhang, Wen, Hu, ACM MM 2024). The official checkpoint is
run as ONNX through onnxruntime (no fairseq).

## Setup (uv-managed)

```bash
cd detectors/xlsr_sls
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` may contain audio files or videos (the soundtrack is
extracted with ffmpeg). The 16 kHz mono waveform is scored in fixed
64,600-sample windows (4.0375 s, hop 2 s); `timeline.png` shows the
per-window fake probability. Set `VERITY_ORT_FORCE_CPU=1` to force the CPU
execution provider.

## Weights

`weights/xlsr-sls.onnx` (1.36 GB) — see `weights/README.md`.
