# LipForensics Weights

## lipforensics_ff.pth (required)

| Property | Value |
|---|---|
| File | `lipforensics_ff.pth` |
| Source | Google Drive (official "Lips Don't Lie" release) |
| Kind | Official FF++-trained detector checkpoint |
| Contents | `{"model": <Lipreading state dict>}` (single-file; no separate backbone) |

Download:

```bash
curl -L -o weights/lipforensics_ff.pth 'https://drive.usercontent.google.com/download?id=1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7&export=download&confirm=t'
```

## FAN landmark weights (auto-downloaded, ~150MB)

Face-alignment's FAN-68 model (`face_detector="none"`) downloads its
weights on first use into the torch hub cache
(`~/.cache/torch/hub/checkpoints/face_alignment_net.pth`, ~150MB) and
reuses them afterwards. No manual download needed.
