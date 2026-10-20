# NPR Detector

NPR ("Rethinking the Up-Sampling Operations in CNN-based Generative Network for
Generalizable Deepfake Detection", Tan et al., CVPR 2024), ported from
`chuangchuangtan/NPR-DeepfakeDetection`.

The model is a truncated ResNet-50 (conv1 + layer1/layer2 only) that receives
the no-reference pixel-level representation
`NPR = x - upsample2x(downsample2x(x))` (nearest-neighbour) scaled by 2/3, with
a single-logit BCE head. `sigmoid(logit)` is the official fake probability
(label 1 = fake in the `0_real`/`1_fake` dataset layout and `validate.py`).

## Setup (uv-managed)

```bash
cd detectors/npr
uv sync
```


## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, attention
maps, timeline, top frames, `result.json`) are written to `OUTPUT_DIR`.

## Weights

Place `NPR.pth` in `weights/` (see `weights/README.md`). Without it the model
is randomly initialized and scores are not meaningful.

```bash
curl -L -o weights/NPR.pth \
  'https://raw.githubusercontent.com/chuangchuangtan/NPR-DeepfakeDetection/main/NPR.pth'
```

## Porting notes

- **Preprocessing**: mirrors the official test pipeline
  (`data/datasets.py`): bilinear resize to 256x256, no center crop, then
  ImageNet mean/std normalization. `detector.yaml` carries those stats
  (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`, resolution
  256), and the engine's normalization matches directly — the model does not
  normalize internally.
- **Class mapping**: the head outputs a single BCE logit whose sigmoid is
  P(fake). `forward_detailed` emits 2-class logits `[0, logit]` so that
  `softmax(logits)[1] == sigmoid(logit)` — index 1 = fake per the engine
  contract.
- **Checkpoint**: the committed `NPR.pth` wraps the state dict under a
  `model` key (alongside `optimizer`/`total_steps`, see
  `networks/base_model.py`); `build_model` unwraps `state_dict`/`model`/`net`
  keys and strips `module.` prefixes, and also accepts a bare state dict
  (the `test.py` form).

## Explainability

Per-face fake probability + Grad-CAM attention overlays (on `layer2`
features) + facial region energy analysis + prediction timeline +
most-suspicious frames.
