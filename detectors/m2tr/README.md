# M2TR Detector

"M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection" (Wang,
Wu, Chen, Jiang, ICMR 2022), ported from
github.com/wdrink/M2TR-Multi-modal-Multi-scale-Transformers-for-Deepfake-Detection
(MIT). On the low-level EfficientNet-B4 texture map the model runs
multi-scale patch self-attention (patch sizes 80/40/20/10), a learned global
filter in the frequency domain and a cross-modality attention block that
fuses frequency evidence back into the RGB stream, four times, before the
remaining EfficientNet stages classify. FF++ c23 accuracy 97.9 % (c40 model
92.9 %).

## Setup (managed by uv)

```bash
cd detectors/m2tr
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

320x320 MTCNN face crops with ImageNet normalisation.

## Weights

`weights/m2tr_ff_c23.pth` (official FF-C23 model-zoo checkpoint, MIT, Google
Drive); the FF-RAW and FF-C40 models are drop-in alternates. See
`weights/README.md`.

## Explainability

Per-face fake probability + Grad-CAM over the 1792-channel head features +
facial region energy + timeline + top frames.
