# ProDet Detector

"Can We Leave Deepfake Data Behind in Training Deepfake Detector?" (Cheng,
Yan, Zhang, Luo, Wang, Li, NeurIPS 2024). Port of the standalone inference
model in github.com/beautyremain/ProDet (`demo/prodet_inference.py`).

ProDet trains an EfficientNet-B4 on a *progression* of supervision: real
faces, blend-only pseudo-fakes, identity-inconsistent pseudo-fakes and real
deepfakes. At inference three auxiliary classifiers (deepfake / blend /
identity inconsistency) are inverted into spatial maps that gate the feature
map before the final classifier, so the model reports both a verdict and the
kind of artifact it is reacting to.

## Setup (managed by uv)

```bash
cd detectors/prodet
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

256x256 MTCNN face crops, mean/std 0.5 (DeepfakeBench convention).

## Weights

`weights/ProDet_best.pth` (official, 386 MB, Google Drive, no license
stated); see `weights/README.md`.

## Explainability

Per-face fake probability, the model's own feature-attention mask as the
attention map (Grad-CAM fallback), auxiliary deepfake / blend / identity
probabilities, facial region energy, timeline and top frames.
