# Community Forensics Detector

"Community Forensics: Using Thousands of Generators to Train Fake Image
Detectors" (Park & Owens, CVPR 2025, [arXiv 2411.04125](https://arxiv.org/abs/2411.04125)),
ported from the official repository
[JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics) (MIT).

The released model is `models.ViTClassifier`: a timm `vit_small_patch16_384`
(ViT-S/16, 384 px input, 21.8 M parameters) with its ImageNet head replaced by
`Linear(384, 1)`, trained end-to-end on the Community Forensics dataset
(fake images from ~4,800 generators + real images). The single logit is a BCE
fake score: `P(fake) = sigmoid(z)` (label 1 = fake). The Verity adapter emits
2-class logits `[0, z]` so `softmax[:, 1] == sigmoid(z)` exactly.

## Setup (uv-managed)

```bash
cd detectors/community_forensics
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (frames, attention
maps, timeline, `result.json`) are written to `OUTPUT_DIR`.

## Input mode

`input_mode: full_frame` — the detector was trained on whole images, not face
crops. Images are scored once; videos are scored on up to `max_frames` (32)
evenly spaced frames and aggregated. No face detection is involved, so the
detector also runs on face-less media.

## Preprocessing (matches `dataloader.py::get_transform(mode="test")`)

For `input_size 384`, `determine_resize_crop_sizes` gives `resize_size 440`,
`crop_size 384`:

1. `transforms.Resize(440)` — shorter side to 440 px, bilinear;
2. `transforms.CenterCrop(384)`;
3. `ToTensor_range(0, 1)` then `Normalize(mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225])` — **ImageNet** statistics (the official code
   does not use CLIP statistics; the backbone was initialised from timm's
   `augreg_in21k_ft_in1k` ViT-S/16-384, which uses the same normalisation).

`model.preprocess` implements exactly this; `detector.yaml` carries the same
numbers (`resize_short: 440`, `center_crop: true`, `resolution: 384`) for the
orchestrator.

## Weights

`weights/model.safetensors` (87,262,324 bytes, MIT) is the official
`PyTorchModelHubMixin` export of the trained `ViTClassifier` from
[`OwensLab/commfor-model-384`](https://huggingface.co/OwensLab/commfor-model-384)
(`config.json`: `model_size: small, input_size: 384, patch_size: 16`). All 152
tensors are prefixed `vit.` and are loaded strictly into the same timm
architecture (`pretrained=False`); the ImageNet-21k init used by the official
class is just the training starting point and is entirely overwritten, so no
timm download happens at run time.

```bash
curl -L -o weights/model.safetensors \
  'https://huggingface.co/OwensLab/commfor-model-384/resolve/main/model.safetensors'
```

See `weights/README.md` for the checksum and citation.

## Explainability

Grad-CAM (`cam: true`) over the 24x24 patch-token grid entering the last
transformer block (the last block's attention routes those tokens into the
CLS token that feeds the head, so gradients are non-zero there), plus the
per-frame timeline for videos.
