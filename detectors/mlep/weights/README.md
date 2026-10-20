# MLEP Weights

| Property | Value |
|---|---|
| File | `model_epoch_best.pth` |
| Size | 5,853,801 bytes |
| Origin | Official MLEP repository (`fkeufss/MLEP`, NeurIPS 2025) |
| Training | ProGAN images (NPR training protocol); evaluated on 32 generators (GAN-set-1/2, Diffusion-set) |
| License | None specified in upstream repo; research use only |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o detectors/mlep/weights/model_epoch_best.pth \
  'https://raw.githubusercontent.com/fkeufss/MLEP/main/pretrained/model_epoch_best.pth'
```

The checkpoint is the raw model state dict (keys `conv1.*`, `layer1.*`,
`layer2.*`, `fc1.*`; no layer3/layer4 — the truncated ResNet50 head) plus
`_metadata`/`version` keys that are stripped before loading.  It is a single
BCE logit with label 1 = fake; `build_model` loads it with `strict=True`.
