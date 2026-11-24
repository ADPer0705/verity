# M2TR weights

| Property | Value |
|---|---|
| File | `m2tr_ff_c23.pth` (default) |
| Size | 224,488,781 bytes |
| sha256 | `4700e0392884fdeeb24689935bd66de4ae52b8cbeae955cb98766f784d8bd061` |
| Origin | Google Drive id `1XRIllA6p5YnITztl1burwcr5l7LAcpqv` (model zoo "FF++ (C23)" in the README of wdrink/M2TR) |
| Alternates | FF-RAW `1_HaPE6r7Zzof2mmLmmc4fbIbqyWs17S0`, FF-C40 `1xhclIjoh8GkVvoVefjDY-itdaV0VaMxY` |
| Training | FaceForensics++ c23 (Deepfakes) |
| License | MIT (repository LICENSE) |

```bash
cd detectors/m2tr/weights
curl -L -o m2tr_ff_c23.pth \
  'https://drive.usercontent.google.com/download?id=1XRIllA6p5YnITztl1burwcr5l7LAcpqv&export=download&confirm=t'
```

Layout: `{"epoch", "model_state", "optimizer_state", "scheduler_state",
"cfg"}`; `model_state` has 928 tensors (`model.*` EfficientNet-B4 incl. an
unused `_fc`, `layers.{0-3}.{0,1,2}.*`, `classifier.projection.*`), loaded
strictly.

```bibtex
@inproceedings{wang2022m2tr,
  title={M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection},
  author={Wang, Junke and Wu, Zuxuan and Chen, Jingjing and Jiang, Yu-Gang},
  booktitle={ICMR},
  year={2022}
}
```
