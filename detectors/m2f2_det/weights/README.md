# M2F2-Det weights

| Property | Value |
|---|---|
| File | `m2f2_det_stage1.pth` |
| Size | 1,990,597,085 bytes |
| sha256 | `ac1cdb94fe341d1193c573d71d683d5786fc2f0dc27a7494bb0e27cd1cdab208` |
| Origin | Google Drive id `1X1ZUZkCwqg9mrsqoOS0EoO3v5WABNBAw` ("M2F2-Det's detector-only weights" in the README of CHELSEA234/M2F2_Det) |
| Training | FaceForensics++ (c23/c40 HDF5 release of the authors) |
| License | MIT (repository LICENSE) |

```bash
cd detectors/m2f2_det/weights
curl -L -o m2f2_det_stage1.pth \
  'https://drive.usercontent.google.com/download?id=1X1ZUZkCwqg9mrsqoOS0EoO3v5WABNBAw&export=download&confirm=t'
```

Layout: `{"epoch", "model_state_dict", "optimizer_state_dict", "iter",
"scheduler", "val_loss"}`; `model_state_dict` has 1,324 `module.*` tensors:
`clip_vision_encoder.model.*` (CLIP ViT-L/14-336 vision tower, identical to
the separately offered `vision_tower.pth`), `clip_text_encoder.*` (text
tower + 7 prompt tokens, unused by the head), `deepfake_encoder.*`
(torchvision EfficientNet-B4 features), `deepfake_proj`, `vision_proj`,
`text_proj`, `linear_1/2/3`, `clip_reduction`, `bridge_adapter_proj`,
`clip_vision_alpha`, `clip_text_alpha`, `output`. The bridge-adapter blocks
themselves are **absent** (see the detector README caveat).

```bibtex
@inproceedings{M2F2_Det_xiao,
  author = {Xiao Guo and Xiufeng Song and Yue Zhang and Xiaohong Liu and Xiaoming Liu},
  title = {Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector},
  booktitle = {CVPR},
  year = {2025},
}
```
