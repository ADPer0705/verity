# Forensics Adapter weights

| Property | Value |
|---|---|
| File | `forensics_adapter.pth` |
| Size | 962,621,690 bytes |
| sha256 | `3d87f9bfabb2a4ab3ad5fa0c2bf43bd0b41377b7546fdad0a9b61a1d4adfbddb` |
| Origin | Google Drive id `1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm` ("Model Weights" in the README of OUC-VAS/ForensicsAdapter) |
| Training | FaceForensics++ c23 |
| License | CC BY-NC 4.0 (repository LICENSE; academic research only) |

```bash
cd detectors/forensics_adapter/weights
curl -L -o forensics_adapter.pth \
  'https://drive.usercontent.google.com/download?id=1UlaAUTtsX87ofIibf38TtfAKIsnA7WVm&export=download&confirm=t'
```

Flat state dict with 1,253 tensors: `clip_model.*` (frozen OpenAI CLIP
ViT-L/14, fp16), `adapter.*`, `rec_attn_clip.*` (shares the CLIP visual
blocks), `masked_xray_post_process.*`, `clip_post_process.*`. Loaded with
`strict=True` after casting to fp32.

```bibtex
@InProceedings{Cui_2025_CVPR,
  author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
  title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
  booktitle = {CVPR},
  year      = {2025}
}
```
