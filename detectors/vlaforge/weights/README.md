# VLAForge weights

| Property | Value |
|---|---|
| File | `vlaforge.pth` |
| Size | 965,834,563 bytes |
| sha256 | `2bc0c1e33d08a241386f88e3a9bc0ee4a504def129be0b4a9197b33f6feaff1f` |
| Origin | Google Drive id `1_PhtirGA_vMZWqnpZHGHHinT6QKF1d4L` (README of mala-lab/VLAForge, "Pre-train Models") |
| Training | FaceForensics++ c23 |
| License | MIT (repository LICENSE) |

```bash
cd detectors/vlaforge/weights
curl -L -o vlaforge.pth \
  'https://drive.usercontent.google.com/download?id=1_PhtirGA_vMZWqnpZHGHHinT6QKF1d4L&export=download&confirm=t'
```

Flat state dict with 1,258 tensors: `clip_model.*` (frozen OpenAI CLIP
ViT-L/14, fp16), `adapter.*` (ForgePerceiver), `rec_attn_clip.*` (shares the
CLIP visual blocks), `masked_xray_post_process.*`, `clip_post_process.*`,
`trainable_layer`, `conv`, `classifier`. Loaded with `strict=True` after
casting to fp32.

```bibtex
@inproceedings{zhu2026dfd,
  title={Unleashing Vision-Language Semantics for Deepfake Video Detection},
  author={Zhu, Jiawen and Miao, Yunqi and Zhang, Xueyi and Deng, Jiankang and Pang, Guansong},
  booktitle={CVPR},
  year={2026}
}
```
