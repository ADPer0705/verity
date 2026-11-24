# SBI weights

| Property | Value |
|---|---|
| File | `FFc23.tar` (default) |
| Size | 141,290,933 bytes |
| sha256 | `2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267` |
| Origin | Google Drive id `1X0-NYT8KPursLZZdxduRQju6E52hauV0` (README of mapooon/SelfBlendedImages) |
| Alternate | `FFraw.tar`, Google Drive id `12sLyqBp0VFwdpA-oZLdIOkOTkz_ZnIhV` (trained on FF++ raw) |
| Training | FF++ c23 real videos + self-blended images (no deepfake data) |
| License | "freely available for research purpose"; commercial use needs a license agreement; the LICENSE forbids redistribution |

The file is **not tracked in git** and is never mirrored by Verity. Download:

```bash
cd detectors/sbi/weights
curl -L -o FFc23.tar \
  'https://drive.usercontent.google.com/download?id=1X0-NYT8KPursLZZdxduRQju6E52hauV0&export=download&confirm=t'
```

Checkpoint layout: `{"model": state_dict, "optimizer": ..., "epoch": ...}`;
the state dict has 706 `net.*` tensors (EfficientNet-B4 + `net._fc` 1792->2)
and is loaded strictly.

```bibtex
@inproceedings{shiohara2022detecting,
  title={Detecting Deepfakes with Self-Blended Images},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={CVPR},
  year={2022}
}
```
