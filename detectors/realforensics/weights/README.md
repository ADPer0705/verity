# RealForensics weights

| Property | Value |
|---|---|
| File | `realforensics_ff.pth` |
| Size | 86,220,575 bytes |
| sha256 | `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d` |
| Origin | Google Drive id `1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ` ("Cross-dataset generalisation" model in the README of ahaliassos/RealForensics) |
| Training | stage 1 BYOL-AV on LRW, stage 2 FaceForensics++ c23 (all four manipulations) |
| License | MIT (repository LICENSE, Copyright (c) 2022 Alexandros Haliassos) |

```bash
cd detectors/realforensics/weights
curl -L -o realforensics_ff.pth \
  'https://drive.usercontent.google.com/download?id=1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ&export=download&confirm=t'
```

Flat state dict: `backbone.*` (CSN-R101, pytorchvideo key names) and
`df_head.weight` (cosine linear, 1 x 2048); loaded with `strict=True`.
The README's cross-manipulation models (`realforensics_allbut*.pth`) and the
no-augmentation robustness model share this layout and are drop-in
alternates via `weights.file`.

The FAN landmark model (face-alignment, BSD-3-Clause) is downloaded by the
library on first use into the torch hub cache.

```bibtex
@inproceedings{haliassos2022leveraging,
  title={Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection},
  author={Haliassos, Alexandros and Mira, Rodrigo and Petridis, Stavros and Pantic, Maja},
  booktitle={CVPR},
  year={2022}
}
```
