# AASIST Weights

| Property | Value |
|---|---|
| File | `AASIST.pth` |
| Size | 1,281,532 bytes |
| sha256 | `51d2d9cf0738172f61e2a384ec50a54a55363240f67c971ed55a92435bc1a1c0` |
| Origin | `models/weights/AASIST.pth` in github.com/clovaai/aasist (NAVER Corp.) |
| Training | ASVspoof2019 LA train (0.83 % EER / 0.0275 min t-DCF on ASVspoof2019 LA eval) |
| License | MIT (repo LICENSE, Copyright (c) 2021-present NAVER Corp.) |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o detectors/aasist/weights/AASIST.pth \
  'https://github.com/clovaai/aasist/raw/main/models/weights/AASIST.pth'
```

The checkpoint is the plain `Model` state dict (no `module.` prefix); it is
loaded with `strict=True`. Upstream logit order is [spoof, bonafide]; the
port swaps columns so Verity's index 1 = fake.

Caveat: AASIST is a 2019-era benchmark model. It does not generalise to modern
in-the-wild deepfake audio (Deepfake-Eval-2024 AUC 0.43, Chandra et al. 2025),
so it is shipped as a baseline reference, not an evidence-grade headline.

Citation:

```bibtex
@inproceedings{jung2022aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={Proc. ICASSP},
  year={2022}
}
```
