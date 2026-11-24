# D³ weights

| Property | Value |
|---|---|
| File | `classifier.pth` |
| Size | 12,606,728 bytes |
| sha256 | `c0bb66e0de538b0330613da1865da025756c1dc7a7290990d5343c35290ab63b` |
| Origin | `ckpt/classifier.pth` in github.com/BigAandSmallq/D3 |
| Content | attention head only: `query/key/value` Linear(1024, 1024) + `fc` Linear(2048, 1) |
| License | the repository ships no LICENSE file (all rights reserved by default); fetched from the authors' repo only |

```bash
curl -L -o detectors/d3/weights/classifier.pth \
  'https://raw.githubusercontent.com/BigAandSmallq/D3/main/ckpt/classifier.pth'
```

The frozen CLIP ViT-L/14 tower is `openai/clip-vit-large-patch14` (MIT),
fetched by transformers into the shared HF cache.

```bibtex
@inproceedings{yang2025d3,
  title={D3: Scaling Up Deepfake Detection by Learning from Discrepancy},
  author={Yang, Yongqi and Qian, Zhihao and Zhu, Ye and Russakovsky, Olga and Wu, Yu},
  booktitle={CVPR},
  year={2025}
}
```
