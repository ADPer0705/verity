# NPR Weights

| Property | Value |
|---|---|
| File | `NPR.pth` |
| Size | 17,393,733 bytes |
| Origin | Official repository `chuangchuangtan/NPR-DeepfakeDetection` (main branch) |
| Paper | "Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection" (CVPR 2024) |
| Training | ForenSynths (ProGAN) real images vs. generated images |
| License | Follows the official repository terms; research use only |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o weights/NPR.pth \
  'https://raw.githubusercontent.com/chuangchuangtan/NPR-DeepfakeDetection/main/NPR.pth'
```

SHA-256 of the official file: `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`

The checkpoint is a wrapped dict `{model, optimizer, total_steps}` (Adam
optimizer state included); `build_model` loads the state dict under the
`model` key. A bare state dict export (as loaded by the repo's `test.py`)
also works.
