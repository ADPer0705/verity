# I3D Weights (backbone only)

| Property | Value |
|---|---|
| File | `I3D_8x8_R50.pth` |
| Source | https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.3/I3D_8x8_R50.pth |
| Kind | Pretrained backbone (no detector-trained checkpoint released) |

This file only initializes the backbone; the classification head is untrained,
so predictions are unreliable. Download via:

```bash
verity weights pull --detector i3d
```
