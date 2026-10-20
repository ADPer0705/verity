# IID Weights (backbone only)

| Property | Value |
|---|---|
| File | `backbone.pth` |
| Source | https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip |
| Kind | Pretrained backbone (no detector-trained checkpoint released) |

This file only initializes the backbone; the classification head is untrained,
so predictions are unreliable. Download via:

```bash
verity weights pull --detector iid
```
