# Xception Weights

| Property | Value |
|---|---|
| File | `xception_best.pth` |
| Size | 87,763,531 bytes |
| Origin | Official DeepfakeBench v1.0.1 release (`SCLBD/DeepfakeBench`) |
| Training | FaceForensics++ (c23) |
| License | Follows DeepfakeBench project terms; research use only |

This file is **not tracked in git**. Download it with:

```bash
verity weights pull --detector xception
# or directly:
curl -L -o detectors/xception/weights/xception_best.pth \
  'https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth'
```

The checkpoint is the full detector state dict (keys `backbone.*`); the model
loads it with `strict=False` and no ImageNet backbone is needed separately.
