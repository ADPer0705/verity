# EfficientNet-B4 Weights

| Property | Value |
|---|---|
| File | `effnb4_best.pth` |
| Size | 70,979,261 bytes |
| Origin | Official DeepfakeBench v1.0.1 release |
| Training | FaceForensics++ (c23) |
| License | Follows DeepfakeBench project terms; research use only |

Download:

```bash
verity weights pull --detector efficientnetb4
```

The checkpoint is the full detector state dict; the model loads it with
`strict=False` and no separate ImageNet backbone is needed.
