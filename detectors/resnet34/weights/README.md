# CNN-Aug (ResNet34) Weights

| Property | Value |
|---|---|
| File | `cnnaug_best.pth` |
| Size | 85,285,901 bytes |
| Origin | Official DeepfakeBench v1.0.1 release |
| Training | FaceForensics++ (c23) |
| License | Follows DeepfakeBench project terms; research use only |

Download:

```bash
verity weights pull --detector resnet34
```

The checkpoint is the full detector state dict; the model loads it with
`strict=False` and no separate ImageNet backbone is needed.
