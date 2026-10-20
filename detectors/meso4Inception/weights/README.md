# MesoInception Weights

| Property | Value |
|---|---|
| File | `meso4Incep_best.pth` |
| Size | 126,419 bytes |
| Origin | Official DeepfakeBench v1.0.1 release |
| Training | FaceForensics++ (c23) |
| License | Follows DeepfakeBench project terms; research use only |

Download:

```bash
verity weights pull --detector meso4Inception
```

The checkpoint is the full detector state dict; the model loads it with
`strict=False` and no separate ImageNet backbone is needed.
