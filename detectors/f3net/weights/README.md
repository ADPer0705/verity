# F3Net Weights

| Property | Value |
|---|---|
| File | `f3net_best.pth` |
| Size | 90,398,839 bytes |
| Origin | Official DeepfakeBench v1.0.1 release |
| Training | FaceForensics++ (c23) |
| License | Follows DeepfakeBench project terms; research use only |

Download:

```bash
verity weights pull --detector f3net
```

The checkpoint is the full detector state dict; the model loads it with
`strict=False` and no separate ImageNet backbone is needed.
