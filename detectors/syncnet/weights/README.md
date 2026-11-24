# SyncNet weights

| Property | Value |
|---|---|
| File | `syncnet_v2.model` |
| Size | 54,573,114 bytes |
| sha256 | `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442` |
| Origin | `download_model.sh` of github.com/joonson/syncnet_python -> https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model |
| Training | BBC/LRS talking-head footage (self-supervised sync) |
| License | MIT (repository LICENSE) |

```bash
curl -L -o detectors/syncnet/weights/syncnet_v2.model \
  'https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model'
```

Plain `S` state dict (88 tensors), loaded with `strict=True`.

```bibtex
@inproceedings{Chung16a,
  author = "Chung, J.~S. and Zisserman, A.",
  title = "Out of time: automated lip sync in the wild",
  booktitle = "Workshop on Multi-view Lip-reading, ACCV",
  year = "2016",
}
```
