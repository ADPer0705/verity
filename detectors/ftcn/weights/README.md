# FTCN weights

| Property | Value |
|---|---|
| File | `ftcn_tt.pth` |
| Size | 59,248,500 bytes |
| sha256 | `3d08fc78174bbfdb3fbede317b95dcac9095202ed7113649eecf6e2ef5ee77d8` |
| Origin | GitHub release `weights` of yinglinzheng/FTCN (linked from the README: "FTCN+TT model trained on FF++") |
| Training | FaceForensics++ c23 |
| License | the repository ships no LICENSE file (all rights reserved by default); fetched from the authors' release only |

```bash
curl -L -o detectors/ftcn/weights/ftcn_tt.pth \
  'https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth'
```

Flat `resnet.*` state dict of the rewired FTCN (temporal-only I3D R50 up to
stage 4 + `head.time_T` TimeTransformer), loaded strictly after the port's
`temporal_only_conv` rewire and head swap.

Alternative (research, backbone-only): the Kinetics `I3D_8x8_R50.pth` from
DeepfakeBench (set `weights.file` and `kind: backbone`); the head is then
untrained and the detector is not evidence-grade.

```bibtex
@inproceedings{zheng2021exploring,
  title={Exploring Temporal Coherence for More General Video Face Forgery Detection},
  author={Zheng, Yinglin and Bao, Jianmin and Chen, Dong and Zeng, Ming and Wen, Fang},
  booktitle={ICCV},
  year={2021}
}
```
