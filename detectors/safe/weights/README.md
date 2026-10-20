# SAFE Weights

| Property | Value |
|---|---|
| File | `checkpoint-best.pth` |
| Size | 5,840,638 bytes |
| SHA-256 | `b3f5ecfb46a154ed553aaaf4bf3ba59182310726ddb0cbb1fe42bd0e22d2f20e` |
| Origin | Official repository `Ouxiang-Li/SAFE`, `checkpoint/checkpoint-best.pth` (main branch, commit `4e998724`) |
| Format | `{"model": state_dict}`, 146 keys (`conv1`, `bn1`, `layer1.*`, `layer2.*`, `fc1` = Linear(512, 2)); 1,448,410 parameters |
| Paper | Li, O., Cai, J., Hao, Y., Jiang, X., Hu, Y., Feng, F. "Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective." KDD 2025. arXiv:2408.06741 |
| Training | ForenSynths / ProGAN (car, cat, chair, horse), 256 px random crops, SAFE augmentations (flip, rotation, color jitter, random mask) |
| License | Apache-2.0 (repository LICENSE) — redistributable |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o weights/checkpoint-best.pth \
  'https://github.com/Ouxiang-Li/SAFE/raw/main/checkpoint/checkpoint-best.pth'
```

```bibtex
@inproceedings{li2025improving,
  title={Improving synthetic image detection towards generalization: An image transformation perspective},
  author={Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Feng, Fuli},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={2405--2414},
  year={2025}
}
```
