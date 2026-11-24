# ProDet weights

| Property | Value |
|---|---|
| File | `ProDet_best.pth` |
| Size | 385,663,781 bytes |
| sha256 | `1de84318fa703978493a4c4bd7e337e2597e8e3e36fc9d144c42842456cb3017` |
| Origin | Google Drive folder `16IDcKVqziJ-Qv_IfZOAcYGwdeZAM-x5s` (README of beautyremain/ProDet), file id `18cqHkDdAxXDnbyZZ5KiZG5gD5gkh6A1v` |
| Training | FaceForensics++ c23 (DeepfakeBench pipeline) with SBI-style blends |
| License | the repository ships no LICENSE file (all rights reserved by default); fetched from the authors' Drive only |

```bash
cd detectors/prodet/weights
curl -L -o ProDet_best.pth \
  'https://drive.usercontent.google.com/download?id=18cqHkDdAxXDnbyZZ5KiZG5gD5gkh6A1v&export=download&confirm=t'
```

Layout: flat state dict with 782 tensors (`backbone.*`, `adjust_feature`,
`*_classifier`, `fea_att`, and the training-only `real2sbi.*` / `bi2fake.*`
mix-up generators which the port drops). The other two files in the folder
(`nearest_face_info_worep.pkl`, `landmark_dict_new.pkl`) are training data
and are not needed.

```bibtex
@inproceedings{cheng2024can,
  title={Can We Leave Deepfake Data Behind in Training Deepfake Detector?},
  author={Cheng, Jikang and Yan, Zhiyuan and Zhang, Ying and Luo, Yuhao and Wang, Zhongyuan and Li, Chen},
  booktitle={NeurIPS},
  year={2024}
}
```
