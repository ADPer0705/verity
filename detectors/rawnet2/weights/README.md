# RawNet2 (RawBoost) weights

| Property | Value |
|---|---|
| File | `Best_model.pth` |
| Size | 70,515,422 bytes |
| sha256 | `d8afc379456876c78fa08e61d5fcc66bba6f459b30bbbf3dcb7a7f0731e701aa` |
| Origin | `pre_trained_model/Best_model.pth` in github.com/TakHemlata/RawBoost-antispoofing |
| Training | ASVspoof2019 LA train with RawBoost augmentation (evaluated on ASVspoof 2021 LA) |
| License | MIT (repository LICENSE, EURECOM) |

```bash
curl -L -o detectors/rawnet2/weights/Best_model.pth \
  'https://raw.githubusercontent.com/TakHemlata/RawBoost-antispoofing/main/pre_trained_model/Best_model.pth'
```

Plain `RawNet` state dict loaded with `strict=True` (the sinc filterbank has
no parameters). Upstream logit order is [spoof, bonafide]; the port swaps
the columns so Verity's index 1 = fake.

```bibtex
@inproceedings{tak2021rawnet2,
  title={End-to-End anti-spoofing with RawNet2},
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={Proc. ICASSP},
  year={2021}
}
@inproceedings{tak2022rawboost,
  title={RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing},
  author={Tak, Hemlata and Kamble, Madhu and Patino, Jose and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={Proc. ICASSP},
  year={2022}
}
```
