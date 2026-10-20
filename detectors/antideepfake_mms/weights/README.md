# AntiDeepfake MMS-300M Weights

| Property | Value |
|---|---|
| Source | HuggingFace `nii-yamagishilab/mms-300m-anti-deepfake` (`model.safetensors`) |
| Size | 1,269,622,944 bytes |
| sha256 | `9bd5d9785bf72f91dfca8508651354946c4b307921cb358505e1b94035102224` |
| License | CC BY-NC-SA 4.0 (checkpoint); code BSD-3-Clause (github.com/nii-yamagishilab/AntiDeepfake) |
| Training | 56,370 h real + 18,280 h fake speech, 100+ languages (ASVspoof 2019/2021/5, MLAAD, CVoiceFake, SpoofCeleb, Codecfake, FLEURS(-R), MLS, VoxCeleb2(-Vocoded), ...), RawBoost (1)+(2) |

Weights auto-download from the HF hub at first run via `huggingface_hub`
(external kind — no local file needed). To pre-fetch:

```bash
hf download nii-yamagishilab/mms-300m-anti-deepfake model.safetensors config.json
```

The checkpoint is a flat safetensors state dict with fairseq wav2vec2 keys
(`m_ssl.model.*`) plus the FC head (`proj_fc.*`); `model.py` remaps them onto
`transformers.Wav2Vec2Model` at load time. Upstream logit order is
[fake, real]; the port swaps columns so Verity's index 1 = fake.

Variant: `VERITY_ANTIDEEPFAKE_REPO=nii-yamagishilab/xls-r-1b-anti-deepfake`
loads the XLS-R-1B detector (3.86 GB, 48 layers / 1280-d) with the same code.

Citation:

```bibtex
@inproceedings{ge2025antideepfake,
  title={Post-training for Deepfake Speech Detection},
  author={Ge, Wanying and Wang, Xin and Liu, Xuechen and Yamagishi, Junichi},
  booktitle={Proc. ASRU},
  year={2025},
  eprint={2506.21090},
  archivePrefix={arXiv}
}
```
