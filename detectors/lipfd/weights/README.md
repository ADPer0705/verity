# LipFD weights

| Property | Value |
|---|---|
| File | `ckpt.pth` |
| Size | 1,805,064,538 bytes |
| sha256 | `156fc8995a2ff7b50b56a1dbafb490b6936d5aa6c18dee5a6c67018853f2d9af` |
| Origin | Google Drive id `1NPAcx0QS8N9v_9qUr-51jBaL9kGDT-cp` ("pertained weights" in the README of AaronComo/LipFD) |
| Training | AVLips (Wav2Lip / TalkLip / MakeItTalk / SadTalker lip-sync fakes) |
| License | the repository ships no LICENSE file (all rights reserved by default); fetched from the authors' Drive only |

```bash
cd detectors/lipfd/weights
curl -L -o ckpt.pth \
  'https://drive.usercontent.google.com/download?id=1NPAcx0QS8N9v_9qUr-51jBaL9kGDT-cp&export=download&confirm=t'
```

Layout: `{"model": state_dict, "optimizer": ..., "total_steps": ...}`;
770 tensors: `conv1.*`, `encoder.*` (full OpenAI CLIP ViT-L/14 incl. the
unused text tower) and `backbone.*` (region-aware ResNet-50 with
`get_weight` and `fc`), loaded strictly. A third-party HF mirror
(`yh007/LipFD`, `model_epoch_10.pth`) exists but is not the official file.

```bibtex
@inproceedings{liu2024lips,
  author = {Liu, Weifeng and She, Tianyi and Liu, Jiawei and Li, Boheng and Yao, Dongyu and Liang, Ziyou and Wang, Run},
  title = {Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes},
  booktitle = {NeurIPS},
  year = {2024}
}
```
