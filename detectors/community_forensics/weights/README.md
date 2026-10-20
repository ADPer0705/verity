# Community Forensics Weights

| Property | Value |
|---|---|
| File | `model.safetensors` |
| Size | 87,262,324 bytes |
| SHA-256 | `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| Origin | HuggingFace `OwensLab/commfor-model-384` (revision `6076002bf0d9dd37537f965ee2f06f826c333b61`, not gated) |
| Format | safetensors, 152 float32 tensors (21,811,969 params), keys prefixed `vit.` (PyTorchModelHubMixin export of `ViTClassifier`) |
| Paper | Park, J., Owens, A. "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors." CVPR 2025. arXiv:2411.04125 |
| Code | https://github.com/JeongsooP/Community-Forensics (MIT) |
| Training | Community Forensics dataset (`OwensLab/CommunityForensics`): ~2.7 M images from ~4,800 generators + real images; ViT-S/16-384, BCE, label 1 = fake |
| License | MIT (HF model card `license: mit`; code repository MIT) — redistributable |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o weights/model.safetensors \
  'https://huggingface.co/OwensLab/commfor-model-384/resolve/main/model.safetensors'
```

A transformers-format re-upload (`buildborderless/CommunityForensics-DeepfakeDet-ViT`)
also exists on the hub; this port uses the authors' own repository.

```bibtex
@inproceedings{park2025community,
  title={Community Forensics: Using Thousands of Generators to Train Fake Image Detectors},
  author={Park, Jeongsoo and Owens, Andrew},
  booktitle={CVPR},
  year={2025}
}
```
