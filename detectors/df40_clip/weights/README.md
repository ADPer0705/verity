# DF40 CLIP weights

DF40: Toward Next-Generation Deepfake Detection (Yan et al., NeurIPS 2024
Datasets & Benchmarks, arXiv 2406.13495). Upstream: https://github.com/YZY-stack/DF40
Checkpoint folder: https://drive.google.com/drive/folders/1HDgIOutGw3jsFXwvSQYeDoVPAzgfYbyr

Two weight sources combine at runtime:

1. **CLIP ViT-L/14 backbone** (`openai/clip-vit-large-patch14`, MIT) is fetched
   from HuggingFace by transformers at first run (shared `HF_HOME` cache, the
   same snapshot univfd/gend/effort use). It is used only to *instantiate* the
   architecture: with a checkpoint present every tensor is overwritten.
2. **DF40 checkpoint** (this directory): the whole fine-tuned vision tower plus
   the `Linear(1024, 2)` head. Loading is strict: any missing/unexpected key
   aborts.

The Drive folder holds `clip.pth` (CLIP ViT-B/16, 343 MB, head 768->2) *and*
`clip_large.pth` (CLIP ViT-L/14, ~1.2 GB, head 1024->2) for each training
protocol. This port implements the ViT-L/14 model of the current upstream
`clip_detector.py`, so only the `clip_large.pth` files are usable here.

| file in Drive | local name (`weights.file`) | trained on | Drive id | targets |
|---|---|---|---|---|
| `train_on_df40/clip_large.pth` (default) | `df40_train_on_df40_clip_large.pth` | all DF40 training families (FS + FR + EFS, FF++ domain) | `1MzvaQSE-SpH9a5mgCXxan_J_PTZbNaPa` | face_swap, face_reenactment, face_synthesis |
| `train_on_fs/clip_large.pth` | `df40_train_on_fs_clip_large.pth` | face-swap methods | `1RN6nCuApp0tSaku0FyBAi7uw3xV-CuWv` | face_swap |
| `train_on_fr/clip_large.pth` | `df40_train_on_fr_clip_large.pth` | face-reenactment methods | `1dvUK1xWOYpReygwwvoCZbwSg71dbOEI_` | face_reenactment |
| `train_on_efs/clip_large.pth` | `df40_train_on_efs_clip_large.pth` | entire-face-synthesis methods | `1RWEoYAscBLTH8Bgeq55Tr5eDmL632AoQ` | face_synthesis |

Default checkpoint: `df40_train_on_df40_clip_large.pth`,
size 1,212,881,455 bytes, SHA-256
`a033da18fb0152e69e47b4e2338867f62c05d51656639c745402bf7c07dd7259`.
The same numbers live in `detectors/weights_manifest.yaml`, which is what
`verity weights pull --detector df40_clip` verifies against.

Download (Google Drive; `confirm=t` skips the virus-scan interstitial):

```bash
cd detectors/df40_clip/weights
curl -L -o df40_train_on_df40_clip_large.pth \
  'https://drive.usercontent.google.com/download?id=1MzvaQSE-SpH9a5mgCXxan_J_PTZbNaPa&export=download&confirm=t'
# or: pip install gdown && gdown 1MzvaQSE-SpH9a5mgCXxan_J_PTZbNaPa -O df40_train_on_df40_clip_large.pth
```

To use an alternate, download it with its id under the local name above, set
`weights.file` in `detector.yaml` to that name and update `targets`
accordingly. Only one checkpoint is loaded per run.

## Preprocessing

Upstream (`clip.yaml`): 224x224 face crops normalised with mean/std 0.5 (not
the CLIP statistics; the `AutoProcessor` is created but never applied).
Faces come from DeepfakeBench's dlib alignment with scale 1.3; Verity uses an
MTCNN box without margin, resized to 224, as for the other DeepfakeBench ports.

## License / citation

Dataset, code and checkpoints: CC BY-NC 4.0 (stated in the upstream README).
Non-commercial use only; do not re-host the checkpoint.

```bibtex
@inproceedings{yan2024df40,
  title     = {DF40: Toward Next-Generation Deepfake Detection},
  author    = {Yan, Zhiyuan and Yao, Taiping and Chen, Shen and Zhao, Yandan and Fu, Xinghe and Zhu, Junwei and Luo, Donghao and Wang, Chengjie and Ding, Shouhong and Wu, Yunsheng and Yuan, Li},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024}
}
```
