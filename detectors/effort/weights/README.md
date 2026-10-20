# Effort weights

Effort: Orthogonal Subspace Decomposition for Generalizable AI-Generated Image
Detection (Yan et al., ICML 2025, arXiv 2411.15633).
Upstream: https://github.com/YZY-stack/Effort-AIGI-Detection

Two weight sources combine at runtime:

1. **CLIP ViT-L/14 backbone** (`openai/clip-vit-large-patch14`, MIT) is fetched
   from HuggingFace by transformers at first run (shared `HF_HOME` cache, the
   same snapshot univfd/gend use). It is used only to *instantiate* the
   architecture: with a checkpoint present every tensor is overwritten.
2. **Effort checkpoint** (this directory). The default is the FaceForensics++
   model; it contains the whole vision tower (frozen CLIP weights, the SVD
   principal part `weight_main` and the learned rank-1 residuals
   `U_residual`/`S_residual`/`V_residual` of all 96 self-attention linears;
   681 keys) plus the `Linear(1024, 2)` head. The upstream module also
   registers the principal factors `U_r`/`S_r`/`V_r` as frozen parameters,
   but the released file does not contain them (they only feed training-time
   regularisers), so the port keeps them as non-persistent buffers. Loading is
   strict: any missing/unexpected key aborts.

| file | trained on | Google Drive id | targets |
|---|---|---|---|
| `effort_clip_L14_trainOn_FaceForensic.pth` (default) | FaceForensics++ c23 (DF, F2F, FS, NT) | `1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI` | face_swap, face_reenactment |
| `effort_clip_L14_trainOn_GenImage_sdv14.pth` (optional) | GenImage (SDv1.4 subset) | `1UXf1hC9FC1yV93uKwXSkdtepsgpIAU9d` | image_synthesis, face_synthesis |
| `effort_clip_L14_trainOn_Chameleon.pth` (optional) | Chameleon | `1GlJ1y4xmTdqV0FfIcyBwNNU6cQird9DR` | image_synthesis, face_synthesis |

Default checkpoint: `effort_clip_L14_trainOn_FaceForensic.pth`,
size 1213769519 bytes, sha256 `8d86711f098d16b49c048962fc3e16a906380f7bb17b8a0e89bd545b926943ee`.

Download (Google Drive; `confirm=t` skips the virus-scan interstitial):

```bash
cd detectors/effort/weights
curl -L -o effort_clip_L14_trainOn_FaceForensic.pth \
  'https://drive.usercontent.google.com/download?id=1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI&export=download&confirm=t'
# or: pip install gdown && gdown 1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI -O effort_clip_L14_trainOn_FaceForensic.pth
```

To use one of the alternates, download it into this directory with the same
command and the respective id, then point `weights.file` in
`detector.yaml` at it and update `targets` accordingly (the FF++ model
targets face deepfakes; the GenImage/Chameleon models target general
AI-generated images). Only one checkpoint is loaded per run.

## Preprocessing (upstream vs Verity)

Upstream feeds a 224x224 face crop normalised with the CLIP mean/std
(`[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`);
the crop is DeepfakeBench's dlib 5-point similarity alignment with scale 1.3
(about 30% margin around the face). Verity's engine uses an MTCNN box without
margin, resized to 224, with the same normalisation; this matches the other
DeepfakeBench-trained ports in this repo.

## License / citation

Code and checkpoints: CC BY-NC 4.0 (README badge; the repo ships no LICENSE
file). Non-commercial use only; do not re-host the checkpoint.

```bibtex
@inproceedings{yan2025effort,
  title     = {Orthogonal Subspace Decomposition for Generalizable AI-Generated Image Detection},
  author    = {Yan, Zhiyuan and Wang, Jiangming and Jin, Peng and Zhang, Ke-Yue and Liu, Chengchun and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
