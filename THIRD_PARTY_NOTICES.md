# Third-party notices

Verity (CC BY-NC 4.0) builds on the work of others. This file lists what is
included or fetched, and under which terms. Checkpoint entries are generated
from `detectors/weights_manifest.yaml`; keep that file authoritative.

## Detector code

The implementations under `detectors/` are inference-only ports:

- **DeepfakeBench** (SCLBD/DeepfakeBench), CC BY-NC 4.0, Copyright (c) 2023
  The Chinese University of Hong Kong, Shenzhen. Source of the Xception,
  MesoNet, MesoInception, CNN-Aug/ResNet-34, EfficientNet-B4, Capsule, CORE,
  F3Net, FFD, SPSL, SRM, UCF, RECCE, I3D, FTCN, AltFreezing, Face X-ray, IID,
  SBI, SLADD, PCL/I2G, LRL, LSDA, Multi-attention, SIA, RFM, DSP-FWA, TALL,
  STIL, UIA-ViT, CLIP, X-CLIP, VideoMAE and TimeSformer detector ports.
  Yan, Z., Zhang, Y., Yuan, X., Lyu, S., Wu, B. "DeepfakeBench: A Comprehensive
  Benchmark of Deepfake Detection." NeurIPS 2023.
- **NPR** (chuangchuangtan/NPR-DeepfakeDetection), no license stated. Tan, C. et al., CVPR 2024.
- **MLEP** (fkeufss/MLEP), no license stated. Yuan, L. et al., NeurIPS 2025.
- **UniversalFakeDetect** (WisconsinAIVision/UniversalFakeDetect), MIT. Ojha, U., Li, Y., Lee, Y. J., CVPR 2023.
- **LipForensics** (ahaliassos/LipForensics), MIT. Haliassos, A. et al., CVPR 2021.
- **GenD** (yermandy/GenD), MIT. Yermakov, A. et al., WACV 2026.
- **AEROBLADE** (jonasricker/aeroblade), training-free method. Ricker, J. et al., CVPR 2024.
- **Effort** (YZY-stack/Effort-AIGI-Detection), CC BY-NC 4.0. Yan, Z. et al., ICML 2025.
- **DF40** (YZY-stack/DF40), CC BY-NC 4.0. Yan, Z. et al., NeurIPS 2024 (Datasets & Benchmarks).
- **AASIST** (clovaai/aasist), MIT, Copyright (c) 2021-present NAVER Corp. Jung, J.-W. et al., ICASSP 2022.
- **AntiDeepfake** (nii-yamagishilab/AntiDeepfake), code BSD-3-Clause; checkpoint CC BY-NC-SA 4.0. Ge, W. et al., ASRU 2025.
- **Community Forensics** (JeongsooP/Community-Forensics), MIT. Park, J., Owens, A., CVPR 2025.
- **SAFE** (Ouxiang-Li/SAFE), Apache-2.0. Li, O. et al., KDD 2025.
- **XLS-R + SLS** (QiShanZhang/SLSforASVspoof-2021-DF), no license stated in the upstream repository; the used ONNX export is re-hosted at HF SpeechAntiSpoofingBenchmarks/XLSR-SLS under an MIT model card. Zhang, Q. et al., ACM MM 2024.

DeepfakeBench itself vendors components under Apache-2.0 (timm,
efficientnet-pytorch) and BSD-3-Clause; their notices apply where that code
is used.

## Checkpoints fetched by `verity weights pull`

| File | Source | License | Distribution | SHA-256 |
|---|---|---|---|---|
| `AASIST.pth` | AASIST (clovaai/aasist) | MIT | mirrored | `51d2d9cf0738…` |
| `model.safetensors` | Community Forensics (OwensLab/commfor-model-384) | MIT | mirrored | `b89f36275f3b…` |
| `I3D_8x8_R50.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `c9e4e6c8e86c…` |
| `capsule_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `40e96997c224…` |
| `cnnaug_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `2ffdc665b9bd…` |
| `core_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `3d34dca33ab6…` |
| `effnb4_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `74cbd82d7be1…` |
| `f3net_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `deee7d25bfc1…` |
| `ffd_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `445f5b70f115…` |
| `meso4Incep_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `16a2d7cec846…` |
| `meso4_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `a362ebbef265…` |
| `recce_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `1349fd80d75c…` |
| `spsl_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `7305d660c68a…` |
| `srm_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `666e8fc7f993…` |
| `ucf_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `4afa2517feeb…` |
| `xception-b5690688.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | BSD-3-Clause | mirrored | `b56906886cba…` |
| `xception_best.pth` | DeepfakeBench (SCLBD/DeepfakeBench) | CC-BY-NC-4.0 | mirrored | `dd61186f9ac8…` |
| `df40_train_on_df40_clip_large.pth` | DF40 (YZY-stack/DF40) | CC-BY-NC-4.0 | authors’ URL only | `a033da18fb01…` |
| `effort_clip_L14_trainOn_FaceForensic.pth` | Effort (YZY-stack/Effort-AIGI-Detection) | CC-BY-NC-4.0 | authors’ URL only | `8d86711f098d…` |
| `hrnetv2_w48_imagenet_pretrained.pth` | HRNet-Image-Classification (HRNet/HRNet-Image-Classification) | MIT | mirrored | `0efec102d97f…` |
| `lipforensics_ff.pth` | LipForensics (ahaliassos/LipForensics) | MIT | mirrored | `4b7790bc8e02…` |
| `model_epoch_best.pth` | MLEP (fkeufss/MLEP) | none-stated | authors’ URL only | `09a6b934001c…` |
| `NPR.pth` | NPR-DeepfakeDetection (chuangchuangtan/NPR-DeepfakeDetection) | none-stated | authors’ URL only | `3939297e9399…` |
| `checkpoint-best.pth` | SAFE (Ouxiang-Li/SAFE) | Apache-2.0 | mirrored | `b3f5ecfb46a1…` |
| `fc_weights.pth` | UniversalFakeDetect (WisconsinAIVision/UniversalFakeDetect) | MIT | mirrored | `477100745713…` |
| `xlsr-sls.onnx` | XLS-R + SLS (QiShanZhang/SLSforASVspoof-2021-DF) | MIT per the HF card; upstream repository ships no LICENSE (none stated) | authors’ URL only | `0aa36298a1a8…` |

"Mirrored" files are re-hosted unmodified on the Verity weights repo with
this attribution; "authors' URL only" files are downloaded from the authors'
repositories because no redistribution right was granted.

The DeepfakeBench checkpoints were trained on FaceForensics++ (Rössler et
al., ICCV 2019), which is released for research purposes only.

## Models fetched at first run by the detectors themselves

| Detector | Models and licenses |
|---|---|
| `aeroblade` | CompVis/stable-diffusion-v1-1 (OpenRAIL-M), stabilityai/stable-diffusion-2-1-base (OpenRAIL++; gated on HF), kandinsky-community/kandinsky-2-1 (Apache-2.0), LPIPS VGG (BSD-2) — training-free |
| `antideepfake_mms` | nii-yamagishilab/mms-300m-anti-deepfake (HF, CC BY-NC-SA 4.0 checkpoint; BSD-3-Clause detector code) — fetched intact by huggingface_hub at first run; XLS-R-1B variant via VERITY_ANTIDEEPFAKE_REPO |
| `clip` | openai/clip-vit-base-patch16 (HF, MIT) |
| `df40_clip` | openai/clip-vit-large-patch14 (HF, MIT) backbone, used to instantiate the architecture; the trained DF40 head itself comes from the weights manifest |
| `effort` | openai/clip-vit-large-patch14 (HF, MIT) backbone, used to instantiate the architecture; the trained Effort head itself comes from the weights manifest |
| `gend` | yermandy/GenD_CLIP_L_14 (HF, MIT) — released detector |
| `stil` | nothing is fetched today (random init); upstream uses SCNet50-v1d ImageNet (backseason/SCNet, MIT) |
| `timesformer` | facebook/timesformer-base-finetuned-k400 (HF, CC BY-NC 4.0) |
| `uia_vit` | nothing is fetched today (random init); upstream uses timm jx_vit_base_p16_224 (Apache-2.0) |
| `univfd` | openai/clip-vit-large-patch14 (HF, MIT) backbone |
| `videomae` | MCG-NJU/videomae-base (HF, CC BY-NC 4.0) |
| `xclip` | microsoft/xclip-base-patch32 (HF, MIT) |

## Runtime dependencies

The orchestrator uses Click (BSD-3), Pydantic (MIT), OpenCV (Apache-2.0),
Pillow (MIT-CMU), NumPy (BSD-3), Matplotlib (PSF-based), WeasyPrint (BSD-3),
FastAPI (MIT), Uvicorn (BSD-3), FFmpeg (LGPL/GPL, system package). The web
console uses React (MIT), Mantine (MIT) and the Inter typeface (OFL-1.1). Detector environments
pin PyTorch (BSD-3), torchvision (BSD-3), facenet-pytorch (MIT), MediaPipe
(Apache-2.0), transformers (Apache-2.0), diffusers (Apache-2.0), timm
(Apache-2.0) and others as declared per detector in `detectors/<id>/pyproject.toml`.
