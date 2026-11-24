---
type: reference
title: Detector catalog (all 58 ports)
description: Complete field guide and reference inventory for all 58 ported detector architectures (39 evidence-grade trained models and 19 research scaffolds), covering modality groups, targets, checkpoint licenses, wiring invariants, caveats, reviewed exclusions, and recommended panels.
tags: [detectors, catalog, reference, trust-tiers, modalities, caveats]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-e95e39a3e70d050c490b69d6
    resource: repo://detectors/aeroblade/model.py
  - id: openwiki-source-f618a306205a5dd4ac94ebef
    resource: repo://detectors/common/spec.py
  - id: openwiki-source-d99d37162a6a9beafd358c12
    resource: repo://detectors/efficientnetb4/model.py
  - id: openwiki-source-490ab6995a830adcd6f58749
    resource: repo://detectors/recce/inference.py
  - id: openwiki-source-d6b6ec4e1d20c1d7ac5ea76a
    resource: repo://detectors/weights_manifest.yaml
  - id: openwiki-source-d4efdb0b02874ae91ad21a2b
    resource: repo://scripts/gen_detector_table.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Detector catalog (all 58 ports)

Verity ships **58 detector architectures** under `detectors/<id>/`:

- **39 `trained`** (evidence-grade): released checkpoints with trained
  classification heads or training-free forensic methods. **Only this tier is
  auto-selected** and counted in panel consensus.
- **4 `backbone-only`**: pretrained ImageNet or Kinetics backbones with untrained
  classification heads (`altfreezing`, `facexray`, `i3d`, `iid`).
- **4 `untrained`**: external foundation encoders (`clip`, `timesformer`,
  `videomae`, `xclip`) with randomly initialized detector heads.
- **11 `random-init`**: research scaffolds (`fwa`, `lrl`, `lsda`,
  `multi_attention`, `pcl_xception`, `rfm`, `sia`, `sladd`, `stil`, `tall`,
  `uia_vit`) awaiting investigator-supplied weights (`trained: true`).

Run `uv run python scripts/gen_detector_table.py` to generate a live summary
directly from `detectors/*/detector.yaml` and `detectors/weights_manifest.yaml`,
or `verity weights status` for on-disk checkpoint verification.

## Cross-cutting wiring & calibration invariants

All 58 ports adhere to strict shared engine contracts (`detectors/common/`):

- **Normalized logit convention**: Upstream checkpoints differ in class index
  order; every `model.py` normalizes outputs so **`logits[:, 1]` = fake/spoof**.
- **Strict key-prefix stripping**: DeepfakeBench checkpoints store keys under
  `backbone.*` and auxiliary `loss_func.*`. `efficientnetb4`, `meso4`,
  `meso4Inception`, `resnet34`, and siblings explicitly unwrap `state_dict` /
  `module.` / `backbone.` so weights load properly instead of silently staying at
  random initialization (`strict=False` mismatches logged clearly).
- **CAM & native attention honesty**: Visual boundary/patch detectors (`laa_net`,
  `forensics_adapter`, `vlaforge`, `df40_clip`, `srm`, `prodet`, `recce`,
  `effort`) expose native attention or Grad-CAM where semantically meaningful,
  while temporal token transformers (`timesformer`, `videomae`, `xclip`, `tall`),
  wavelet sub-band models (`safe`), LDM distance detectors (`aeroblade`), frozen
  linear probes (`univfd`), and waveform audio detectors explicitly declare
  `cam: false` instead of emitting misleading gradients.
- **Deterministic seeds & sampling**: Seeds (`0xC0FFEE`), sequential frame reads,
  and seeded transforms (`d3` patch shuffle, `forensics_adapter` fusion projections,
  `aeroblade` VAE generators) ensure repeatable scores across runs.

---

## 1. Image & video face-crop detectors (35 ports: 24 trained, 11 research)

Consume **MTCNN face crops** (`input_mode: face_crop`), resized to each
detector's `resolution` with `face_margin` expansion when upstream training used a
crop margin. Emit `faces/`, `attention_maps/*_attention.png`, `region_energy`,
`full_frames/*_frame.png`, and `timeline.png`.

### Trained face-crop detectors (24)

| id | Model & Paper | Weights / License | Key Characteristics & Explainability |
|---|---|---|---|
| `xception` | Xception (Rosler et al., ICCV 2019) | `xception_best.pth` (`CC-BY-NC-4.0`, mirrored) | FF++ c23 classic panel baseline; spatial blend artifacts + Grad-CAM |
| `efficientnetb4` | EfficientNet-B4 (Bonettini et al. 2021) | `effnb4_best.pth` (`CC-BY-NC-4.0`, mirrored) | Unwraps `backbone.*` prefix; 1792-d features + Grad-CAM |
| `resnet34` | CNN-Aug / ResNet-34 | `cnnaug_best.pth` (`CC-BY-NC-4.0`, mirrored) | Augmentation-trained CNN; Grad-CAM |
| `meso4` | MesoNet-4 (Afchar et al. 2018) | `meso4_best.pth` (`CC-BY-NC-4.0`, mirrored) | Lightweight (~120 KB) mesoscopic CNN; fast compressed-video signal + Grad-CAM |
| `meso4Inception` | MesoInception-4 | `meso4Incep_best.pth` (`CC-BY-NC-4.0`, mirrored) | Inception blocks with wavelet-inspired filters + Grad-CAM |
| `capsule_net` | Capsule-Forensics V2 (Hashmi et al. 2021) | `capsule_best.pth` (`CC-BY-NC-4.0`, mirrored) | VGG19 + dynamic capsule routing + Grad-CAM (caveat: can over-commit on degraded inputs) |
| `f3net` | F3Net (Qiu et al. 2020) | `f3net_best.pth` (`CC-BY-NC-4.0`, mirrored) | Xception + 12-ch FAD frequency-domain decomposition; strong when traces hide outside RGB |
| `spsl` | SPSL (Liu et al. 2021) | `spsl_best.pth` (`CC-BY-NC-4.0`, mirrored) | Xception + 4-ch DCT phase spectrum capturing upsampling artifacts |
| `srm` | SRM (DeepfakeBench) | `srm_best.pth` (`CC-BY-NC-4.0`, mirrored) | Dual Xception (RGB + SRM high-pass filters) + native cross-modal attention |
| `ffd` | FFD (Li et al., ECCV 2020) | `ffd_best.pth` (`CC-BY-NC-4.0`, mirrored) | Dual-branch Xception + regression mask classification |
| `core` | CORE (Zhao et al. 2021) | `core_best.pth` (`CC-BY-NC-4.0`, mirrored) | Consistency-regularized Xception representations + Grad-CAM |
| `ucf` | UCF (DeepfakeBench) | `ucf_best.pth` (`CC-BY-NC-4.0`, mirrored) | Dual Xception fingerprint disentanglement + Grad-CAM |
| `recce` | RECCE (Cao et al., CVPR 2022) | `recce_best.pth` (`CC-BY-NC-4.0`, mirrored) | Cooperative reconstruction-classification. **Honest audit**: official checkpoint has an untrained decoder (`std < 1e-4`), so RECCE suppresses fake recon maps and relies on classifier logits + guided attention |
| `gend` | GenD (WACV 2026) | `yermandy/GenD_CLIP_L_14` (`MIT`, runtime HF) | LayerNorm fine-tuned CLIP ViT-L/14 + L2 linear probe; covers face & full synthesis + ViT CAM |
| `df40_clip` | DF40 CLIP-L/14 (Yan et al., NeurIPS 2024) | `df40_train_on_df40_clip_large.pth` (`CC-BY-NC-4.0`, **origin-only**) | Fine-tuned CLIP ViT-L/14 across DF40 swaps/reenactment/synthesis + CLS patch attention |
| `effort` | Effort (Yan et al., ICML 2025) | `effort_clip_L14_trainOn_FaceForensic.pth` (`CC-BY-NC-4.0`, **origin-only**) | CLIP ViT-L/14 with orthogonal SVD rank-residual attention; generalizes strongly from FF++ |
| `sbi` | Self-Blended Images (Shiohara & Yamasaki, CVPR 2022) | `FFc23.tar` (`SBI-academic`, **origin-only**) | EfficientNet-B4 trained purely on real faces + self-blended pseudo-fakes (380 px, 25% `face_margin`) |
| `prodet` | ProDet (Cheng et al., NeurIPS 2024) | `ProDet_best.pth` (`none-stated`, **origin-only**) | Progressive real -> blend -> id -> fake auxiliary heads stored in `aux` + `FeatureAttentionBlock` map |
| `laa_net` | LAA-Net (Nguyen et al., CVPR 2024) | `laa_net_sbi.pth` (`SnT-academic`, **origin-only**) | PoseEfficientNet-B4 + E-FPN explicit 96x96 **blending boundary heatmap** (384 px, 25% margin, flip-test avg) |
| `m2tr` | M2TR (Wang et al., ICMR 2022) | `m2tr_ff_c23.pth` (`MIT`, mirrored) | Static-pad EfficientNet-B4 + 4-stage multi-scale patch Transformer + rFFT global filter |
| `forensics_adapter` | Forensics Adapter (Cui et al., CVPR 2025) | `forensics_adapter.pth` (`CC-BY-NC-4.0`, mirrored) | Frozen CLIP ViT-L/14 steered by 128-query ViT-tiny blending adapter + RecAttn boundary attention map |
| `vlaforge` | VLAForge (Zhu et al., CVPR 2026) | `vlaforge.pth` (`MIT`, mirrored) | ForgePerceiver + identity-injected VLA text/patch head (`clip_branch` & `vla_branch` in `aux`) |
| `m2f2_det` | M2F2-Det Stage-1 (Guo et al., CVPR 2025) | `m2f2_det_stage1.pth` (`MIT`, mirrored) | CLIP ViT-L/14-336 + EfficientNet-B4 fusion. **Caveat**: upstream `stage_1` weights omitted 3 bridge-adapter blocks, so 128 of 3,712 features are deterministically re-initialized |

### Research face-crop scaffolds (11 — excluded by default)

- `facexray` (`backbone-only`, `hrnetv2_w48_imagenet_pretrained.pth`, `MIT`, mirrored)
- `iid` (`backbone-only`, `xception-b5690688.pth`, `BSD-3-Clause`, mirrored)
- `clip` (`untrained`, `openai/clip-vit-base-patch16`, `MIT`, runtime HF)
- `fwa`, `lrl`, `lsda`, `multi_attention`, `pcl_xception`, `rfm`, `sia`, `sladd` (`random-init`, no checkpoints)

---

## 2. AI-generated whole-frame detectors (7 ports, all trained)

Consume **`full_frame` images and sampled video frames without face cropping**
(able to examine landscapes, documents, CGI scenes, and non-facial generative
imagery). Write `frames/`, optional `attention_maps/`, and video `timeline.png`.

| id | Model & Paper | Weights / License | Key Characteristics |
|---|---|---|---|
| `univfd` | UniversalFakeDetect (Ojha et al., CVPR 2023) | `fc_weights.pth` (`MIT`, mirrored) + runtime CLIP ViT-L/14 | Frozen 768-d linear probe; `cam: false` (no spatial head) |
| `community_forensics` | Community Forensics (Park & Owens, CVPR 2025) | `model.safetensors` (`MIT`, mirrored) | ViT-S/16 @ 384 px trained across ~4,800 generators + patch-token Grad-CAM |
| `npr` | NPR (Tan et al., CVPR 2024) | `NPR.pth` (`none-stated`, **origin-only**) | Neighbourhood Pixel Ratio over native frames (ResNet-50 trunk) + Grad-CAM |
| `mlep` | MLEP (Yuan et al., NeurIPS 2025) | `model_epoch_best.pth` (`none-stated`, **origin-only**) | 9-ch multi-scale local entropy maps on native frames + entropy-space Grad-CAM |
| `safe` | SAFE (Li et al., KDD 2025) | `checkpoint-best.pth` (`Apache-2.0`, mirrored) | **No resize**: native 256x256 center crop -> `bior1.3` DWT HH sub-band (official eval behavior documented in reasoning); `cam: false` |
| `d3` | D³ (Yang et al., CVPR 2025) | `d3_classifier.pth` (`none-stated`, **origin-only**) + CLIP ViT-L/14 | Seeded 14 px patch-shuffle discrepancy vs intact image over frozen CLIP features |
| `aeroblade` | AEROBLADE (Ricker et al., CVPR 2024) | Training-free: SD1.1 VAE + SD2.1 mirror (`LanguageMachines/stable-diffusion-2-1-base`) + Kandinsky 2.1 MoVQ (`Apache-2.0`/`OpenRAIL`, runtime HF) | VGG LPIPS-2 reconstruction distance (`sd1_lpips`, `sd2_lpips`, `kd2_lpips`, `min_lpips` exposed in `aux`). Generated images reconstruct better than real ones (`THRESHOLD = 0.2`); `cam: false` |

*(Note: `gend` also targets `image_synthesis` alongside faces.)*

---

## 3. Temporal video face-clip detectors (10 ports: 3 trained, 7 research)

Consume **`face_clip` sequences (`video_mode: true`)**: consecutive tracked face
crops per window to detect frame-to-frame motion or coherence anomalies.

### Trained temporal detectors (3)

| id | Model & Paper | Weights / License | Key Characteristics |
|---|---|---|---|
| `lipforensics` | LipForensics (Haliassos et al., CVPR 2021) | `lipforensics_ff.pth` (`MIT`, mirrored) | 25-frame mouth crops -> Conv3d + ResNet-18 + MultiscaleTCN (`lip_sync`, swaps, reenactment) |
| `ftcn` | FTCN (Zheng et al., ICCV 2021) | `ftcn_tt.pth` (`none-stated`, **origin-only**) | 16-frame clips -> temporal-only 1x1xK I3D R50 + TimeTransformer head + temporal Grad-CAM |
| `realforensics` | RealForensics (Haliassos et al., CVPR 2022) | `realforensics_ff.pth` (`MIT`, mirrored) | 25-frame FAN inner-face aligned clips -> BYOL-AV CSN-R101 cosine head (`scale=64`) |

### Research temporal scaffolds (7 — excluded by default)

- `i3d` & `altfreezing` (`backbone-only`, shared Kinetics `I3D_8x8_R50.pth`, `CC-BY-NC-4.0`, mirrored)
- `timesformer` (`facebook/timesformer-base-finetuned-k400`, `CC-BY-NC-4.0`, `untrained`, `cam: false`)
- `videomae` (`MCG-NJU/videomae-base`, `CC-BY-NC-4.0`, `untrained`, `cam: false`)
- `xclip` (`microsoft/xclip-base-patch32`, `MIT`, `untrained`, `cam: false`)
- `stil` & `tall` (`random-init`, `cam: false` on `tall`)

---

## 4. Audio-visual lip-sync detectors (2 ports, both trained)

Consume **`av_clip` windows** (synchronized video face sequences + matching audio
segments), writing spectrogram overlays (`timeline.png`), `frames/`, `faces/`, and
`per_face[*].info` measurements:

- **`lipfd`** (`LipFD`, NeurIPS 2024; `lipfd_ckpt.pth`, `none-stated`, **origin-only**):
  5 frames stacked with viridis mel spectrogram -> CLIP ViT-L/14 + 3-scale gated ResNet-50 crops.
- **`syncnet`** (`SyncNet v2`, ACCV 2016; `syncnet_v2.model`, `MIT`, mirrored):
  Measurement model reporting **`av_offset_frames`**, **`min_distance`**, and LSE-C
  **`confidence`** in `aux_aggregate` (`fake_prob = sigmoid(4 - confidence)`).
  *Examiner rule*: authentic speech sits around LSE-C `6–10` at offset `~0`; a
  large constant offset with *high* confidence indicates an A/V muxing delay,
  not synthetic lip manipulation.

---

## 5. Audio anti-spoofing detectors (4 ports, all trained)

Consume **`audio` 16 kHz mono windows** (standalone audio files or extracted
video soundtracks), outputting `timeline.png` over spectrograms and `per_face`
time-span rows (`start_s`–`end_s`). Waveform detectors judge *synthetic speech*,
never speaker identity:

| id | Model & Paper | Weights / License | Key Characteristics & Caveats |
|---|---|---|---|
| `antideepfake_mms` | AntiDeepfake MMS-300M (Ge et al., ASRU 2025) | `nii-yamagishilab/mms-300m-anti-deepfake` (`CC-BY-NC-SA-4.0`, runtime HF) | 24-layer wav2vec2 trained on 74,650 h across 100+ languages (`VERITY_ANTIDEEPFAKE_REPO` switches to XLS-R-1B); strongest multilingual audio prior |
| `xlsr_sls` | XLS-R + SLS (Zhang et al., ACM MM 2024) | `xlsr-sls.onnx` (`none-stated`/MIT card, **origin-only**) | ONNX Runtime (`onnxruntime`, `VERITY_ORT_FORCE_CPU=1` optional); strong in-the-wild DF performance |
| `rawnet2` | RawNet2 + RawBoost (Tak et al., ICASSP 2021/2022) | `Best_model.pth` (`MIT`, mirrored) | Mel sinc filterbank + FMS ResBlocks + 3-layer GRU on 64,600-sample windows; ASVspoof baseline |
| `aasist` | AASIST (Jung et al., ICASSP 2022) | `AASIST.pth` (`MIT`, mirrored) | SincConv(70) + graph attention; **caveat**: 2019 ASVspoof LA baseline (low in-the-wild AUC 0.43), shipped strictly as baseline cross-check |

---

## Reviewed and intentionally omitted repositories

Surveyed architectures were excluded when: (a) weights live behind unscriptable
passwords (USTC RecDrive / Baidu for AltFreezing & DIRE), (b) no trained detector
checkpoint exists (`DID`, `TALL4Deepfake`, `HiFE`, `DCL`, `OSDD`, `RFFR`),
(c) 7B–13B LLaVA+SAM pipelines (`SIDA`, ~30 GB) exceed isolated detector envs,
or (d) scope mismatches passive forensic detection (proactive watermarking `LIDMark`,
speaker-ID `Jungjee/RawNet`, or LLM benchmark `FVBench`).

## Recommended investigator panels

| Investigation Goal | Recommended Evidence-Grade Detectors |
|---|---|
| Face swap / reenactment (video) | Auto-selected union (`xception`, `efficientnetb4`, `core`, `f3net`, `spsl`, `meso4`, `sbi`, `laa_net`, `prodet`, `m2tr`, `df40_clip`, `effort`, `forensics_adapter`, `vlaforge`, `lipforensics`, `ftcn`, `realforensics`) |
| Lip-sync / dubbing check | `lipfd` + `syncnet` (inspect raw AV offset & LSE-C confidence) + `lipforensics` |
| AI-generated still image | `univfd` + `gend` + `community_forensics` + `npr` + `mlep` + `safe` + `d3` + `aeroblade` |
| Voice clone / TTS audio | `antideepfake_mms` + `xlsr_sls` (with `aasist` & `rawnet2` as ASVspoof baselines) |

## Related pages

- [Shared detector engine](../architecture/detector-engine.md) — preprocessing, CAM, and `aux_aggregate` contracts.
- [Detector selection and trust tiers](../architecture/detector-selection.md) — automated inclusion/exclusion rules.
- [Weights and checkpoints](../architecture/weights-and-checkpoints.md) — SHA-256 verification and `ADPer/verity-weights` attribution.
- [Explainability artifacts and on-disk formats](explainability-artifacts.md) — artifacts emitted by each detector category.
