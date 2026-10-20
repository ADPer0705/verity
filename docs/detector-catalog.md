# Detector catalog

A field guide to the 47 architectures ported under `detectors/`: what each
one actually looks for, what media it can consume, what explainability
artifacts it produces, what to watch out for, and where the checkpoint comes
from. The generated table (tier, modality, weights, license) is in
[detectors.md](detectors.md); this page explains the *how and why* per
detector. Generated facts live in `detectors/<id>/detector.yaml` +
`README.md`; this catalog hand-summarizes them.

How to read the trust tiers:

- **trained** — released checkpoint with an actually-trained classification
  head, or a training-free method with published statistics. Evidence-grade;
  auto-selected by default.
- **backbone-only** — a pretrained backbone (ImageNet/Kinetics) with an
  *untrained* head. Research use; findings flagged, never score-bearing.
- **random-init** — no released checkpoint; weights are random. Research
  scaffold for plugging in your own head/checkpoint.
- **untrained** (external fetch, head untrained) — a foundation model fetched
  at first run whose detector head is random.

Common pipeline facts (apply to every port unless noted):

- All face inputs come from MTCNN detection by the shared engine
  (`detectors/common/engine.py`), cropped without margin and resized to the
  detector's `resolution`; normalization comes from `detector.yaml`.
- Every detector emits `result.json` (`prediction`, `confidence`,
  `reasoning`, `confidence_factors`); per-face video results additionally
  produce a `timeline.png` and, where possible, attention/Grad-CAM overlays.
- Upstream logit orders vary; the port normalizes so **index 1 = fake**
  everywhere. Where the upstream behaves unusually, it is noted.
- Each detector runs in its own uv environment declared by
  `detectors/<id>/pyproject.toml` — torch versions and dependencies are
  pinned per detector and cannot conflict.

The weights/licensing facts behind every checkpoint: `detectors/weights_manifest.yaml`
(+ `verity weights status`), and [weights.md](weights.md) for the machinery.

## Image & video face detectors (FF++-era CNN family)

All consume **face crops** (`input_mode: face_crop` default), trained on
FaceForensics++ c23 via DeepfakeBench (CC BY-NC 4.0 checkpoints unless
stated), and produce Grad-CAM attention maps + per-face probabilities +
timelines for videos. Shared caveats: FF++-era models are ~2019/2020
technology — strong on classic low-quality video manipulations, weaker on
modern photorealistic, full-synthesis or heavy-recompression inputs; treat
them as a complementary panel, not a single oracle.

### `xception` — Xception (Rosler et al. 2019)
The classic baseline; Xception CNN + linear head on 256×256 face crops, FF++ c23
trained. Reads color and blend-boundary artifacts. Grad-CAM `cam: true`. The
reference point of the panel: when it disagrees with the newer ones, that
disagreement itself is a finding to investigate.

### `efficientnetb4` — EfficientNet-B4
Stronger CNN backbone than Xception, same FF++ recipe. Grad-CAM through the
convolutional trunk.

### `resnet34` (`cnnaug_best.pth`) — CNN-Augmentation, ResNet-34
ResNet-34 trained with CNN-augmentation kernels. Same artifact family as
Xception/B4.

### `meso4` / `meso4Inception` — MesoNet / MesoInception (Afchar et al. 2018)
Deliberately tiny CNNs (~120 KB checkpoints) tuned for *mesoscopic* features
of face regions — cheap, fast, surprisingly robust on compressed video.
MesoInception swaps in Inception blocks with wavelet-inspired decomposition.
Grad-CAM: yes.

### `capsule_net` — Capsule (Capsule-Forensics V2, Hashmi et al. 2021)
VGG19 features + capsule network routing, FF++ c23. Produces Grad-CAM maps.
Caveat noted upstream: capsule models can over-commit to their primary class
on adversarial/degraded input — pair it with a spectrum- or frequency-based
detector before concluding.

### `f3net` — F3Net (Qiu et al. 2020)
Xception backbone + FAD (frequency artifact decomposition) head over 12
frequency channels. Useful complement when a forgery is *invisible in color
space* but visible as spectral peaks; the port infers the DCT-domain
branches in torch.

### `spsl` — SPSL (Liu et al. 2021)
Xception + phase-spectrum (4-channel DCT phase) head. Phase anomalies track
with upsampling-style synthesis — the go-to pair with F3Net when hunting GAN
fingerprints on faces.

### `srm` — SRM (DeepfakeBench)
Dual Xception encoders (RGB + SRM-filter-kernel branch) with cross-modal
attention, FF++ c23. The model returns a native spatial attention map, and
Grad-CAM is enabled as a fallback (`cam: true`) — the engine prefers the
native map.

### `ffd` — FFD (frequency-aware forgery detector, Li et al., ECCV 2020)
Dual-branch Xception variant with a fine-grained frequency-aware head and
mask regression. The mask channel is part of the checkpoint; the port runs
the classification path.

### `core` — CORE (Zhao et al. 2021)
Xception backbone + consistency-aware dual encoder/decoder head, FF++ c23.
Grad-CAM from the backbone.

### `sbi` — SBI (Shiohara & Yamasaki 2022, self-blended images)
EfficientNet-B4 + self-blended-image augmentation. **No usable published
checkpoint** (upstream weights exist, license forbids redistribution) — ships
random-init for experimentation. **Drop an SBI checkpoint into
`detectors/sbi/weights/` and it becomes evidence-grade.**

### `ucf` — UCF (DeepfakeBench)
Dual Xception encoders + fingerprint reconstruction head; trained on FF++ c23.
Grad-CAM yes.

### `recce` — RECCE (Cao et al. 2022)
ResNet features + **reconstruction-classification** cooperative learning: a
decoder reconstructs the face image, the reconstruction *error* map
highlights manipulated regions and feeds both the classifier and the
**native attention map**. FF++ c23 checkpoint (`recce_best.pth`).

**Honest-degradation audit.** The released DeepfakeBench v1.0.1 checkpoint
ships with a decoder that maps every input to a near-constant ~0 — an
*untrained* reconstruction head. Verity audits decoder liveness at load
(`audit_reconstruction` in `inference.py`): when dead, reconstruction
visualizations and any claim based on them are omitted and the finding's
`confidence_factors` say so. The verdict itself remains valid — it rests on
the trained classifier head and the attention map. This is the template of
how Verity handles a checkpoint with a broken/absent sub-module.

### `facexray` — Face X-ray (Li et al. 2020) — *backbone-only*
Blending-boundary detector for face swaps; HRNet-W48 backbone (MIT). No
released detector head in Verity — the model uses the ImageNet HRNet weights
with an untrained head, so it is **not** evidence-grade as shipped. Provide a
trained Face X-ray checkpoint to upgrade it. Grad-CAM available.

### `iid` — IID (implicit identity leakage) — *backbone-only*
Xception + an iresnet50 identity branch (Intertwined Identity Detection).
Only the ImageNet Xception backbone is wired in; the identity branch needs
InsightFace weights (`backbone.pth`, MS1MV3-ArcFace) which are not fetched.
Research scaffold, not evidence-grade as shipped.

## Audio-family detectors

Common facts: 16 kHz mono waveform extracted by ffmpeg, scored in fixed
windows (4.0375 s windows with 2 s hop for the fixed-64,600-sample models; 4 s
windows for the MMS one), tile-padding when shorter. Artifact for every one:
`timeline.png` — **per-window fake probability**, so a splice boundary or a
concatenated clone shows as a step, not just a mix. `cam` never applies;
there is no image-space attention. Labeled `targets: audio_synthesis`.

### `aasist` — AASIST (Jung et al., ICASSP 2022)
Raw-waveform anti-spoofing: learned SincConv filterbank (70 filters), 6
residual blocks, heterogeneous spectro-temporal graph attention. Official
checkpoint (MIT, ASVspoof2019 LA). Windows of 4.0375 s at hop 2 s, fixed
(64,600-sample) input; `timeline.png` shows per-window score.
**Caveat (records in its own reasoning):** 2019-era LA model; generalises
poorly to modern in-the-wild deepfake audio (Deepfake-Eval-2024 AUC 0.43).
Use as a *baseline*, not a headline.

### `antideepfake_mms` — AntiDeepfake MMS-300M (Ge et al., ASRU 2025)
wav2vec 2.0 speech foundation model (MMS-300M, 24-l, 1024-d) post-trained on
**56,370 h real + 18,280 h fake speech in 100+ languages** — the strongest
multilingual prior Verity ships. Official HF checkpoint
(`nii-yamagishilab/mms-300m-anti-deepfake`, CC BY-NC-SA 4.0; the fairseq
state dict is remapped onto `transformers.Wav2Vec2Model` at load time.
Windows of 4 s at hop 2 s; timeline over spectrogram.
`VERITY_ANTIDEEPFAKE_REPO` selects the 1.2 GB XLS-R-1B sibling
(`xls-r-1b-anti-deepfake`, 3.86 GB) instead.

### `xlsr_sls` — XLS-R + SLS (Zhang, Wen & Hu, ACM MM 2024)
XLS-R-300M front-end with **Sensitive Layer Selection** per-layer gating +
BN/SELU/max-pool + FC(22847, 1024) + FC(1024, 2). Runs as **ONNX via
onnxruntime** (CUDA EP when available, else CPU; `VERITY_ORT_FORCE_CPU=1`
pins CPU). Official checkpoint opset-17 export
(`SpeechAntiSpoofingBenchmarks/XLSR-SLS` — card says MIT; upstream repo ships
no LICENSE, so redistribution is treated as uncertain and the file is always
fetched from that HF repo, never mirrored). **Strong 2021-era benchmark
numbers** (ASVspoof2021 DF EER 1.9 %, In-the-Wild 7.5 %) and noticeably
better on modern generators than AASIST; log-softmax [spoof, bonafide]
swapped so index 1 = fake.

## AI-generated-image detectors

These consume **whole frames/images** (never face crops — no face required),
with the detector's own preprocessing (`resize_short`/`center_crop`/native)
declared per detector. For videos, up to `max_frames` evenly spaced frames
are scored and aggregated with a per-frame timeline. Attention/CAM semantics
differ (see notes below). Targets cover `image_synthesis` and/or
`face_synthesis`.

### `univfd` — UniversalFakeDetect (Ojha et al., CVPR 2023)
The "low-classifier-signal idea made practical": score a **frozen CLIP
ViT-L/14** feature space with a linear probe (768→1). MIT weight file
(`fc_weights.pth`, 4 KB) + auto-downloaded CLIP backbone. Exactly the
*opposite* licensure of most ports — but with the caveat that feature probes
need the same preprocessing as upstream (`real-guidance`-style CLIP
features). Grad-CAM: not applicable (no per-class spatial head on the
classifier); per-frame timeline for videos.

### `gend` — GenD (WACV 2026)
CLIP ViT-L/14 with **LayerNorm-layer fine-tuning** and an L2-normalized
linear probe. Released detector (`yermandy/GenD_CLIP_L_14`, HF, MIT). The
published model is one of the strongest general detectors of Verity's crop
for *entirely synthesized imagery* (StyleGAN, diffusion, Midjourney, Flux).
`cam: true` through the ViT — attention over the patch grid; for videos,
face-crop based (face-crop input mode) with per-face attention overlays.

### `community_forensics` — Community Forensics (Park & Owens, CVPR 2025)
ViT-S/16 at 384 px, fine-tuned end-to-end on a corpus of **~4,800 generators**
(Community Forensics dataset). The point of the architecture is *breadth*:
instead of guessing the generative trick, learn what "generated-ness" looks
like across an enormous generator population. MIT (authors' HF repo). Single
BCE logit, label 1 = fake; the port emits `[0, z]` so `softmax[:, 1]` is the
fake probability exactly. Grad-CAM over the 24×24 patch-token grid entering
the last transformer block + per-frame timeline for videos.

### `npr` — NPR (Tan et al., CVPR 2024)
**N**eighbourhood **P**ixel **R**atio — tracks the pixel-pair ratio rather
than pixel values, so it generalizes across GAN up-sampling artifacts.
Truncated ResNet-50 (conv1 + layer1/layer2) over the NPR residual input,
1-logit BCE head. Weight has *no license statement* upstream (`none-stated`)
→ **origin-only fetch, never mirrored.** Whole-frame input at native
resolution, ImageNet normalization inside the model. Grad-CAM `cam: true`.

### `mlep` — MLEP (Yuan et al., NeurIPS 2025)
Multi-granularity **local entropy patterns** (9-channel multi-scale entropy
map) fed to a truncated ResNet-50 with a 2-class head. Weight *no license
statement* upstream (`none-stated`) → **origin-only, never mirrored.**
Whole-frame native input with the model's own ImageNet normalization
(yaml mean/std neutral). Grad-CAM `cam: true` on the entropy-map CNN — the
report shows the map on the *entropy* representation, which is still
examiner-interpretable.

### `safe` — SAFE (Li et al., KDD 2025)
One-level **bior1.3 wavelet HH sub-band** (diagonal high-freq) of the image,
classified by a tiny truncated ResNet-50 (~1.4 M params). The thesis is that
*resizing destroys the exact high-frequency evidence* synthetic-image
detectors rely on, so the official evaluation (and this port) does a
**256×256 center crop at native resolution with no resize and no
normalization** — meaning large frames are examined only in their central
256×256 window, which is *the official evaluation behavior*, noted in the
report reasoning. Apache-2.0. `cam: false` on purpose — a CAM over a wavelet
sub-band would localize wavelet energy, not image content; per-frame timeline
instead.

### `aeroblade` — AEROBLADE (Ricker et al., CVPR 2024) — *training-free*
For each image, first reconstruct with **three latent-diffusion
autoencoders** (SD1.1, SD2.1-base, Kandinsky 2.1) and measure LPIPS
(VGG-2) reconstruction distance. **The minimum across autoencoders is what
decides**: real content should be reconstructed poorly by *every* encoder,
while latent-diffusion-generated content reconstructs *very well* by its own
(and near-well by the others in that family). No training at all — the
checkpoint is the model weights of the three networks; `weights.kind` is
`external`, all fetched from the HF hub at first run (SD2.1-base is gated
upstream: the port falls back to a verified un-gated mirror of the same
weights; OAI/CompVis OpenRAIL licenses listed in the manifest). For videos,
8 frames minimum-error-aggregated. **No attention map by design** — it's a
distance, not a classifier; the reasoning states this. Entire-video
detection: works on synthesized-image-style videos, not temporal-generated
ones.

### `df40_clip` — DF40 CLIP-L/14 (Yan et al., NeurIPS 2024 D&B)
Full CLIP ViT-L/14 **vision tower fine-tuned for deepfake detection** on the
DF40 benchmark (face-swap + face-reenactment + entire-face-synthesis
families). Default `train_on_df40/clip_large.pth` (per-family alternates
`train_on_fs`, `train_on_fr`, `train_on_efs` are drop-in via
`detector.yaml::weights.file`). SOTA-class quality on face-targeted
manipulations. CC BY-NC 4.0 (checkpoint lives on the authors' Google Drive —
upstream requests no re-hosting, so origin-only). Backbones fetched fresh
(`openai/clip-vit-large-patch14`, MIT) only to *instantiate* the
architecture, then fully overwritten. Explainability: last-layer CLS
**attention over the 16×16 patch grid** overlay (no gradient CAM through the
fine-tuned ViT), facial-region energy, prediction timeline, most-suspicious
frames.

### `effort` — Effort (Yan et al., ICML 2025)
CLIP ViT-L/14 where every self-attention projection carries an
**orthogonal (SVD) rank-residual decomposition** — a lighter-touch
fine-tuning that generalizes better than full fine-tuning on FaceForensics++
c23. The shipped checkpoint (FaceForensics++-trained) contains the full tower
with the SVD residuals (681 keys, strict loading). CC BY-NC 4.0, Drive
origin-only. Explainability: face-crop input + timeline; no attention map
documented — like DF40, the last-layer CLS attention is not surfaced (the
upstream does not expose it), so Verity's per-face fake probability +
timeline are the evidence channel.

Toggle: `effort_clip_L14_trainOn_GenImage_sdv14.pth` /
`effort_clip_L14_trainOn_Chameleon.pth` are documented alternates for
`image_synthesis` targets via `weights.file`.

## Video (temporal, face-clip) detectors

Consume **full face-tracked clips** (`input_mode: face_clip`,
`video_mode: true`): `clip_size` consecutive per-face crops, so genuinely
temporal architectures can look at both spatial *and* frame-to-frame
patterns. FF++-based training unless noted.

### `lipforensics` — LipForensics (Haliassos et al., CVPR 2021)
Conv3d frontend + ResNet-18 + MultiscaleMultibranchTCN over **mouth-region
crops** (25-frame clips at 25 fps): lip-sync artifacts. Official FF++-trained
checkpoint (MIT). One of only two Verity detectors that target `lip_sync`
(asynchronous voice/dubbing). `timeline.png` over the mouth sequences.

### `ftcn` — FTCN (Zheng et al., 2021) — *backbone-only*
Face-Temperal Consistency: I3D 3D-ResNet-50 + temporal-coherence head.
Only the Kinetics-pretrained I3D backbone ships (CC BY-NC 4.0) — head is
untrained, so this is a research scaffold. 16-frame clips. Grad-CAM
`cam: true` (the backbone surface is live even without a trained head).

### `altfreezing` — AltFreezing (2020) — *backbone-only*
3D-ResNet-50 trained by alternating freezing (RGB branch then 3D branch).
Same I3D R50 backbone, untrained head: research scaffold. 16-frame clips.

### `i3d` — I3D (plain video backbone) — *backbone-only*
The raw 3D-ResNet-50 (Kinetics) backbone, no head. Shared file
(`I3D_8x8_R50.pth`) with ftcn/altfreezing (hardlinked on pull). The base
model for anyone implementing a new video detector.

### `timesformer` — TimeSformer — *untrained (external fetch, head random)*
facebook/timesformer-base-finetuned-k400 encoder (CC BY-NC 4.0) with an
untrained classification head. Research scaffold: the encoder is live, the
verdict is not evidence.

### `videomae` — VideoMAE — *untrained (external fetch, head random)*
MCG-NJU/videomae-base (CC BY-NC 4.0) with untrained `fc_norm`+head. Research
scaffold for video-news tasks — plug your own head.

### `xclip` — X-CLIP — *untrained (external fetch, head random)*
microsoft/xclip-base-patch32 (MIT video tower) + untrained head. Research
scaffold.

### `stil` — STIL — *random-init, nothing fetched*
SCNet50-v1d + spatiotemporal-inconsistency head, entirely random-initialized
(upstream's SCNet ImageNet init is referenced but not implemented — stated
in `detector.yaml`). Architecture-study scaffold.

### `tall` — TALL — *random-init*
Swin thumbnail-layout transformer (temporal), fully random init. Scaffold.

## Image-face & video-frame research scaffolds (random-init / backbone-only)

One paragraph each — they exist for researchers to drop in their own
checkpoint (set `trained: true` after placing the file), not as evidence:

- **`sbi` (EfficientNet-B4 + self-blended images)** — directly usable once a
  checkpoint is provided: upstream SBI weights exist but their license
  forbids redistribution, so Verity ships it random-init and waits.
- **`sladd` (Xception + adversarial synthesizer)** — scaffold. Upstream
  describes the adversarial self-synthesis pipeline but ships no detector
  checkpoint — the port keeps the architecture; provide your own checkpoint.
- **`uia_vit` (ViT-B/16 + UIA inconsistency-attention head)** — fully random
  init; the ImageNet ViT initialisation upstream mentions is not fetched.
- **`fwa` (DSP-FWA, Xception + face-warp artifacts)** — scaffold; the classic
  "detect face warping from GANs" idea, in DSP-FWA form.
- **`lrl` (EfficientNet-B4 dual-encoder RGB+DCT)** — scaffold from the LRL
  paper recipe.
- **`lsda` (EfficientNet-B4 + latent space augmentation)** — scaffold.
- **`multi_attention` (EfficientNet-B4 + multi-attentional head)** — scaffold.
- **`pcl_xception` (Xception + pairwise consistency / PCL-I2G)** — scaffold.
- **`rfm` (Xception + representative forgery mining)** — scaffold.
- **`sia` (EfficientNet-B4 + attention-driven head, SIA)** — scaffold.
- **`clip` (CLIP ViT-B/16, openai weights + random 2-way head)** — scaffold;
  the *fetched* weights are the CLIP image tower, the head is random.
  (Distinct from `univfd`/`gend`/`df40_clip`/`effort`, which all load
  **trained** head weights.)

## Trimmed port coverage (per-arch tally)

- **Face-crop ports (29: 17 trained + 12 research):** xception,
  efficientnetb4, resnet34, meso4, meso4Inception, capsule_net, f3net, spsl,
  srm, ffd, core, ucf, recce, gend, df40_clip, effort (trained) · sbi, fwa,
  lrl, lsda, multi_attention, pcl_xception, rfm, sia, sladd, facexray, iid,
  clip (research).
- **Full-frame / AI-generated-image ports (6, all evidence-grade):** univfd,
  community_forensics, npr, mlep, safe, aeroblade (training-free).
- **Video-temporal face-clip ports (9: lipforensics trained, 8 research):**
  lipforensics · ftcn, altfreezing, i3d, stil, tall, timesformer, videomae,
  xclip.
- **Audio ports (3, all trained):** aasist, antideepfake_mms, xlsr_sls.

## Choosing a set (how the selector and examiners usually run)

| Goal | Recommended trained panel |
|---|---|
| Face swap / reenactment (video) | `xception` + `efficientnetb4` + `core` + `f3net` + `spsl` + `meso4` + `lipforensics` + `df40_clip` + `effort` |
| AI-generated still image | `univfd` + `gend` + `community_forensics` + `npr` + `mlep` + `safe` + `aeroblade` |
| Voice clone / TTS | `antideepfake_mms` + `xlsr_sls` (AASIST as the 2019 baseline cross-check) |
| Whole-video coverage | the union — plus the report's disagreement view |

Default (`verity run`, no flags) auto-selects every trained detector whose
media requirements and weights are satisfied; the cases where a detector is
*excluded* are recorded in `selection.json` with the exact reason. Nothing is
picked by vibes, and removing a detector never mis-reports the case —
`--detector` names exactly what should run.

## Common honest caveats (repeated across entries so no one misses them)

- FF++-era checkpoints are 2019–2021 technology; treat low-quality, deeply
  compressed, or novel-generator inputs as *least-trust* zones and rely on
  cross-detector agreement, not a single model.
- `none-stated` licenses (NPR, MLEP) mean the upstream gave no permission
  statement at all: Verity never re-hosts those files; they are fetched
  from the authors' own release points, every time.
- Every audio model is a *speech-spoofing* model — it does not judge a
  speaker's identity; it rates synthetic-ness.
- Report conclusions are per-examen findings shown with full artifacts: the
  examiner decides, Verity organizes the evidence.
