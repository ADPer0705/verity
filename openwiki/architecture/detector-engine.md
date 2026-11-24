---
type: subsystem
title: Shared detector engine
description: The common inference engine in detectors/common/ covering deterministic seeding, all six input modes, Grad-CAM vs native attention, auxiliary detector signals, reasoning synthesis, honest-degradation guards, and the result.json contract.
tags: [detectors, inference, pytorch, explainability, determinism, grad-cam]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-6244384b2ac289d182ad2fe8
    resource: repo://detectors/common/cam.py
  - id: openwiki-source-2fbc661d38d1c9f143c7b320
    resource: repo://detectors/common/engine.py
  - id: openwiki-source-dfa9bb6a866c9480a8673565
    resource: repo://detectors/common/face_utils.py
  - id: openwiki-source-42afeb14ea690707df084262
    resource: repo://detectors/common/reasoning.py
  - id: openwiki-source-62904f50f1b61324a8e6f018
    resource: repo://detectors/common/sampling.py
  - id: openwiki-source-a4e8a27770b21d3f8160d259
    resource: repo://detectors/common/viz.py
  - id: openwiki-source-490ab6995a830adcd6f58749
    resource: repo://detectors/recce/inference.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Shared detector engine

Every detector port under `detectors/<id>/` is a thin package built around four
files: `detector.yaml`, `pyproject.toml`, `model.py`, and `inference.py`. Aside
from bespoke exceptions such as RECCE's decoder-audit entrypoint, `inference.py`
delegates entirely to `detectors/common/engine.py`:

```python
from common.engine import run_detector
from model import build_model

run_detector(build_model, Path(__file__).with_name("detector.yaml"))
```

Centralizing media decoding, face detection, deterministic frame sampling,
normalization, auxiliary tensor extraction, Grad-CAM/attention rendering, and
`result.json` serialization in `detectors/common/` ensures two critical forensic
invariants: (1) all 58 detector ports apply identical sampling and coordinate
conventions, and (2) raw upstream logit ordering quirks (e.g. RawNet2 / XLS-R+SLS
putting `spoof` at index 0) are normalized in each `model.py` so index 1 always
means fake before hitting the engine.

## Specification loading

`detectors/common/spec.py` loads each detector's `detector.yaml` into a
`DetectorSpec` dataclass defaulting to the DeepfakeBench image baseline
(`256x256` RGB, mean/std `(0.5, 0.5, 0.5)`). `load_spec()` honors:

- `input_mode` (`face_crop`, `face_clip`, `full_frame`, `full_clip`, `audio`, or
  `av_clip`, inferring `face_clip` when `video_mode: true` if omitted);
- `face_margin` (for detectors trained on expanded face boxes such as SBI and
  LAA-Net, which require a 25% total box margin);
- `resize_short` and `center_crop` (whole-frame preprocessing);
- `cam` (default `True`; when explicitly `False`, pixel-space Grad-CAM fallback is
  disabled by design — used for waveform audio models, wavelet-space `safe`,
  distance-based `aeroblade`, frozen feature probe `univfd`, and temporal transformers
  `timesformer`, `videomae`, `xclip`, `tall` whose inputs/heads lack meaningful
  spatial Grad-CAM surfaces);
- `region_analysis` (defaults to `True` only for `face_crop`/`face_clip` when `cam`
  is enabled);
- `max_frames` (overridable via `VERITY_MAX_FRAMES`, where `0` disables the cap);
- `caveat` (prepended verbatim to `confidence_factors` whenever known limitations
  exist, such as `m2f2_det`'s omitted bridge adapter blocks or `aasist`'s
  ASVspoof2019 generalization boundary).

## Determinism and bounded memory

Before loading or running any model, `detectors/common/engine.py` fixes:

```python
SEED = 0xC0FFEE
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Video decoding uses `VideoReader` (`detectors/common/sampling.py`), which plans the
evenly spaced target frame positions (and neighboring `clip_size` windows in clip
mode) up front via `_plan()`, then performs a single **sequential** `cv2.VideoCapture`
pass retaining only needed frames in memory. Avoiding random seeks prevents
decoder keyframe-drift from shifting frame selection across runs, while keeping only
sampled frames avoids exhausting RAM on long 1080p videos.

## Input modes

`DetectorSpec.input_mode` controls what `run_detector` extracts, scores, and names as
its `unit`:

| `input_mode` | Consumed input | Scored `unit` | Key behavior |
|---|---|---|---|
| `face_crop` | Face crops per sampled frame | `face` | MTCNN `FaceDetector` (`min_face` filter) + `_expand_box(face_margin)` |
| `face_clip` | `clip_size` consecutive face crops | `face` | `VideoReader.clip` + `crop_rgb` stacked as `(1, 3, T, H, W)` |
| `full_frame` | Whole image or evenly sampled frames | `frame` | Calls optional `model.preprocess(pil)` else resize/center-crop/normalize |
| `full_clip` | `clip_size` consecutive whole frames | `clip` | Stacked `(1, 3, T, H, W)` without face cropping |
| `audio` | 16 kHz mono waveform windows | `window` | Prefers extracted `artifacts/audio/audio.wav`, else ffmpeg; tiles short audio |
| `av_clip` | `clip_size` frames + matching audio span | `window` | Streams frame ranges once; passes window dict to `model.forward_detailed` |

If `mode` is `audio` or `av_clip` and no decodable soundtrack exists, `AudioUnavailable`
is caught cleanly so `result.json` records the audio gap instead of crashing the
examination.

## Native attention, Grad-CAM, and auxiliary signals

After scoring via `model.forward_detailed(x)`:

1. **Native vs Grad-CAM (`_cam_for`)**: The engine inspects `out.get("attention_map")`
   first. If present and spatially non-constant (`std > 1e-6`), it min-max normalizes
   the native map (for example LAA-Net's 96x96 boundary heatmap, Forensics Adapter /
   VLAForge / DF40 ViT patch attention, SRM spatial attention, or ProDet's gated
   attention mask). Otherwise, if `spec.cam` is `True`, it invokes `grad_cam(model, x)`
   using the `feat` tensor returned by `forward_detailed`.
2. **Region-energy attribution**: When MediaPipe `FaceMesh` succeeds and
   `spec.region_analysis` is `True`, attention energy is projected onto six facial
   regions (`left_eye`, `right_eye`, `nose`, `mouth`, `jaw`, `forehead`) and stored in
   `per_face[*].region_energy` and `region_aggregate`.
3. **Auxiliary sub-scores (`_extract_aux` & `aggregate_aux`)**: Models that compute
   interpretable secondary measurements return them in `out["aux"]` (or per-window
   `out["info"]` in AV mode) — for example ProDet's blend/identity heads, VLAForge's
   query vs VLA-gated scores, AEROBLADE's per-VAE LPIPS distances (SD1.1, SD2.1,
   Kandinsky), or SyncNet's AV offset and LSE-C confidence. The engine rounds each
   scalar to `per_face[*].aux`, aggregates dataset-wide means into `aux_aggregate`,
   and surfaces them in `confidence_factors` via `build_reasoning`.

## RECCE honest-degradation audit

`detectors/recce/inference.py` exemplifies how Verity handles checkpoints with dead or
untrained sub-modules. The official DeepfakeBench v1.0.1 `recce_best.pth` checkpoint
leaves the reconstruction decoder untrained (producing near-constant ~0 tensors). At
model load time, `audit_reconstruction(model)` runs a synthetic gradient probe and
checks whether the reconstruction tensor varies (`std > 1e-4` and `min < max`).
When `decoder_alive` is `False`, RECCE explicitly omits `reconstructions/*_recon.png`,
drops reconstruction residual claims from `region_energy`, records the audit telemetry
in `tensor_audit`, and explains in `reasoning` and `confidence_factors` that the
verdict rests solely on classifier logits and guided attention maps.

## Aggregation, reasoning, and `result.json`

`detectors/common/reasoning.py` computes deterministic panel statistics across all
scored units in `per_face`:

- `aggregate` computes `mean_p`, `confidence = 1.0 - min(std_p * 2.0, 0.9)` (so
  cross-frame/cross-face variance lowers confidence automatically), and `stats`
  covering `mean`, `min`, `max`, `std`, `median`, `count`, `threshold`,
  `above_threshold`, `below_threshold`, `high_confidence_fake` (`> 0.7`), and
  `high_confidence_real` (`< 0.3`).
- `build_reasoning` formats an explicit `FAKE` (`> 0.7`), `REAL` (`< 0.3`), or
  `UNCERTAIN` (`0.30–0.70`) summary plus itemized `confidence_factors` covering weight
  provenance, CAM availability, top region energy, `aux_aggregate`, unit count, and
  execution device.

`run_detector` writes:

- `result.json` with `prediction`, `confidence`, `reasoning`, `confidence_factors`,
  `stats`, `weights`, `timing_seconds`, `device`, `model`, `unit`, `input_mode`,
  `per_face`, `audio`, `total_video_frames`, `sampling`, `region_aggregate`,
  `aux_aggregate`, and `execution_context`;
- `per_face_results.json` mirroring `per_face`;
- Explainability images (`faces/`, `frames/`, `attention_maps/`, `full_frames/`, and
  `timeline.png`).

## Related pages

- [Detector runtime and isolation](detector-runtime.md) — how `DetectorRunner` launches
  and parses the engine's outputs.
- [Detector catalog (all 58 ports)](../reference/detector-catalog.md) — per-detector
  input modes, CAM flags, auxiliary signals, and caveats.
- [Explainability artifacts and on-disk formats](../reference/explainability-artifacts.md) —
  full file layout and evidence contract.
