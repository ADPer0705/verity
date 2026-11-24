---
type: subsystem
title: Detector selection and trust tiers
description: Deterministic rule-ordered eligibility checks, weight trust tiers separating the 39 evidence-grade models from the 19 research scaffolds, explicit overrides, and manipulation-family coverage reporting.
tags: [detectors, selection, trust, taxonomy, evidence]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-df1d7602f907026105e434b2
    resource: repo://src/verity/analyze/detector_selector.py
  - id: openwiki-source-2b146e51d74f7afd2ccd960f
    resource: repo://src/verity/analyze/models.py
  - id: openwiki-source-e4dc7d4c95cc68eb2f4b4656
    resource: repo://src/verity/cli.py
  - id: openwiki-source-0aae5504aa2409b43a281345
    resource: repo://src/verity/detect/base_detector.py
  - id: openwiki-source-539be2a80a0bea12a949fc4c
    resource: repo://src/verity/models/taxonomy.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Detector selection and trust tiers

Running every model blindly on every exhibit produces noise, wastes GPU hours,
and breaks chain-of-custody clarity. Instead, `DetectorSelector`
(`src/verity/analyze/detector_selector.py`) evaluates every registered detector
definition (`DetectorDefinition` in `src/verity/detect/base_detector.py`) against
deterministic facts from `MediaAnalysis`, recording an explicit
`SelectionDecision` (`included` + human-readable `reasons`) in sorted ID order.

The persisted `selection.json` guarantees a supervisor or opposing expert can
always answer both *which detectors ran* and *why any of the remaining 58
architectures were excluded*.

## Engineering rationale: fail-fast ordered eligibility

Why apply ordered stop-on-first-failure rules rather than a loose scoring rubric?

1. **Rule zero prevents untrained noise from masquerading as evidence.** Verity
   ports **58 architectures**, of which **39 are evidence-grade (`trained`)** —
   either released deepfake checkpoints (`kind: official` or `external` with
   `trained: true`) or training-free methods (`aeroblade`, `syncnet`). The other
   **19 ports** are `backbone-only`, `random-init`, or `untrained` scaffolds
   retained for researchers bringing their own weights. Running random heads by
   default would sit uninformative ~0.5 noise next to real findings.
2. **Pre-filtering avoids wasted subprocess startups.** Evaluating media type,
   audio presence, face presence, minimum face resolution, clip duration, and
   local checkpoint presence before launching any `uv` subprocess prevents
   spawning isolated Python environments only to abort with zero usable inputs.
3. **Offline integrity is enforced before launch.** When `VERITY_OFFLINE=1` (or
   `offline: true` in `<config_dir>/config.yaml` after `verity setup --airgap-ready`),
   selectors exclude uncached external/runtime-fetched models deterministically
   with `"offline mode; weights not cached locally"`.

## Inputs

`DetectorSelector.select(analysis)` iterates `sorted(registry.list_detectors())`
and combines:

- `MediaAnalysis.media` (`media_type`, `has_audio_track`, `face_present`,
  `face_count`, `max_face_size`, `sampled_frames`, `frame_count`,
  `duration_seconds`, `fps`);
- Each `DetectorDefinition`'s derived `capabilities` (from `input_mode` /
  `artifacts.required`), `min_face`, `clip_size`, and `weights.tier`;
- Construction flags `offline` (defaulting from `VERITY_OFFLINE`) and
  `include_untrained` (defaulting from `VERITY_INCLUDE_UNTRAINED`).

## The six ordered rules

Rules execute in fixed order (`_decide`), stopping on the first failure:

### 0. Trained classification head

Detectors where `not definition.weights.is_trained` are excluded unless
`include_untrained` is set (`--include-untrained` or explicit `--detector`
override):

> `<tier> head: not evidence-grade; excluded by default (use --include-untrained to run it anyway)`

When permitted via `--include-untrained` or override, the decision reason notes
`not evidence-grade` and `Finding.tier` ensures consensus and scores ignore it.

### 1. Media-type compatibility

`DetectorCapabilities.accepts(media_type)` derives supported modalities from
`input_mode`:

| `effective_input_mode` | `media_types` | `faces_required` | `audio_required` |
|---|---|---|---|
| `face_crop` | `["image", "video"]` | yes | no |
| `face_clip` | `["video"]` | yes | no |
| `full_frame` | `["image", "video"]` | no | no |
| `full_clip` | `["video"]` | no | no |
| `audio` | `["audio", "video"]` | no | yes |
| `av_clip` | `["video"]` | yes | yes |

### 1b. Audio track presence

If `caps.audio_required` and `media_type == "video"` with `not media.has_audio_track`:

> `requires an audio track; the video has none`

### 2. Face presence

If `caps.faces_required` and `not media.face_present`:

> `requires faces; none detected in <N> sampled frames`

### 3. Face scale

If `caps.faces_required`, `min_face > 0`, and `media.max_face_size < definition.min_face`:

> `faces too small: largest <px>px < required min <min_face>px`

### 4. Temporal sufficiency (video-only detectors)

For detectors with `set(caps.media_types) == {"video"}`, `_available_frames` checks
`frame_count`, `round(duration_seconds * fps)`, or `sampled_frames` against
`max(clip_size or 1, 2)`:

> `video too short: ~<N> frames available < clip size <clip>`

### 5. Checkpoint availability

`_weight_status` permits:

- `kind == "random"`: always eligible (no external file);
- `weights.file` set: requires `registry.weights_available(detector_id)` (checking
  `detectors/<id>/weights/<file>`, which symlinks to `<data_dir>/weights/<id>` in
  standalone/container mode); if absent, excludes with the origin host hint
  `checkpoint missing: weights/<file> (download from <host>)`;
- `external` / file-less weights: eligible online (`weights: downloaded at first run`),
  excluded in offline mode (`offline mode; weights not cached locally`).

## Trust tiers (`base_detector.Weights`)

| Tier | Condition | Meaning | Consensus impact |
|---|---|---|---|
| `trained` | `weights.trained: true`, or `kind == "official"` by default | Deepfake-trained head or training-free forensic method; evidence-grade | Counted & auto-selected |
| `backbone-only` | `kind == "backbone"`, `trained` false/unset | Pretrained backbone (ImageNet/Kinetics/ResNet) with untrained head | Excluded & zero-weight |
| `random-init` | `kind == "random"` | No checkpoint anywhere; random weights | Excluded & zero-weight |
| `untrained` | Other kinds with `trained` false/unset | External foundation model with untrained classification head | Excluded & zero-weight |

## Explicit detector overrides

Both CLI (`verity run --detector a,b`) and API (`run_select`) treat named
detectors as an explicit research or targeted examiner request:

1. `include_untrained` is forced `True` when an override is supplied, allowing
   untrained scaffolds to run if named explicitly.
2. Unknown detector IDs are logged/printed (`Unknown detectors ignored: ...`)
   and dropped via intersection with the registry.
3. The original suitability reasons in `selection.decisions` are preserved while
   `decision.included` is rewritten to match the override set, so saved
   `selection.json` and reports match what actually executed.

## Manipulation-family coverage

`src/verity/models/taxonomy.py` defines eight manipulation families:
`face_swap`, `face_reenactment`, `lip_sync`, `face_synthesis`, `image_synthesis`,
`image_editing`, `video_synthesis`, and `audio_synthesis`. Each detector declares
its covered `targets` in `detector.yaml`. Both `verity analyze`, the API
(`GET /api/detectors/coverage`), and `ReportGenerator._coverage_html` group
evidence-grade findings per modality-applicable family — explicitly marking
untested families as **not examined** so absence of detection is never mistaken
for proof of authenticity.

## Related pages

- [Detector runtime and isolation](detector-runtime.md) — execution after selection.
- [Findings, consensus and aggregation](findings-and-consensus.md) — how trust tiers
  gate consensus and panel split bands.
- [Detector catalog (all 58 ports)](../reference/detector-catalog.md) — per-detector
  targets, tiers, and caveats.
