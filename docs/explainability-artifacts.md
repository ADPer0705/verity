# Explainability artifacts

Every finding in the repository is designed to be **senior-reviewable**. This document catalogs every artifact Verity writes during a run, organized by where in the pipeline it appears, what file(s) it writes, and what its evidence value is.

## Artifact taxonomy

| Category | Written by | Evidence role | Duration |
|---|---|---|---|
| **Case record** | Ingest, extract, analyze, select | Chain of custody, examinability guarantees, selection rationale | Full run |
| **Forensics artifacts** | Model-free analyzers | Classic forensic methods, examiner diagrams | Full run |
| **Detector artifacts** | Per-detector inference | ML evidence for each detector's verdict | Per detector |
| **Compiled / reported artifacts** | Compile, report | The deliverable that a supervisor signs | End of run |

All paths below live under `workspace/` (`$VERITY_WORKSPACE`) and `output/` (`$VERITY_OUTPUT_DIR`). Media IDs are the SHA-256 checksums of ingested files; report paths are `<output_dir>/<media_id>/`.

---

## 1. Case record artifacts

### 1.1 `analysis.json` — media characterization

`analyze` stage output, per-media. Produced deterministically by the same
media file. Contains:

| Field group | Examples |
|---|---|
| **Media identity** | `media_id` (SHA-256), `media_type` (image/video/audio), file format |
| **Container facts** | video codec, duration, fps, resolution, frame count, has_audio_track |
| **Face-presence scan** | per-frame face count and max face size over up to 30 evenly sampled frames (+`summary`) |

Evidence role: the **basis of detector selection.** A missing face or short clip changes
which detectors are run, so the analysis must be examinable in a review.

### 1.2 `selection.json` — why each detector ran (or didn't)

One entry per installed detector: `included: true/false` and the full **reason**
string chain. Rules, in order:

0. Trust tier — untrained heads (`random-init`, `backbone-only`) excluded by
   default (flag `--include-untrained` overrides, findings flagged)
1. Media-type compatibility
1b. Audio-track requirement for audio detectors on video
2. Face presence requirements
3. Minimum face size requirements
4. Clip length requirements for video detectors
5. Checkpoint availability (offline mode honored)

Stored per media in `workspace/artifacts/<media_id>/analysis/selection.json`.
The web console shows the "why" for **all** detectors, selected or not, via
`GET /api/media/{id}`.

### 1.3 Exposure record — `finding.json` → `findings_collection.json`

`compile` collects every detector's `finding.json` into a single
`FindingsCollection`, ordered by detector id, with the trust-tier annotation
promoted to first-class:

```jsonc
// finding.json, one per detector
{
  "detector_id": "spsl",
  "detector_version": "1.0.0",
  "classification": "FAKE",           // REAL / FAKE / UNCERTAIN
  "confidence": 0.9321,               // 0.0 .. 1.0
  "reasoning": {
    "summary": "SPSL FAKE: Average fake probability …",
    "confidence_factors": [ "Weight file …", "Gradient-weighted …" ],
    "evidence": ["paths/into/artifacts"]
  },
  "raw_output": { …verbatim detector JSON… },
  "tier": "trained"                    // trained / backbone-only / random-init / untrained
}
```

Evidence role: the **signed record**. Every detector statement, its
provenance, and the raw output used to make it appear in the collection.

---

## 2. Detector artifacts

Each detector writes into its own directory under `workspace/output/<media_id>_<detector>/`
(that is also the per-detector sandbox the runner uses — see the API docs for
how these are surfaced through HTTP). The files below are produced by
`detectors/common/engine.py` and the detector's `inference.py`; the naming is
a convention shared by all 47 ports, so a supervisor always knows where to
look.

| Artifact | Content | Evidence role |
|---|---|---|
| `result.json` | Detector's verdict schema (`prediction`, `confidence`, `reasoning`, `confidence_factors`, optional `attention_maps`, `reconstruction`) | The **unmodified ML verdict** the orchestrator parses into `finding.json` |
| `faces/<face_id>.png` | Face crops exactly as fed to the model (the model's actual input view) | Confirms what shares of the image drive the score; catches wrong detections |
| `attention_maps/<face_id>_attention.png` | For face detectors: 3-panel compositing (original → heatmap → overlay) of the model's attention or Grad-CAM per face | Visualizes **where** the model decided; direct evidence for/against its verdict |
| `frames/img_NNNNN.png` | For full-frame detectors: the whole frames actually scored | Confirms every picture evaluated |
| `frames/..._frame.png` | For videos: thumbnails of most-suspicious frames with bbox and fake-prob caption | Examiner entry point into a long video |
| `timeline.png` | Per-frame prediction timeline: mean fake probability vs per-face/clip overlay | Temporal coherence view — is the score stable, or spiking on single frames? |
| `reconstructions/*.png` | RECCE only: reconstruction error heatmaps of face crops (when the decoder is alive — see the RECCE honest-degradation audit) | Provides generation-side visual evidence |
| `stdout.log`, `stderr.log` | Verbatim detector stdout/stderr (last 20,000 chars) | Debugging, reproduction, and auditor review if a run behaves oddly |

All image artifacts are lossless PNG. Nothing in this directory is
hand-edited after inference; the runner only adds the two log files.

### 2.1 Region-energy analysis (face detectors with `region_analysis: true`)

For detectors that run with a face mesh, `engine.py` projects the attention
map onto 6 named face regions (left/right eye, nose, mouth, jaw, forehead)
using MediaPipe. Each face record gets

```jsonc
"region_energy": { "left_eye": {"attention_pct": 4.1, "face_count": 1},
                    "mouth": {"attention_pct": 61.3, "face_count": 1}, … }
```

The reasoning builder then converts that into examinable sentences, e.g.
*"Most-implicated region: mouth (61.3% of attention energy)"* when fake, or
*"attention peaks in N (…%), consistent with natural structure."* when real.
This makes the **why** behind a verdict a first-class artifact, directly
comparable across detectors.

### 2.2 Native attention vs Grad-CAM

The engine prefers a detector's **native** attention map (`attention_map`
returned by `forward_detailed`, e.g. RECCE's reconstruction-error map, DF40's
CLS attention over the patch grid) and falls back to **Grad-CAM** (`cam: true`
in `detector.yaml`) for pixel-space backbones (Xception, EfficientNet,
Capsule, ResNet). A few detectors deliberately produce neither:

- **audio detectors** — attention is computed over the waveform, so the
  artifact is the per-window timeline instead (see the audio analyzers);
- **wavelet-input detectors** (SAFE) — CAM over wavelet energy would
  localize energy rather than semantics; per-frame timelines are produced
  instead;
- **training-free detectors** (`aeroblade`, `univfd`) — have no gradient
  surface to map; the report states this plainly.

Every inclusion/exclusion reason of this form lands in the detector's
`confidence_factors`, so the absence of a map is itself documented.

### 2.3 Per-detector explainability contracts

`detector.yaml` declares the artifact contract per detector through:

| key | meaning |
|---|---|
| `artifacts.required` | artifact kind the detector needs to run: `face_crop`, `frame`, `video`, or `audio` (with resolution parameters) |
| `cam: bool` | native attention produced directly by `forward_detailed` (if explicitly `false`, the map cannot be produced and it will not be retasked) |
| `region_analysis: bool` | project the attention onto the 6 named face regions |

This contract is the contract honored by 47 ports. It is the reason the web
UI can offer identical UI affordances on every detector's evidence panel.

### 2.4 Trust-tier honesty embedded in the artifacts

`finding.json` flags anything that runs with untrained variables, and the
engine's reasoning builder writes this into the finding's
`confidence_factors`, in effect *pre-sealed*:

- `random-init` → *"Weights file not present / random — scores NOT meaningful"*
- `backbone-only` → *"Only a pretrained backbone is available; head is untrained, treat scores as unreliable"*

Findings with `tier != "trained"` never count toward the consensus score and
are visually flagged in the UI/report.

---

## 3. Forensic artifacts

Documented exhaustively in [forensics.md](forensics.md). One-line summary of
what each analyzer puts on disk:

| Analyzer | Files written |
|---|---|
| `metadata` | `metadata/metadata.json`, `metadata/exif.json` |
| `compression` | `compression/ela.png`, `compression/blockiness.png`, `compression/dct_histogram.png` |
| `noise` | `noise/noise_map.png`, `noise/spectrum.png` |
| `video_temporal` | `video_temporal/temporal.json`, `video_temporal/timeline.png` |
| `audio` | `audio/spectrogram.png`, `audio/spectrum.png`, `audio/stats.json` |

All live under `workspace/artifacts/<media_id>/forensics/`, indexed by
`forensics.json`, which is what the web console and reports read.

---

## 4. Reported artifacts

```text
output/<media_id>/
├── report.html              the full HTML report (Web UI linked, deterministic content)
├── report.pdf               the PDF rendering of the same document
├── analysis.json            §1.1
└── selection.json           §1.2
```

The report embeds the **explanatory substance** of the case, not the raw
records: verdict, reasoning, per-detector summary and the top-3 artifacts per
type, with the complete record in dropdowns. Every artifact path included in
the report prints into "Exhibits" when it can be accompanied contextually.

## 5. Design invariants (why these artifacts exist in this shape)

1. **Determinism.** Same media → same artifacts. Media identity is
   SHA-256-based; sampling is every-N, not wall-clock-based.
2. **Provenance.** Every number in a summary sentence can be traced to a raw
   value inside `result.json` / `forensics.json`.
3. **Examiner-first.** The default presentation is the artifact a
   human can inspect — images over tables, tables over black-box digits.
4. **Honest absence.** If a signal cannot be produced (no CAM, dead
   reconstruction head, no audio), the pipeline states that absence explicitly
   in the reasoning instead of silently omitting evidence.
