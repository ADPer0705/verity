---
type: subsystem
title: Report generation
description: Deterministic self-contained HTML and PDF report generation in ReportGenerator, covering selection filtering, modality-aware evidence rendering, forensics embedding, and the no-aggregate-verdict policy.
tags: [reporting, html, pdf, weasyprint, explainability, determinism]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-479291a034ea1b3214af9e14
    resource: repo://src/verity/report/report_generator.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Report generation

The generated report is the portable deliverable an investigator attaches to case
files or hands to a peer reviewer. `ReportGenerator`
(`src/verity/report/report_generator.py`) compiles the `Report` model
(`src/verity/models/report.py`) from the case's `MediaInfo`,
`FindingsCollection`, optional `MediaAnalysis` and `DetectorSelection`, plus
`artifacts/<media_id>/forensics/forensics.json`, and emits a self-contained HTML
document (and optional PDF via WeasyPrint).

Why embed every PNG asset as a base64 `data:` URI inside `report.html`? So an
examiner can archive, email, or open the dossier inside an offline evidence vault
without breaking image links or shipping the entire mutable workspace.

## Output files (`output_dir/<media_id>/`)

| Method | Files written | Behavior |
|---|---|---|
| `save_html` (`--format html`) | `report.html` | Always self-contained; requires no WeasyPrint system libs |
| `save` (`--format pdf`, default) | `report.html` + `report.pdf` | Writes `report.html` first, then renders `report.pdf` via WeasyPrint |

Notice `verity_api.service.run_report` mirrors this exact rule: HTML is written
before WeasyPrint touches Pango/Cairo, so PDF native-library failures never erase
the HTML exhibit.

## Why exported reports omit verdicts and aggregate scores

Both CLI/UI consensus bands and `FindingsCollection.aggregated_score` exist to
orient an examiner exploring a case interactively. By strict design invariant
(enforced by `tests/test_report_generator.py`), the exported dossier **never renders
an overall verdict box, an aggregate score, timing/benchmark noise, or fake/real
summary labels** at the top level. Every finding and model-free forensic observation
is presented separately so a court or reviewer weighs the full technical evidence
rather than deferring to a single collapsed number.

## Report structure

### 1. Chain-of-custody header

Opens with the file name, `media_id`, byte size, full `SHA-256` checksum, container
characteristics (`duration`, `resolution`, `fps`, `face_count`), and UTC generation
timestamp.

### 2. Manipulation-family Coverage table

For every manipulation family in `FAMILIES` matching the media's modality,
`_coverage_html` summarizes whether evidence-grade (`trained`) detectors covered
it (`examined by N`) or not (`not examined`, explicitly warning that an untested
family is **not** proof of authenticity). Untrained scaffolds never inflate
coverage counts.

### 3. Findings Summary & Exclusion Rationale

Lists every detector included in `selection` with its `Weights` trust-tier badge
(`trained` vs flagged untrained), analyzed unit count, mean fake probability, and
confidence (`no finding` if a selected detector crashed). Excluded detectors appear
in a collapsible `<details>` block alongside their exact selector reasons.

### 4. Forensic Analyses (model-free)

When `forensics.json` exists, `_forensics_html` embeds warning/notice/info counts,
each analyzer's verbatim `method` caveat, sorted observation tables with key-value
evidence metrics, and diagnostic figures (`timeline.png` / `spectrogram.png` full
width; `ela.png`, `ela_overlay.png`, `blockiness.png`, `noise_residual.png`,
`noise_variance.png`, `spectrum.png` tiled).

### 5. Per-finding evidence pages (`_finding_html`)

Rendered deterministically in sorted `detector_id` order for findings included in
this run's `selection`:

- **Heading & Trust Warning**: Method name, `detector_id`, version, trust badge,
  and an explicit "Not evidence-grade" banner if `tier != "trained"`.
- **Reasoning & Confidence Factors**: Blockquote reasoning summary, bulleted
  `confidence_factors`, and collapsible `Face Region Energy Attribution` table when
  `region_aggregate` exists.
- **Key Figures (`_key_figures_html`)**: Units analyzed, `mean_fake_probability`,
  score range, above-threshold count, video sampling coverage %, and numeric
  **`Auxiliary Signals`** (`aux_aggregate` — such as per-VAE LPIPS distances or
  AV sync offset/confidence).
- **Visual Exhibits (`_artifacts_html`)** loaded from
  `workspace/output/<media_id>_<detector_id>/`:
  - `timeline.png` (per-frame / per-window probability trace)
  - `full_frames/*_frame.png` (top-3 suspicious full video frames with face boxes)
  - `faces/*.png` (top-6 highest-probability face crops + collapsible remainder)
  - `frames/*.png` (top-6 evaluated whole frames/windows when no face crops exist)
  - `attention_maps/*_attention.png` (top-4 native attention / Grad-CAM overlays)
  - `reconstructions/*_recon.png` (top-4 reconstruction residuals when alive)
- **Modality-aware detail table**: Renders either **Per-Window Results**
  (`Window ID`, `Time Span` in seconds, `Fake Prob`) when `unit == "window"` or
  **Per-Face Results** (`Face ID`, `Frame`, `BBox`, `Fake Prob`), sorted by
  descending probability with ID tie-breaker.

## Selection filtering and determinism

`ReportGenerator.generate` filters compiled findings against `selection.decisions`
so prior detector runs on the same `media_id` never leak into a new run's report.
Except for `generated_at`, sorted filenames, sorted detector IDs, and deterministic
tie-breakers ensure identical byte-level HTML across repeated runs
(`test_report_is_deterministic`).

## Related pages

- [Findings, consensus and aggregation](findings-and-consensus.md) — shared
  `mean_fake_probability` rule and why consensus stays out of reports.
- [Model-free forensic analyses](forensics.md) — source of embedded forensic tables
  and figures.
- [Explainability artifacts and on-disk formats](../reference/explainability-artifacts.md) —
  full artifact paths and `verity export` zip bundles.
- [Web API and investigator console](web-api-and-console.md) — `/api/media/{id}/report`
  and interactive dossier tab.
