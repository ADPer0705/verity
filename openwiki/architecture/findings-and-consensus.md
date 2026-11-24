---
type: subsystem
title: Findings, consensus and aggregation
description: How per-detector outputs compile into a deterministic FindingsCollection, how confidence-weighted aggregation and consensus bands preserve disagreement as a primary finding, and how the UI visualizes the detector probability spectrum.
tags: [findings, consensus, scoring, aggregation, evidence]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-21T06:14:40.994Z
sources:
  - id: openwiki-source-27c03b24282b02c206c94fdd
    resource: repo://src/verity_api/service.py
  - id: openwiki-source-18618a398ed388c9a583c1be
    resource: repo://src/verity_ui/src/routes/tabs/OverviewTab.tsx
  - id: openwiki-source-c404ef3d927c29633b437871
    resource: repo://src/verity/compile/findings_compiler.py
  - id: openwiki-source-78849de1710bc0a43d599863
    resource: repo://src/verity/models/findings.py
  - id: openwiki-source-479291a034ea1b3214af9e14
    resource: repo://src/verity/report/report_generator.py
generated: { by: "antigravity", at: "2026-11-21T06:14:40.994Z" }
---

# Findings, consensus and aggregation

A detector run produces a `Finding`; a completed examination produces a
`FindingsCollection`. This layer enforces Verity's central design rule:
independent detector outputs stay separate with full provenance, and any panel
summary is strictly an orientation aid — never a replacement for individual
findings and visual exhibits.

Because FF++-era face models, self-blended boundary detectors, vision-language
transformers, and frequency/wavelet classifiers fail differently across generator
families and recompression levels, **disagreement between methods is an
investigative signal**, not noise to average away.

## The `Finding` model

`src/verity/models/findings.py` defines:

| Type | Fields |
|---|---|
| `Classification` | `REAL`, `FAKE`, `UNCERTAIN` |
| `Reasoning` | `summary`, `evidence` (artifact paths), `confidence_factors` |
| `Finding` | `detector_id`, `detector_version`, `classification`, `confidence` (`0.0..1.0`), `reasoning`, `raw_output`, `tier`, `timestamp` |
| `FindingsCollection` | `media_id`, `findings`, `aggregated_score` |

`Finding.is_trained` is `True` only when `tier == "trained"`. That single gate
protects every downstream summary so `backbone-only`, `random-init`, and
`untrained` scaffolds never influence scores or consensus bands.

## Compiling findings

`FindingsCompiler.compile` (`src/verity/compile/findings_compiler.py`) scans
`workspace/output/` for `<media_id>_*` subdirectories, reads every `finding.json`,
and:

1. Backfills a missing `tier` from the current `DetectorRegistry` definition so
   older workspace artifacts cannot silently pass as `trained`.
2. Sorts findings monotonically by `detector_id` for deterministic reports.
3. Computes `_aggregate(findings)` over `is_trained` findings only.

## Confidence-weighted aggregate score

`FindingsCompiler._aggregate` weights each evidence-grade `FAKE` indicator (`1.0`
vs `0.0`) by its detector confidence:

```python
findings = [f for f in findings if f.is_trained]
weighted = sum(f.confidence * (1.0 if f.classification == Classification.FAKE else 0.0)
               for f in findings)
total = sum(f.confidence for f in findings)
score = weighted / total if total > 0 else None
```

Returns `None` when zero evidence-grade findings exist. The exported HTML/PDF report
deliberately omits even this aggregate score, keeping the examiner focused on per-method
coverage and exhibits.

## Mean fake probability consistency

Both the HTTP API (`verity_api.service.mean_fake_probability`) and the report
generator (`ReportGenerator._mean_probability`) compute per-detector probability via
the exact same three-step rule:

1. Mean of `raw_output["per_face"][*]["fake_prob"]` when non-empty;
2. Otherwise `raw_output["prediction"]`;
3. Otherwise `0.0`.

Sharing this rule prevents the web console and the PDF/HTML dossier from ever
displaying contradictory numbers for the same case.

## Consensus bands and panel splits

`verity_api.service.consensus(findings)` excludes untrained findings (`untrained_excluded`)
and summarizes the panel without hiding contention. If zero trained findings exist, it
returns `band: "no_analysis"`.

Otherwise, it computes `agreement = max(fake, real) / total`, `disagreement = fake > 0 and real > 0`,
`mean_fake_probability`, and `leaning_split` (whether at least one detector leans
fake at `>= 0.6` while another leans real at `<= 0.4`), selecting the consensus band
in strict precedence order:

| Condition | Band |
|---|---|
| `disagreement` and `min(fake, real) / total >= 0.25` | `inconclusive` |
| `fake / total >= 0.7` and `mean_fake_probability >= 0.6` | `strong_manipulation` |
| `fake > real` | `likely_manipulation` |
| `real / total >= 0.7` and `mean_fake_probability <= 0.4` | `strong_authentic` |
| `real > fake` | `likely_authentic` |
| Otherwise | `inconclusive` |

If at least a quarter of evidence-grade detectors dissent across `FAKE` and `REAL`,
the panel always resolves to `inconclusive` regardless of how confident individual
models are.

## UI probability spectrum and inspection workflow

On the Web Console **Overview** tab (`src/verity_ui/src/routes/tabs/OverviewTab.tsx`),
evidence-grade findings are additionally rendered as a **Detector probability
spectrum** sorted by descending `mean_fake_probability`. Each spectrum bar displays the
detector's probability meter, unit count, classification badge, and confidence, and
clicks directly into the **Evidence & XAI** tab focused on that specific detector so
the examiner can inspect why disagreeing detectors diverged (e.g., face crop framing,
lip-sync offset, or post-compression degradation).

## Consumers

- `verity compile` — prints sorted findings.
- `GET /api/media/{id}` and `GET /api/media/{id}/findings` — return serialized findings,
  `aggregated_score`, and `consensus`.
- Web console Overview, Findings, and Evidence tabs (`OverviewTab.tsx`, `FindingsTab.tsx`,
  `DetectorEvidence.tsx`).
- `ReportGenerator` (`report.html` / `report.pdf` — without aggregate score or verdict).

## Related pages

- [Detector selection and trust tiers](detector-selection.md) — how trust tiers are assigned.
- [Report generation](reporting.md) — why exported reports omit aggregate scores.
- [Web API and investigator console](web-api-and-console.md) — consensus and probability
  spectrum presentation.
