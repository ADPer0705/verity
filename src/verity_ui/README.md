# Verity UI

Web console for the Verity deepfake examination pipeline, built for
investigators, students and researchers. React 18 + TypeScript + Vite,
[Mantine](https://mantine.dev) components, [Tabler](https://tabler.io/icons)
icons and the Inter typeface (all MIT/OFL, all bundled), talking to
`verity_api` over REST.

## Running it

The UI needs the API. From the repository root:

```bash
# once
pip install -e ".[api]"
cd src/verity_ui && npm install
```

### Development (two processes, hot reload)

```bash
# terminal 1 — API on :8000
uvicorn verity_api.main:app --reload --port 8000

# terminal 2 — UI on :5173, proxying /api to :8000
cd src/verity_ui && npm run dev
```

Point the proxy elsewhere with `VERITY_API=http://host:port npm run dev`.

### Production (one process, one port)

```bash
cd src/verity_ui && npm run build     # writes dist/
verity-server                          # serves the API *and* the built SPA
```

`verity-server` honours `VERITY_HOST` (default `127.0.0.1`) and `VERITY_PORT`
(default `8000`). It serves `dist/` with history-API fallback, so deep links
like `/case/<id>?tab=findings` work on refresh.

`npm run typecheck` runs `tsc -b` in strict mode; `npm run build` type-checks
and bundles.

## One interface

There are no profiles. Every case opens with a plain-language summary (what
the panel of detectors found, the agreement bar, what the result does not
tell you, and a link to evidence-handling guidance) and the full record in
tabs underneath: analysis, detector selection with reasons, findings grouped
by manipulation family, explainability evidence, report and activity.

The agreement bar is always shown. Hiding detector disagreement would reduce
Verity to the single real/fake score it exists to avoid.

Controls never change their own label: a checkbox or segmented control states
what it does and shows its state; nothing reads backwards when toggled.

## Coverage by manipulation family

Every detector declares the manipulation families it was trained to
recognise (`targets` in `detector.yaml`; the taxonomy lives in
`verity.models.taxonomy`). The console uses this in three places:

- **Detectors page** — a coverage section lists each family with its
  modalities, the evidence-grade (trained-head) detectors that target it and
  the untrained ones, and a clear "not covered" state. The registry table
  carries a Targets column and can be filtered by family and by head tier.
- **Case → Findings** — a coverage strip shows, for each family relevant to
  the media's modality, how many evidence-grade detectors ran and which way
  they lean, plus the families that were not examined at all. Untrained-head
  findings are tagged red and sorted after evidence-grade ones.
- **Run examination → manual override** — tick every detector targeting a
  family in one go, alongside the tier quick-picks.

## CLI coverage

Every `verity` command is reachable from the UI:

| CLI | Where |
|---|---|
| `verity ingest` | Cases → drop a file |
| `verity extract` | Case → ⋯ → Extract artifacts |
| `verity analyze` | Case → ⋯ → Analyze; results on the Analysis and Selection tabs |
| `verity run` | Case → Run examination (detector override, extractor subset, report format) |
| `verity compile` | Case → ⋯ → Compile findings |
| `verity report --format pdf\|html` | Case → Report tab |
| `verity setup --all/--detector/--force` | Detectors page |

## Layout

```
src/
  api/          client.ts (fetch + XHR upload), hooks.ts (polling + SSE), types.ts
  components/   Shell, ConsensusPanel, JobConsole, ImageGallery, MediaPreview,
                DetectorEvidence, FindingTables, Primitives
  lib/          format.ts, verdict.ts (band → plain language),
                taxonomy.ts (family labels, per-case coverage)
  routes/       Cases · CaseDetail (+tabs/) · RunPanel · Detectors · Activity · System · Help
  styles/       index.css (custom layer; every colour is a Mantine CSS variable)
  theme.ts      Mantine theme (Inter, monospace stack, dense defaults)
```

## Notes

- **Fonts.** Inter is bundled from `@fontsource/inter` and monospace text
  uses the platform stack (`ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace`), so nothing is fetched from a CDN and air-gapped deployments
  render identically.
- **Live progress.** Jobs stream over SSE with an automatic fall back to
  polling — some proxies buffer SSE into uselessness, and a frozen progress
  bar during a long detector run is worse than a chattier poll.
- **Lazy evidence.** Per-detector artifacts, raw JSON and logs are fetched
  only when their panel is opened, so a case with thirty findings does not
  fire ninety requests on load.
- **URL state.** The active case tab and the selected detector on the
  Evidence tab live in the query string (`?tab=evidence&detector=…`) so a
  view can be bookmarked or handed to a colleague.
