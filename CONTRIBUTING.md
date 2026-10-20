# Contributing to Verity

Thanks for your interest. Verity is an evidence-oriented tool: correctness,
provenance and explainability matter more than feature count. Please read the
scope below before opening a large pull request.

## Ways to contribute

- **Bug reports** — reproducible pipeline/API/UI issues, wrong selection
  decisions, artifact problems, documentation errors.
- **Detector ports** — new architectures or fixes to existing ones, as long
  as the artifact and trust-tier contract is respected (below).
- **Forensics** — new model-free analyzers or improvements to the existing
  five. Observations must come with the numbers and the limits of the method.
- **Docs and tests** — always welcome; the docs are part of the product.

Security issues: **do not open a public issue** — see [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/ADPer0705/verity.git
cd verity
uv sync --extra api
cd src/verity_ui && npm ci && cd -
uv run pytest -q
```

System packages for PDF reports: `ffmpeg libpango-1.0-0 libpangocairo-1.0-0
libcairo2`. Detector environments are provisioned on demand:

```bash
uv run verity weights pull --trained
uv run verity setup --trained            # or --detector <id>
```

Never commit checkpoints, `workspace/`, `output/`, `.env`, `.venv/`,
`.detectorenvs/` or `node_modules/` (all are gitignored).

## Repository conventions

- Python 3.10+ for the orchestrator; detector projects pin their own Python
  and dependencies in `detectors/<id>/pyproject.toml`.
- Keep the pipeline deterministic: sorted iteration, fixed seeds, media ids
  derived from SHA-256. No wall-clock or random values in findings.
- Findings must remain explainable: if a signal cannot be produced, state the
  absence in `confidence_factors` rather than omitting it silently.
- The exported report must not gain an aggregate real/fake verdict.
- Public functions and modules get docstrings explaining *why*, not only what.

## Adding a detector

1. Create `detectors/<id>/` with:
   - `detector.yaml` — id, name, description, architecture, `targets`
     (families from `src/verity/models/taxonomy.py`), `input_mode`,
     preprocessing values, `artifacts.required`, `cam`/`region_analysis`, and
     a `weights` block that honestly states the tier.
   - `model.py` — architecture + `build_model(weights_path, device)` and a
     `forward_detailed()` that returns at least `logits` (and `feat` for
     Grad-CAM, `attention_map` if the model exposes one).
   - `inference.py` — a thin entrypoint calling
     `detectors/common/engine.run_detector`.
   - `output_parser.py` — only if the raw result needs custom handling;
     otherwise the default parser is used.
   - `pyproject.toml` — exact pins, only this detector's dependencies.
   - `weights/README.md` — origin, size, SHA-256, license, citation.
2. Add the checkpoint to `detectors/weights_manifest.yaml`
   (`files`, `sources`, `detectors`) if Verity may fetch it; use
   `runtime_fetched` for models the detector downloads itself, and
   `no_checkpoint` when no usable weights exist. Record `redistributable`
   truthfully — when in doubt, `false`.
3. Verify the tier: a head trained on a deepfake/generation task is
   `trained`; a pretrained backbone with an untrained head is
   `backbone-only`; random weights are `random-init`.
4. Run `uv run python scripts/gen_detector_table.py` and
   `uv run python scripts/gen_third_party_notices.py`.
5. Add/extend tests:
   - registry loads the YAML and derives the expected capability/tier,
   - output parser maps a representative raw result to a `Finding`,
   - the detector is selected/excluded for the right media characteristics.
6. Document the detector in `docs/detector-catalog.md` (what it looks for,
   artifacts, caveats) and add its citation to `THIRD_PARTY_NOTICES.md`.

A port that only runs on the author's machine, or that hides a missing
checkpoint behind plausible scores, will not be merged.

## Tests and checks

Before opening a PR:

```bash
uv run pytest -q
uv run python scripts/gen_detector_table.py --help >/dev/null  # imports cleanly
uv run verity weights status
cd src/verity_ui && npm run typecheck && npm run build
```

If your change touches the manifest, run `verity weights verify` where the
files are present. If it touches the report, attach a sample case (media can
be synthetic) or describe the generated artifact.

## Commit and PR style

- Small, focused commits with a clear subject:
  `feat(detectors): add X`, `fix(api): …`, `docs: …`, `test: …`,
  `build: …`, `chore: …`.
- Reference the affected area (`pipeline`, `detectors`, `api`, `ui`,
  `weights`, `forensics`, `docs`).
- Describe **what changed and why**, plus any evidence/verification steps.
- Keep generated files (`docs/detectors.md`, notice tables) in the same PR as
  the change that caused them.
- By contributing you agree your work is licensed under the project's
  CC BY-NC 4.0 license and that you have the right to submit it.

## Scope and review

Maintainers may decline changes that:

- add an aggregate verdict or otherwise collapse per-detector findings;
- ship weights without recorded provenance/checksums/licensing;
- weaken offline behaviour, checksum verification or the artifact contract;
- introduce network calls at import time or non-deterministic inference.

Be patient and precise in review. "It works on my machine" is not evidence
for an evidence tool.
