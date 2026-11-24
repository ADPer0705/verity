## Summary

<!-- What changed, and why. Link the issue if one exists. -->

## Area

- [ ] pipeline / CLI
- [ ] detector(s)
- [ ] selection
- [ ] forensics
- [ ] report
- [ ] API
- [ ] web UI
- [ ] weights / manifest
- [ ] container / build
- [ ] docs

## Verification

<!-- Commands run and results; sample artifact paths or screenshots if useful. -->

- [ ] `uv run pytest -q`
- [ ] `uv run verity weights status`
- [ ] `cd src/verity_ui && npm run typecheck && npm run build` (UI changes)
- [ ] regenerated `THIRD_PARTY_NOTICES.md` and refreshed OpenWiki if behavior/manifest changed

## Checklist

- [ ] Findings stay per-detector; no aggregate real/fake verdict was added.
- [ ] New checkpoints have origin, SHA-256, size and license in the manifest.
- [ ] Untrained heads are labelled and excluded from consensus.
- [ ] No weights, media, workspace/output data, secrets or personal data are included.
- [ ] Documentation updated for behaviour changes.
