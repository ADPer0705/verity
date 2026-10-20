from __future__ import annotations
import json
import os
import sys
import yaml
import click
from pathlib import Path

from verity.models.media import MediaType
from verity.ingest.media_ingestor import MediaIngestor
from verity.extract.frame_extractor import FrameExtractor
from verity.extract.face_extractor import FaceExtractor
from verity.extract.metadata_extractor import MetadataExtractor
from verity.extract.audio_extractor import AudioExtractor
from verity.forensics import ForensicsRunner
from verity.analyze.media_analyzer import MediaAnalyzer
from verity.analyze.detector_selector import DetectorSelector
from verity.detect.detector_registry import DetectorRegistry
from verity.detect.detector_runner import DetectorRunner
from verity.compile.findings_compiler import FindingsCompiler
from verity.report.report_generator import ReportGenerator

EXTRACTORS = ["frame", "face", "metadata", "audio"]


def _default_config() -> dict:
    return {
        "detectors_dir": os.environ.get("VERITY_DETECTORS_DIR", ""),
        "workspace": os.environ.get("VERITY_WORKSPACE", "workspace"),
        "output_dir": os.environ.get("VERITY_OUTPUT_DIR", ""),
        "extractors": EXTRACTORS,
    }


def _load_config(config: str | None) -> dict:
    cfg = _default_config()
    if config and Path(config).exists():
        with open(config) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if k in cfg})
    return cfg


def _detectors_dir(cfg: dict) -> Path:
    if cfg.get("detectors_dir"):
        return Path(cfg["detectors_dir"])
    repo = Path(__file__).resolve().parents[2]
    repo_detectors = repo / "detectors"
    if repo_detectors.exists():
        return repo_detectors
    raise click.ClickException(
        "Cannot locate detectors/. Set detectors_dir in config or "
        "VERITY_DETECTORS_DIR."
    )


def _workspace(cfg: dict) -> Path:
    return Path(cfg.get("workspace", "workspace"))


def _output_dir(cfg: dict, workspace: Path) -> Path:
    return Path(cfg.get("output_dir") or (workspace / "output"))


def _extractors(ws: Path, cfg: dict):
    enabled = cfg.get("extractors") or EXTRACTORS
    mapping = {
        "frame": FrameExtractor,
        "face": FaceExtractor,
        "metadata": MetadataExtractor,
        "audio": AudioExtractor,
    }
    return [mapping[name](ws) for name in enabled if name in mapping]


def _media_type(ws: Path, media_id: str) -> MediaType:
    """Media type recorded by the ingestor for an already-ingested item."""
    from verity.models.media import MediaInfo
    meta = ws / "media" / f".{media_id}.json"
    if meta.exists():
        return MediaInfo(**json.loads(meta.read_text())).media_type
    return MediaType.UNKNOWN


def _run_forensics(ws: Path, media_id: str, media_type: MediaType):
    """Model-free forensic analyses; failures never abort the pipeline."""
    report = ForensicsRunner(ws).run(media_id, media_type)
    c = report.counts
    click.echo(
        f"Forensics: {len(report.results)} analyzer(s), "
        f"{c.get('warning', 0)} warning(s), {c.get('notice', 0)} notice(s), {c.get('info', 0)} info"
    )
    return report


def _media_files(media: Path) -> list[Path]:
    media = media.resolve()
    if media.is_dir():
        return sorted(p for p in media.iterdir() if p.is_file())
    return [media]


@click.group()
def cli():
    pass


@cli.command()
@click.option("--media", default=None, type=click.Path(exists=True),
              help="Media file or directory of media files (default: $INPUT_DIR)")
@click.option("--detector", default=None,
              help="Comma-separated detector override (default: auto-select)")
@click.option("--workspace", default=None, type=click.Path(), help="Workspace directory")
@click.option("--config", default=None, type=click.Path(), help="Pipeline config file")
@click.option("--output", default=None, type=click.Path(), help="Report output directory")
@click.option("--format", "fmt", default="pdf", type=click.Choice(["pdf", "html"]),
              help="Report format")
@click.option("--include-untrained", is_flag=True,
              help="Also auto-select detectors whose classification head is untrained "
                   "(random-init / backbone-only). Research use only; findings are flagged.")
@click.option("--no-forensics", is_flag=True,
              help="Skip the model-free forensic analyses (metadata, compression, noise, temporal, audio)")
def run(media: str | None, detector: str | None, workspace: str | None, config: str | None,
        output: str | None, fmt: str, include_untrained: bool, no_forensics: bool):
    """Full pipeline: ingest -> extract -> forensics -> analyze -> select -> run -> compile -> report."""
    media_path = Path(media) if media else Path(os.environ.get("INPUT_DIR", "/input"))
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    if output:
        cfg["output_dir"] = output
    ws = _workspace(cfg)
    out_dir = _output_dir(cfg, ws)

    registry = DetectorRegistry(_detectors_dir(cfg))
    runner = DetectorRunner(ws, registry)
    ingestor = MediaIngestor(ws)
    analyzer = MediaAnalyzer(ws / "media")
    # An explicit --detector list is an explicit request: untrained detectors
    # named there run (and are flagged); auto-selection stays evidence-only.
    selector = DetectorSelector(registry, include_untrained=include_untrained or bool(detector))
    compiler = FindingsCompiler(ws, registry.list_detectors())
    generator = ReportGenerator(ws, registry.list_detectors())

    detector_override = [d.strip() for d in detector.split(",")] if detector else None

    for media_path in _media_files(media_path):
        click.echo(f"\n=== {media_path.name} ===")
        media_copy = ingestor.ingest(media_path)
        media_id = media_copy.media_id
        click.echo(f"Ingested: {media_id} ({media_copy.original.media_type.value})")

        for ext in _extractors(ws, cfg):
            ext.extract(media_id)
        if not no_forensics:
            _run_forensics(ws, media_id, media_copy.original.media_type)

        analysis = analyzer.analyze(media_id, media_copy.original.media_type)
        faces = analysis.media.face_count
        click.echo(
            f"Analysis: {analysis.media.format or 'n/a'}, "
            f"faces detected: {faces} in {analysis.media.sampled_frames} sampled frames"
        )

        selection = selector.select(analysis)
        selected = (
            sorted(set(detector_override) & set(registry.list_detectors()))
            if detector_override
            else selection.selected
        )
        if detector_override:
            # The saved selection must reflect what actually ran: keep the
            # analyzer's decisions (and reasons) but mark the override set as
            # included.
            selection = selection.model_copy(deep=True)
            override_set = set(selected)
            for decision in selection.decisions:
                decision.included = decision.detector_id in override_set
        skipped = [d for d in (detector_override or []) if d not in registry.list_detectors()]
        if skipped:
            click.echo(f"Unknown detectors ignored: {', '.join(skipped)}")

        n_excluded = len([d for d in selection.decisions if not d.included])
        click.echo(f"Detectors selected: {len(selected)}"
                   + ("" if detector_override else " (auto)")
                   + f" / excluded: {n_excluded}")
        for det_id in selected:
            click.echo(f"  running {det_id} ...")
            finding = runner.run(media_id, det_id)
            click.echo(
                f"    -> {finding.classification.value} "
                f"(fake prob: {finding.raw_output.get('prediction', 0.5):.4f}, "
                f"confidence: {finding.confidence:.4f})"
            )

        findings = compiler.compile(media_id)
        click.echo(f"Findings: {len(findings.findings)}")

        report = generator.generate(media_id, analysis=analysis, selection=selection)
        report_dir = out_dir / media_id
        report_dir.mkdir(parents=True, exist_ok=True)
        analyzer.save(analysis, report_dir)
        selector.save(selection, report_dir)
        base = report_dir / "report"
        if fmt == "html":
            generator.save_html(report, base.with_suffix(".html"))
            click.echo(f"Report: {base.with_suffix('.html')}")
        else:
            generator.save(report, base.with_suffix(".html"))
            click.echo(f"Report: {base.with_suffix('.pdf')}")


@cli.command()
@click.option("--media", default=None, type=click.Path(exists=True),
              help="Media file or directory of media files (default: $INPUT_DIR)")
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Print analysis as JSON")
@click.option("--include-untrained", is_flag=True,
              help="Treat untrained-head detectors as eligible (research use only)")
def analyze(media: str | None, workspace: str | None, config: str | None, as_json: bool,
            include_untrained: bool):
    """Analyze media and show which detectors are suitable and why."""
    media_path = Path(media) if media else Path(os.environ.get("INPUT_DIR", "/input"))
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)

    registry = DetectorRegistry(_detectors_dir(cfg))
    ingestor = MediaIngestor(ws)
    analyzer = MediaAnalyzer(ws / "media")
    selector = DetectorSelector(registry, include_untrained=include_untrained)

    for media_path in _media_files(media_path):
        media_copy = ingestor.ingest(media_path)
        analysis = analyzer.analyze(media_copy.media_id, media_copy.original.media_type)
        selection = selector.select(analysis)
        if as_json:
            click.echo(json.dumps({
                "media_id": media_copy.media_id,
                "analysis": analysis.model_dump(mode="json"),
                "selection": selection.model_dump(mode="json"),
            }, indent=2))
            continue

        char = analysis.media
        click.echo(f"\n=== {media_path.name} ===")
        click.echo(f"Media id : {char.media_id}")
        click.echo(f"Type     : {char.media_type.value} ({char.format or 'n/a'})")
        if char.duration_seconds is not None:
            click.echo(f"Duration : {char.duration_seconds:.2f}s")
        if char.width and char.height:
            click.echo(f"Size     : {char.width}x{char.height}")
        if char.fps:
            click.echo(f"FPS      : {char.fps}")
        click.echo(
            f"Faces    : {'yes (' + str(char.face_count) + ' in ' + str(char.sampled_frames) + ' sampled frames)' if char.face_present else 'none'}"
        )
        from verity.models.taxonomy import FAMILIES
        defs = registry.list_detectors()
        click.echo("\nCoverage (evidence-grade detectors selected per manipulation family):")
        for fam_id, meta in FAMILIES.items():
            if char.media_type.value not in meta["modalities"]:
                continue
            chosen = [d for d in selection.selected if fam_id in defs[d].targets and defs[d].weights.is_trained]
            label = meta["label"]
            click.echo(f"  {label:<24} {len(chosen):2d}  {'not examined' if not chosen else ', '.join(chosen)}")
        click.echo(f"\nSelected ({len(selection.selected)}):")
        for d in selection.selected:
            click.echo(f"  + {d}")
        excluded = [d for d in selection.decisions if not d.included]
        if excluded:
            click.echo(f"\nExcluded ({len(excluded)}):")
            for d in excluded:
                click.echo(f"  - {d.detector_id}: {'; '.join(d.reasons)}")


@cli.command()
@click.option("--all", "all_", is_flag=True, help="Provision all detector environments")
@click.option("--trained", is_flag=True,
              help="Provision only detectors with a trained classification head (the default evidence set)")
@click.option("--detector", default=None, help="Provision one or more detectors (comma-separated)")
@click.option("--force", is_flag=True, help="Re-provision even if up to date")
@click.option("--concurrency", type=int, default=8, show_default=True,
              help="Number of detector environments to build in parallel")
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def setup(all_: bool, trained: bool, detector: str | None, force: bool, concurrency: int,
          workspace: str | None, config: str | None):
    """Provision detector environments with uv (cached under $VERITY_DETECTOR_ENVS)."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    registry = DetectorRegistry(_detectors_dir(cfg))
    runner = DetectorRunner(ws, registry)
    if detector:
        ids = [d.strip() for d in detector.split(",") if d.strip()]
    elif trained:
        ids = sorted(i for i, d in registry.list_detectors().items() if d.weights.is_trained)
    elif all_:
        ids = None
    else:
        raise click.ClickException("Use --all, --trained or --detector <id>[,<id>...]")
    sys.exit(runner.setup(detector_ids=ids, force=force, concurrency=concurrency))


@cli.command()
@click.option("--media", required=True, type=click.Path(exists=True))
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def ingest(media: str, workspace: str | None, config: str | None):
    """Ingest media: create checksummed read-only copy."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    for media_path in _media_files(Path(media)):
        copy = MediaIngestor(ws).ingest(media_path)
        click.echo(f"Media ID: {copy.media_id}")
        click.echo(f"Type: {copy.original.media_type.value}")
        click.echo(f"Checksum: {copy.original.checksum}")
        click.echo(f"Copy: {copy.copy_path}")


@cli.command()
@click.option("--media-id", required=True)
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def extract(media_id: str, workspace: str | None, config: str | None):
    """Extract frames, faces, metadata and audio from ingested media, then run forensics."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    total = 0
    for ext in _extractors(ws, cfg):
        result = ext.extract(media_id)
        total += len(result.artifacts)
        click.echo(f"  {ext.__class__.__name__}: {len(result.artifacts)} artifact(s)")
    click.echo(f"Total artifacts extracted: {total}")
    _run_forensics(ws, media_id, _media_type(ws, media_id))


@cli.command()
@click.option("--media-id", required=True)
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
@click.option("--json", "as_json", is_flag=True, help="Print the forensics report as JSON")
def forensics(media_id: str, workspace: str | None, config: str | None, as_json: bool):
    """Run the model-free forensic analyses alone (after `extract`)."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    report_obj = _run_forensics(ws, media_id, _media_type(ws, media_id))
    if as_json:
        click.echo(json.dumps(report_obj.model_dump(mode="json"), indent=2, sort_keys=True))
        return
    for obs in report_obj.observations:
        click.echo(f"  [{obs.severity:<7}] {obs.analyzer}: {obs.statement}")
    click.echo(f"Saved: {ForensicsRunner(ws).report_path(media_id)}")


@cli.command()
@click.option("--media-id", required=True)
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def compile(media_id: str, workspace: str | None, config: str | None):
    """Compile all findings for a media item."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    registry = DetectorRegistry(_detectors_dir(cfg))
    collection = FindingsCompiler(ws, registry.list_detectors()).compile(media_id)
    click.echo(f"Findings: {len(collection.findings)}")
    for f in collection.findings:
        click.echo(f"  {f.detector_id}: {f.classification.value} ({f.confidence:.4f})")


@cli.command()
@click.option("--media-id", required=True)
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
@click.option("--output", default=None, type=click.Path())
@click.option("--format", "fmt", default="pdf", type=click.Choice(["pdf", "html"]))
def report(media_id: str, workspace: str | None, config: str | None,
           output: str | None, fmt: str):
    """Generate a report for one media item."""
    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    registry = DetectorRegistry(_detectors_dir(cfg))
    generator = ReportGenerator(ws, registry.list_detectors())
    report_obj = generator.generate(media_id)
    report_dir = Path(output) if output else _output_dir(cfg, ws) / media_id
    report_dir.mkdir(parents=True, exist_ok=True)
    base = report_dir / "report"
    if fmt == "html":
        generator.save_html(report_obj, base.with_suffix(".html"))
        click.echo(f"Report: {base.with_suffix('.html')}")
    else:
        generator.save(report_obj, base.with_suffix(".html"))
        click.echo(f"Report: {base.with_suffix('.pdf')}")


# ---------------------------------------------------------------------- #
#  Weights                                                                #
# ---------------------------------------------------------------------- #


def _manifest(cfg: dict):
    from verity.weights import load_manifest
    return load_manifest(_detectors_dir(cfg) / "weights_manifest.yaml")


def _weights_ids(cfg: dict, detector: str | None, trained: bool, all_: bool) -> list[str] | None:
    registry = DetectorRegistry(_detectors_dir(cfg))
    if detector:
        ids = [d.strip() for d in detector.split(",") if d.strip()]
        unknown = sorted(set(ids) - set(registry.list_detectors()))
        if unknown:
            raise click.ClickException(f"Unknown detector(s): {', '.join(unknown)}")
        return ids
    if trained:
        return sorted(i for i, d in registry.list_detectors().items() if d.weights.is_trained)
    if all_:
        return None
    raise click.ClickException("Use --all, --trained or --detector <id>[,<id>...]")


@cli.group()
def weights():
    """Fetch, verify and bundle detector checkpoints (see detectors/weights_manifest.yaml)."""


@weights.command("pull")
@click.option("--all", "all_", is_flag=True, help="Every checkpoint in the manifest")
@click.option("--trained", is_flag=True, help="Only detectors with a trained head (the default evidence set)")
@click.option("--detector", default=None, help="One or more detector ids (comma-separated)")
@click.option("--source", default="auto", type=click.Choice(["auto", "hub", "origin"]),
              help="auto: Verity hub first, then the authors' URL")
@click.option("--force", is_flag=True, help="Re-download even if the local file verifies")
@click.option("--config", default=None, type=click.Path())
def weights_pull(all_: bool, trained: bool, detector: str | None, source: str, force: bool,
                 config: str | None):
    """Download the checkpoints the selected detectors need, verifying SHA-256."""
    from verity.weights import pull
    cfg = _load_config(config)
    manifest = _manifest(cfg)
    ids = _weights_ids(cfg, detector, trained, all_)
    if not manifest.hub_repo and source != "origin":
        click.echo("hub not configured (hub.huggingface_repo / VERITY_WEIGHTS_HUB); using origin URLs",
                   err=True)
    result = pull(manifest, _detectors_dir(cfg), ids, source=source, force=force,
                  log=lambda m: click.echo(m, err=True))
    click.echo(f"fetched {len(result.fetched)}, linked {len(result.linked)}, "
               f"already present {len(result.already)}, failed {len(result.failed)}")
    for key, why in sorted(result.failed.items()):
        click.echo(f"  FAILED {key}: {why}")
    sys.exit(0 if result.ok else 1)


@weights.command("status")
@click.option("--verify", "check", is_flag=True, help="Also check SHA-256 (slow for large files)")
@click.option("--config", default=None, type=click.Path())
def weights_status(check: bool, config: str | None):
    """Show which checkpoints are present (and, with --verify, intact)."""
    from verity.weights import status
    cfg = _load_config(config)
    manifest = _manifest(cfg)
    registry = DetectorRegistry(_detectors_dir(cfg))
    rows = status(manifest, _detectors_dir(cfg), None, check=check)
    click.echo(f"{'detector':<16} {'tier':<14} {'file':<44} {'size':>9}  state")
    for r in rows:
        d = registry.get(r.target.detector_id)
        tier = d.weights.tier if d else "?"
        if not r.present:
            state = "missing"
        elif r.verified is None:
            state = "present"
        else:
            state = "verified" if r.verified else "CORRUPT"
        click.echo(f"{r.target.detector_id:<16} {tier:<14} {r.target.dest:<44} "
                   f"{r.target.spec.size / 1e6:8.1f}M  {state}")
    for det_id, note in sorted(manifest.runtime_fetched.items()):
        d = registry.get(det_id)
        click.echo(f"{det_id:<16} {(d.weights.tier if d else '?'):<14} {'(fetched at first run)':<44} "
                   f"{'':>9}  {note[:60]}")
    bad = [r for r in rows if r.verified is False]
    sys.exit(1 if bad else 0)


@weights.command("verify")
@click.option("--config", default=None, type=click.Path())
def weights_verify(config: str | None):
    """Verify every present checkpoint against the manifest SHA-256."""
    from verity.weights import verify
    cfg = _load_config(config)
    rows = verify(_manifest(cfg), _detectors_dir(cfg))
    bad = [r for r in rows if r.present and not r.verified]
    missing = [r for r in rows if not r.present]
    for r in bad:
        click.echo(f"CORRUPT  {r.target.detector_id}/{r.target.dest}")
    click.echo(f"verified {len(rows) - len(bad) - len(missing)}, missing {len(missing)}, corrupt {len(bad)}")
    sys.exit(1 if bad else 0)


@weights.command("bundle")
@click.option("--repo", required=True, help="Hugging Face repo id the bundle will be uploaded to, e.g. you/verity-weights")
@click.option("--out", default="dist-weights", type=click.Path(), help="Output directory")
@click.option("--config", default=None, type=click.Path())
def weights_bundle(repo: str, out: str, config: str | None):
    """Lay out the Hugging Face model repo (files, model card, notices, checksums).

    Nothing is uploaded; the commands to publish are printed at the end."""
    from verity.weights import build_bundle
    cfg = _load_config(config)
    manifest = _manifest(cfg)
    out_dir = Path(out) / repo.split("/")[-1]
    info = build_bundle(manifest, _detectors_dir(cfg), out_dir, repo)
    click.echo(f"bundle: {out_dir}  ({info['total_bytes'] / 1e9:.2f} GB, {len(info['copied'])} files)")
    if info["missing"]:
        click.echo("missing locally (pull them first):")
        for m in info["missing"]:
            click.echo(f"  {m}")
    click.echo("\nTo publish (run yourself):")
    click.echo(f"  hf auth login")
    click.echo(f"  hf repo create {repo} --type model")
    click.echo(f"  hf upload {repo} {out_dir} . --repo-type model --commit-message 'Verity weights bundle'")
    click.echo(f"\nThen set hub.huggingface_repo: \"{repo}\" in detectors/weights_manifest.yaml")
    sys.exit(1 if info["missing"] else 0)


if __name__ == "__main__":
    cli()
