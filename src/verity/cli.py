from __future__ import annotations
import json
import os
import sys
import yaml
import click
from pathlib import Path

from verity import __version__, paths
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
    user_cfg = paths.load_user_config()
    return {
        "detectors_dir": os.environ.get("VERITY_DETECTORS_DIR", "") or str(user_cfg.get("detectors_dir") or ""),
        "workspace": os.environ.get("VERITY_WORKSPACE", "") or str(user_cfg.get("workspace") or ""),
        "output_dir": os.environ.get("VERITY_OUTPUT_DIR", "") or str(user_cfg.get("output_dir") or ""),
        "extractors": user_cfg.get("extractors") or EXTRACTORS,
    }


def _load_config(config: str | None) -> dict:
    paths.ensure_runtime_path()
    cfg = _default_config()
    if config and Path(config).exists():
        with open(config) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if k in cfg})
    return cfg


def _detectors_dir(cfg: dict) -> Path:
    if cfg.get("detectors_dir"):
        return Path(cfg["detectors_dir"]).expanduser().resolve()
    ddir = paths.detectors_dir()
    if ddir.exists():
        return ddir
    raise click.ClickException(
        "Cannot locate detectors/. Run `verity setup` or set VERITY_DETECTORS_DIR."
    )


def _workspace(cfg: dict) -> Path:
    if cfg.get("workspace"):
        return Path(cfg["workspace"]).expanduser().resolve()
    return paths.workspace_dir()


def _output_dir(cfg: dict, workspace: Path) -> Path:
    if cfg.get("output_dir"):
        return Path(cfg["output_dir"]).expanduser().resolve()
    return paths.output_dir(workspace)


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
@click.version_option(version=__version__, prog_name="verity")
def cli():
    """Verity — deterministic, air-gap-ready forensic deepfake examination pipeline."""
    paths.ensure_runtime_path()


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
    selector = DetectorSelector(
        registry,
        offline=paths.is_offline(),
        include_untrained=include_untrained or bool(detector),
    )
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
@click.option("--all", "all_", is_flag=True, help="Provision all 58 detector environments")
@click.option("--trained", is_flag=True,
              help="Provision all 39 evidence-grade detectors with trained heads (default)")
@click.option("--detector", default=None, help="Provision one or more detectors (comma-separated)")
@click.option("--profile", default="auto", type=click.Choice(["auto", "cuda", "cpu"]),
              show_default=True, help="Hardware profile (cuda, cpu, or auto-detect)")
@click.option("--force", is_flag=True, help="Re-provision environments even if up to date")
@click.option("--concurrency", type=int, default=4, show_default=True,
              help="Number of detector environments to build in parallel")
@click.option("--skip-weights", is_flag=True,
              help="Skip downloading and verifying model checkpoints")
@click.option("--skip-warmup", is_flag=True,
              help="Skip pre-warming Hugging Face / Torch Hub runtime caches")
@click.option("--skip-ffmpeg", is_flag=True,
              help="Skip auto-fetching static ffmpeg/ffprobe if absent on host")
@click.option("--airgap-ready", is_flag=True,
              help="Verify all checkpoints and seal offline mode in user config so the workstation can disconnect from the internet")
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def setup(all_: bool, trained: bool, detector: str | None, profile: str, force: bool,
          concurrency: int, skip_weights: bool, skip_warmup: bool, skip_ffmpeg: bool,
          airgap_ready: bool, workspace: str | None, config: str | None):
    """One-command workstation setup: bootstraps uv/Python, builds detector
    environments, downloads & verifies checkpoints, and warms runtime caches
    for air-gapped forensic operation."""
    from verity.runtime import env_manager
    from verity.weights import pull as weights_pull_fn, verify as weights_verify_fn

    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace

    if paths.is_standalone_mode():
        click.echo(f"[setup] data dir  : {paths.data_dir()}")
        click.echo(f"[setup] cache dir : {paths.cache_dir()}")
        click.echo(f"[setup] config    : {paths.config_file()}")
        paths.ensure_bundled_detectors_extracted(force=force)

    ddir = _detectors_dir(cfg)
    registry = DetectorRegistry(ddir)

    if detector:
        ids = [d.strip() for d in detector.split(",") if d.strip()]
    elif all_:
        ids = sorted(registry.list_detectors())
    else:
        # Default to --trained (evidence-grade set) when no filter flag is given
        ids = sorted(i for i, d in registry.list_detectors().items() if d.weights.is_trained)

    unknown = sorted(set(ids) - set(registry.list_detectors()))
    if unknown:
        raise click.ClickException(f"Unknown detector(s): {', '.join(unknown)}")

    eff_profile = env_manager.detect_profile(profile)
    click.echo(f"[setup] profile   : {eff_profile}")
    click.echo(f"[setup] detectors : {len(ids)}")

    # 1. Bootstrap uv (+ optional static ffmpeg/ffprobe)
    uv_bin = env_manager.ensure_uv(allow_download=True)
    click.echo(f"[setup] uv        : {uv_bin}")
    if not skip_ffmpeg:
        ff, fp = env_manager.ensure_ffmpeg(allow_download=True)
        if ff and fp:
            click.echo(f"[setup] ffmpeg    : {ff}")

    # 2. Provision isolated detector virtualenvs via uv
    click.echo(f"\n[setup 1/3] Provisioning {len(ids)} isolated detector environment(s)...")
    rc = env_manager.setup_detectors(
        detector_ids_list=ids,
        profile=eff_profile,
        force=force,
        concurrency=concurrency,
        offline=False,
        detectors_dir=ddir,
    )
    if rc != 0:
        raise click.ClickException("One or more detector environments failed to provision.")

    # 3. Download & SHA-256-verify checkpoints from HF mirror + upstream origins
    if not skip_weights:
        click.echo("\n[setup 2/3] Fetching and SHA-256 verifying model checkpoints...")
        manifest = _manifest(cfg)
        wres = weights_pull_fn(
            manifest, ddir, ids, source="auto", force=False,
            log=lambda m: click.echo(m, err=True),
        )
        click.echo(
            f"[weights] fetched {len(wres.fetched)}, linked {len(wres.linked)}, "
            f"already verified {len(wres.already)}, failed {len(wres.failed)}"
        )
        for key, why in sorted(wres.failed.items()):
            click.echo(f"  FAILED {key}: {why}", err=True)

    # 4. Warm up runtime-fetched HF / Torch Hub foundation caches
    if not skip_warmup:
        click.echo("\n[setup 3/3] Warming runtime Hugging Face / Torch Hub caches...")
        wstatus = env_manager.warmup_detectors(
            detector_ids_list=ids,
            profile=eff_profile,
            detectors_dir=ddir,
        )
        ok_n = sum(1 for v in wstatus.values() if v == "ok")
        fail_n = sum(1 for v in wstatus.values() if v.startswith("failed"))
        click.echo(f"[warmup] {ok_n} cached, {fail_n} failed, {len(wstatus) - ok_n - fail_n} skipped")

    # 5. Persist hardware profile & optionally seal air-gap offline mode
    user_patch = {"hw_profile": eff_profile}
    if airgap_ready:
        click.echo("\n[setup] Verifying checkpoint integrity before air-gap seal...")
        rows = weights_verify_fn(_manifest(cfg), ddir, ids)
        corrupt = [r for r in rows if r.present and not r.verified]
        missing = [r for r in rows if not r.present]
        if corrupt:
            for r in corrupt:
                click.echo(f"  CORRUPT {r.target.detector_id}/{r.target.dest}", err=True)
            raise click.ClickException("Corrupt checkpoints found; refusing to seal air-gap mode.")
        if missing:
            click.echo(
                f"  WARN: {len(missing)} checkpoint(s) still missing; affected detectors will be skipped offline.",
                err=True,
            )
        user_patch["offline"] = True
        cfg_saved = paths.save_user_config(user_patch)
        click.echo(f"\n[setup] Sealed for air-gapped operation (offline: true in {cfg_saved}).")
        click.echo("        You may now disconnect this workstation from the network.")
    else:
        paths.save_user_config(user_patch)
        click.echo("\n[setup] Workstation setup complete. Run `verity doctor` or `verity serve`.")


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


@weights.command("warmup")
@click.option("--all", "all_", is_flag=True, help="Warm runtime caches for all detectors")
@click.option("--trained", is_flag=True, help="Warm runtime caches for trained detectors (default)")
@click.option("--detector", default=None, help="Comma-separated detector ids")
@click.option("--profile", default="auto", type=click.Choice(["auto", "cuda", "cpu"]))
@click.option("--config", default=None, type=click.Path())
def weights_warmup(all_: bool, trained: bool, detector: str | None, profile: str, config: str | None):
    """Pre-download runtime Hugging Face / Torch Hub / FAN caches for offline use."""
    from verity.runtime import env_manager

    cfg = _load_config(config)
    ddir = _detectors_dir(cfg)
    ids = _weights_ids(cfg, detector, trained or not (detector or all_), all_)
    statuses = env_manager.warmup_detectors(
        detector_ids_list=ids,
        trained_only=not (detector or all_),
        profile=profile,
        detectors_dir=ddir,
    )
    failed = {k: v for k, v in statuses.items() if v.startswith("failed")}
    for k, v in sorted(statuses.items()):
        click.echo(f"  {k:<20} {v}")
    sys.exit(1 if failed else 0)


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


@cli.command("export")
@click.option("--media-id", required=True, help="Case media id to package")
@click.option("--out", default=None, type=click.Path(), help="Destination zip path")
@click.option("--workspace", default=None, type=click.Path())
@click.option("--config", default=None, type=click.Path())
def export_cmd(media_id: str, out: str | None, workspace: str | None, config: str | None):
    """Export a self-contained forensic evidence preservation bundle (.zip) with
    MANIFEST.json and SHA256SUMS.txt."""
    from verity.export import export_evidence_bundle

    cfg = _load_config(config)
    if workspace:
        cfg["workspace"] = workspace
    ws = _workspace(cfg)
    res = export_evidence_bundle(
        workspace=ws,
        output_dir=_output_dir(cfg, ws),
        detectors_dir=_detectors_dir(cfg),
        media_id=media_id,
        out_path=Path(out) if out else None,
    )
    click.echo(f"Evidence bundle : {res['bundle_path']}")
    click.echo(f"Bundle SHA-256  : {res['bundle_sha256']}")
    click.echo(f"Files archived  : {res['file_count']} ({res['size_bytes'] / 1e6:.2f} MB)")
    click.echo(f"Media integrity : {'VERIFIED' if res['integrity_verified'] else 'MISMATCH!'}")
    sys.exit(0 if res["integrity_verified"] else 1)


@cli.command("doctor")
@click.option("--verify", "check_sha", is_flag=True, help="Verify SHA-256 checksums of present checkpoints")
@click.option("--config", default=None, type=click.Path())
def doctor(check_sha: bool, config: str | None):
    """Verify workstation readiness, tools, weights, environments and air-gap status."""
    import shutil
    from verity.runtime import env_manager
    from verity.weights import status as weights_status_fn

    cfg = _load_config(config)
    ddir = _detectors_dir(cfg)
    ws = _workspace(cfg)
    registry = DetectorRegistry(ddir)
    defs = registry.list_detectors()
    trained_ids = sorted(i for i, d in defs.items() if d.weights.is_trained)
    profile = env_manager.detect_profile()

    click.echo(f"Verity v{__version__} ({'standalone' if paths.is_standalone_mode() else 'dev'})")
    click.echo(f"  offline mode   : {'ENABLED (air-gap sealed)' if paths.is_offline() else 'disabled (online)'}")
    click.echo(f"  hw profile     : {profile}")
    click.echo(f"  data dir       : {paths.data_dir()}")
    click.echo(f"  cache dir      : {paths.cache_dir()}")
    click.echo(f"  config file    : {paths.config_file()}")
    click.echo(f"  detectors dir  : {ddir} ({len(defs)} total, {len(trained_ids)} evidence-grade)")
    click.echo(f"  workspace dir  : {ws}")

    uv_bin = env_manager.find_uv()
    ffmpeg_bin = shutil.which("ffmpeg")
    ffprobe_bin = shutil.which("ffprobe")
    click.echo(f"  uv             : {uv_bin or 'MISSING (run `verity setup`)'}")
    click.echo(f"  ffmpeg/ffprobe : {ffmpeg_bin or 'MISSING'} / {ffprobe_bin or 'MISSING'}")

    try:
        import weasyprint  # noqa: F401
        click.echo("  PDF export     : WeasyPrint OK")
    except Exception as exc:
        click.echo(f"  PDF export     : unavailable ({exc})")

    envs_dir = paths.envs_root() / profile
    ready_envs = 0
    if envs_dir.is_dir():
        for det_id in trained_ids:
            if (envs_dir / det_id / ".venv").is_dir():
                ready_envs += 1
    click.echo(f"  trained envs   : {ready_envs}/{len(trained_ids)} provisioned ({envs_dir})")

    manifest = _manifest(cfg)
    rows = weights_status_fn(manifest, ddir, trained_ids, check=check_sha)
    present = sum(1 for r in rows if r.present)
    corrupt = sum(1 for r in rows if r.verified is False)
    click.echo(
        f"  checkpoints    : {present}/{len(rows)} trained present"
        + (f", {corrupt} CORRUPT" if check_sha else " (use --verify for SHA-256)")
    )

    hf_files = sum(1 for p in paths.hf_cache_dir().rglob("*") if p.is_file()) if paths.hf_cache_dir().is_dir() else 0
    torch_files = sum(1 for p in paths.torch_cache_dir().rglob("*") if p.is_file()) if paths.torch_cache_dir().is_dir() else 0
    click.echo(f"  runtime caches : {hf_files} HF file(s), {torch_files} Torch file(s)")

    usage = shutil.disk_usage(ws.parent if ws.parent.exists() else Path.cwd())
    click.echo(f"  free disk      : {usage.free / (1024 ** 3):.1f} GB")

    sys.exit(1 if corrupt > 0 or not uv_bin else 0)


@cli.group()
def airgap():
    """Export or import a complete offline Verity transfer kit (binary, uv,
    Python runtimes, verified checkpoints, wheel cache, and HF/Torch caches)."""


@airgap.command("export")
@click.option("--out", required=True, type=click.Path(), help="Destination folder (e.g. USB drive path)")
@click.option("--profile", default="auto", type=click.Choice(["auto", "cuda", "cpu"]))
@click.option("--config", default=None, type=click.Path())
def airgap_export(out: str, profile: str, config: str | None):
    """Package this provisioned workstation into a verifiable offline transfer kit."""
    from verity.export import export_airgap_kit

    cfg = _load_config(config)
    res = export_airgap_kit(
        out_dir=Path(out),
        profile=profile,
        detectors_dir=_detectors_dir(cfg),
    )
    click.echo(f"\nAir-gap kit exported to: {res['out_dir']}")
    click.echo("Carry this folder to the air-gapped workstation and run:")
    click.echo(f"  {Path(res['out_dir']) / 'verity'} airgap import --from {res['out_dir']}")


@airgap.command("import")
@click.option("--from", "kit_dir", required=True, type=click.Path(exists=True),
              help="Directory containing the exported air-gap kit")
@click.option("--skip-verify", is_flag=True, help="Skip SHA256SUMS verification (not recommended)")
@click.option("--concurrency", type=int, default=4, show_default=True)
def airgap_import(kit_dir: str, skip_verify: bool, concurrency: int):
    """Restore an air-gap kit onto this offline workstation and rebuild detector envs locally."""
    from verity.export import import_airgap_kit

    res = import_airgap_kit(
        kit_dir=Path(kit_dir),
        verify_checksums=not skip_verify,
        concurrency=concurrency,
    )
    click.echo(f"\nRestored {res['detectors_restored']} detector environment(s) on profile '{res['profile']}'.")
    click.echo(f"Offline mode sealed in {res['config_file']}.")
    click.echo("Run `verity doctor` to confirm readiness, then `verity serve` to open the console.")
    sys.exit(res["sync_returncode"])


@cli.command()
@click.option("--host", default=None, help="Bind address (default: $VERITY_HOST or 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Bind port (default: $VERITY_PORT or 8000)")
@click.option("--reload", "reload_", is_flag=True, help="Enable auto-reload for development")
@click.option("--offline", is_flag=True, help="Enforce strict offline air-gapped mode for this session")
def serve(host: str | None, port: int | None, reload_: bool, offline: bool):
    """Launch the Verity HTTP API and Web UI console (same as `verity-server`)."""
    if host:
        os.environ["VERITY_HOST"] = host
    if port is not None:
        os.environ["VERITY_PORT"] = str(port)
    if reload_:
        os.environ["VERITY_RELOAD"] = "1"
    if offline or paths.is_offline():
        os.environ["VERITY_OFFLINE"] = "1"
    from verity_api.main import main as run_server

    run_server()


if __name__ == "__main__":
    cli()
