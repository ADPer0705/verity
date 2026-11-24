"""Forensic evidence preservation bundle & air-gap transfer kit export/import.

1. ``export_evidence_bundle``: Packages a single examined case into a
   self-contained, deterministic ``.zip`` archive with ``MANIFEST.json`` and
   ``SHA256SUMS.txt`` (verifiable offline with standard ``sha256sum -c``).
2. ``export_airgap_kit`` / ``import_airgap_kit``: Transfers a complete
   provisioned Verity installation (binary, ``uv``, managed Python runtimes,
   verified checkpoints, detector lockfiles, ``uv`` wheel cache, and pre-warmed
   Hugging Face / Torch Hub model caches) to a permanently air-gapped forensic
   workstation and recreates every detector virtualenv offline via
   ``uv sync --offline --no-python-downloads``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from verity import __version__, paths
from verity.compile.findings_compiler import FindingsCompiler
from verity.detect.detector_registry import DetectorRegistry
from verity.forensics import load_report as load_forensics_report
from verity.ingest.media_ingestor import MediaIngestor
from verity.models.media import MediaInfo
from verity.runtime import env_manager

CHUNK = 1 << 20
LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(f"[verity] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  1. Forensic Evidence Preservation Bundle (.zip with SHA256SUMS.txt)        #
# --------------------------------------------------------------------------- #


def _resolve_media_file(media_dir: Path, media_id: str) -> Optional[Path]:
    if not media_dir.exists():
        return None
    for p in sorted(media_dir.iterdir()):
        if p.is_file() and p.stem == media_id:
            return p
    return None


def export_evidence_bundle(
    workspace: Path,
    output_dir: Path,
    detectors_dir: Path,
    media_id: str,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a self-contained chain-of-custody zip bundle for ``media_id``."""
    media_dir = workspace / "media"
    meta_path = media_dir / f".{media_id}.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No ingested media record for id: {media_id}")
    info = MediaInfo(**json.loads(meta_path.read_text()))
    working_copy = _resolve_media_file(media_dir, media_id)
    if working_copy is None:
        raise FileNotFoundError(f"Working copy missing for media id: {media_id}")

    actual_sha = MediaIngestor(workspace)._compute_checksum(working_copy)  # noqa: SLF001
    integrity_ok = actual_sha == info.checksum and working_copy.stat().st_size == info.size_bytes

    orig_name = info.metadata.get("original_name") or Path(info.original_path).name
    stem = Path(orig_name).stem or media_id
    report_dir = output_dir / media_id
    report_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_path or (report_dir / f"verity-evidence-{stem}-{media_id}.zip")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect (archive_relpath, source_file_path) pairs deterministically
    entries: List[tuple[str, Path]] = []

    # 1) Read-only working copy + ingest metadata
    entries.append((f"media/{orig_name}", working_copy))
    entries.append(("media/ingest_record.json", meta_path))

    # 2) Reports, analysis & selection
    for fname in ("report.html", "report.pdf", "analysis.json", "selection.json"):
        p = report_dir / fname
        if p.is_file():
            entries.append((f"report/{fname}", p))

    # 3) Model-free forensic outputs & diagnostic figures
    forensics_dir = workspace / "artifacts" / media_id / "forensics"
    if forensics_dir.is_dir():
        for p in sorted(forensics_dir.rglob("*")):
            if p.is_file():
                entries.append((f"forensics/{p.relative_to(forensics_dir).as_posix()}", p))

    # 4) Detector findings, raw outputs & explainability visual exhibits
    det_out_root = workspace / "output"
    prefix = f"{media_id}_"
    if det_out_root.is_dir():
        for subdir in sorted(det_out_root.iterdir()):
            if not subdir.is_dir() or not subdir.name.startswith(prefix):
                continue
            det_id = subdir.name[len(prefix):]
            for p in sorted(subdir.rglob("*")):
                if p.is_file():
                    entries.append((f"detectors/{det_id}/{p.relative_to(subdir).as_posix()}", p))

    # Compute SHA-256 of every staged file
    checksum_lines: List[str] = []
    file_records: List[Dict[str, Any]] = []
    for arcname, src in entries:
        digest = sha256_file(src)
        checksum_lines.append(f"{digest}  {arcname}")
        file_records.append({"path": arcname, "sha256": digest, "size_bytes": src.stat().st_size})

    # Compile summary metadata for MANIFEST.json
    registry = DetectorRegistry(detectors_dir) if detectors_dir.is_dir() else None
    defs = registry.list_detectors() if registry else {}
    findings_col = FindingsCompiler(workspace, defs).compile(media_id)
    forensics_rep = load_forensics_report(workspace, media_id)

    manifest = {
        "schema": "verity-evidence-bundle/1",
        "verity_version": __version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "offline_mode": paths.is_offline(),
        "media": {
            "media_id": media_id,
            "file_name": orig_name,
            "media_type": info.media_type.value,
            "mime_type": info.mime_type,
            "size_bytes": info.size_bytes,
            "ingested_sha256": info.checksum,
            "verified_sha256": actual_sha,
            "integrity_verified": integrity_ok,
        },
        "findings_summary": [
            {
                "detector_id": f.detector_id,
                "detector_version": f.detector_version,
                "tier": f.tier,
                "classification": f.classification.value,
                "confidence": f.confidence,
                "prediction": f.raw_output.get("prediction"),
            }
            for f in findings_col.findings
        ],
        "forensics_counts": forensics_rep.counts if forensics_rep else None,
        "file_count": len(file_records),
        "files": file_records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    checksum_lines.append(f"{manifest_sha}  MANIFEST.json")
    checksum_lines.sort(key=lambda line: line.split("  ", 1)[1])
    checksums_bytes = ("\n".join(checksum_lines) + "\n").encode("utf-8")

    # Write zip archive atomically
    tmp_zip = bundle_path.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("MANIFEST.json", manifest_bytes)
        zf.writestr("SHA256SUMS.txt", checksums_bytes)
        for arcname, src in entries:
            zf.write(src, arcname=arcname)
    tmp_zip.replace(bundle_path)

    return {
        "bundle_path": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
        "bundle_sha256": sha256_file(bundle_path),
        "file_count": len(entries) + 2,
        "integrity_verified": integrity_ok,
    }


# --------------------------------------------------------------------------- #
#  2. Cross-Machine Air-Gap Transfer Kit (export / import)                    #
# --------------------------------------------------------------------------- #


def _tar_dir(src_dir: Path, archive_path: Path, exclude_names: tuple[str, ...] = ()) -> bool:
    """Tar+gzip ``src_dir`` into ``archive_path``. Returns False if ``src_dir`` is empty/missing."""
    if not src_dir.is_dir():
        return False
    has_files = any(p.is_file() for p in src_dir.rglob("*"))
    if not has_files:
        return False
    tmp = archive_path.with_suffix(".tar.gz.part")
    with tarfile.open(tmp, "w:gz") as tf:
        for item in sorted(src_dir.iterdir()):
            if item.name in exclude_names:
                continue
            tf.add(item, arcname=item.name)
    tmp.replace(archive_path)
    return True


def export_airgap_kit(
    out_dir: Path,
    profile: str = "auto",
    detectors_dir: Optional[Path] = None,
    log: LogFn = _default_log,
) -> Dict[str, Any]:
    """Package everything needed to deploy Verity on a permanently air-gapped
    workstation (binary, uv, managed Python builds, verified checkpoints,
    detector lockfiles, uv wheel cache, and HF/Torch Hub model caches)."""
    ddir = detectors_dir or paths.detectors_dir()
    eff_profile = env_manager.detect_profile(profile, log=log)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_files: List[Path] = []

    # 1) Copy standalone binary if running frozen
    if paths.is_frozen():
        exe = Path(sys.executable)
        dest_exe = out_dir / "verity"
        shutil.copy2(exe, dest_exe)
        dest_exe.chmod(0o755)
        exported_files.append(dest_exe)
        log(f"copied standalone binary -> {dest_exe.name}")

    # 2) Private bin/ (uv, optional static ffmpeg/ffprobe)
    if _tar_dir(paths.bin_dir(), out_dir / "bin.tar.gz"):
        exported_files.append(out_dir / "bin.tar.gz")
        log("archived bin/ (uv / tools)")

    # 3) Managed standalone Python runtimes
    if _tar_dir(paths.python_dir(), out_dir / "python.tar.gz"):
        exported_files.append(out_dir / "python.tar.gz")
        log("archived python/ (managed standalone Python runtimes)")

    # 4) Verified model checkpoints
    weights_archive = out_dir / "weights.tar.gz"
    wroot = paths.weights_root()
    if wroot is not None and _tar_dir(wroot, weights_archive):
        exported_files.append(weights_archive)
        log("archived weights/ from VERITY_WEIGHTS_ROOT")
    else:
        # Collect from detectors/<id>/weights/
        tmp = weights_archive.with_suffix(".tar.gz.part")
        count = 0
        with tarfile.open(tmp, "w:gz") as tf:
            for det_dir in sorted(ddir.iterdir()):
                wdir = det_dir / "weights"
                if not wdir.is_dir():
                    continue
                for f in sorted(wdir.rglob("*")):
                    if f.is_file() and f.name != "README.md":
                        rel = f.relative_to(wdir)
                        tf.add(f, arcname=f"{det_dir.name}/{rel.as_posix()}")
                        count += 1
        if count > 0:
            tmp.replace(weights_archive)
            exported_files.append(weights_archive)
            log(f"archived {count} checkpoint file(s) -> weights.tar.gz")
        else:
            tmp.unlink(missing_ok=True)

    # 5) Detector lockfiles & pyproject copies (NOT .venv — rebuilt offline on target!)
    prof_envs = paths.envs_root() / eff_profile
    locked_detectors: List[str] = []
    if prof_envs.is_dir():
        locks_archive = out_dir / "locks.tar.gz"
        tmp = locks_archive.with_suffix(".tar.gz.part")
        with tarfile.open(tmp, "w:gz") as tf:
            for det_proj in sorted(prof_envs.iterdir()):
                if not det_proj.is_dir():
                    continue
                added = False
                for fname in ("pyproject.toml", "uv.lock"):
                    fp = det_proj / fname
                    if fp.is_file():
                        tf.add(fp, arcname=f"{eff_profile}/{det_proj.name}/{fname}")
                        added = True
                if added:
                    locked_detectors.append(det_proj.name)
        if locked_detectors:
            tmp.replace(locks_archive)
            exported_files.append(locks_archive)
            log(f"archived {len(locked_detectors)} detector lockfile(s) -> locks.tar.gz")
        else:
            tmp.unlink(missing_ok=True)

    # 6) uv wheel cache
    if _tar_dir(paths.uv_cache_dir(), out_dir / "cache-uv.tar.gz"):
        exported_files.append(out_dir / "cache-uv.tar.gz")
        log("archived uv wheel cache -> cache-uv.tar.gz")

    # 7) Pre-warmed Hugging Face + Torch Hub caches
    models_cache_archive = out_dir / "cache-models.tar.gz"
    tmp = models_cache_archive.with_suffix(".tar.gz.part")
    added_model_cache = False
    with tarfile.open(tmp, "w:gz") as tf:
        for name, cdir in (("huggingface", paths.hf_cache_dir()), ("torch", paths.torch_cache_dir())):
            if cdir.is_dir() and any(p.is_file() for p in cdir.rglob("*")):
                tf.add(cdir, arcname=name)
                added_model_cache = True
    if added_model_cache:
        tmp.replace(models_cache_archive)
        exported_files.append(models_cache_archive)
        log("archived HF / Torch Hub caches -> cache-models.tar.gz")
    else:
        tmp.unlink(missing_ok=True)

    # 8) Manifest + SHA256SUMS
    kit_manifest = {
        "schema": "verity-airgap-kit/1",
        "verity_version": __version__,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "machine": platform.machine(),
        "profile": eff_profile,
        "detectors": locked_detectors,
        "files": [p.name for p in sorted(exported_files, key=lambda x: x.name)],
    }
    manifest_path = out_dir / "airgap-manifest.json"
    manifest_path.write_text(json.dumps(kit_manifest, indent=2) + "\n")
    exported_files.append(manifest_path)

    sums_lines = [
        f"{sha256_file(p)}  {p.name}"
        for p in sorted(exported_files, key=lambda x: x.name)
    ]
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text("\n".join(sums_lines) + "\n")

    total_bytes = sum(p.stat().st_size for p in exported_files)
    log(f"air-gap kit ready at {out_dir} ({total_bytes / 1e9:.2f} GB, {len(exported_files)} files)")
    return {
        "out_dir": str(out_dir),
        "profile": eff_profile,
        "detectors": locked_detectors,
        "files": [p.name for p in sorted(exported_files, key=lambda x: x.name)],
        "total_bytes": total_bytes,
    }


def _safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")


def import_airgap_kit(
    kit_dir: Path,
    verify_checksums: bool = True,
    concurrency: int = 4,
    log: LogFn = _default_log,
) -> Dict[str, Any]:
    """Import an air-gap kit on an offline workstation and rebuild all detector
    environments locally using ``uv sync --offline --no-python-downloads``."""
    kit_dir = kit_dir.expanduser().resolve()
    sums_path = kit_dir / "SHA256SUMS"
    manifest_path = kit_dir / "airgap-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Not a valid air-gap kit (missing {manifest_path})")

    if verify_checksums and sums_path.is_file():
        log("verifying kit SHA256SUMS...")
        for line in sums_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            expected, fname = line.split(None, 1)
            target = kit_dir / fname.strip()
            if not target.is_file():
                raise FileNotFoundError(f"Missing kit archive: {fname}")
            actual = sha256_file(target)
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch for {fname}: expected {expected}, got {actual}")
        log("all kit checksums verified")

    kit_manifest = json.loads(manifest_path.read_text())
    profile = kit_manifest.get("profile", "auto")
    detectors = kit_manifest.get("detectors") or []

    # Restore tools, python, weights, caches and locks into platformdirs
    if (kit_dir / "bin.tar.gz").is_file():
        log(f"restoring bin/ -> {paths.bin_dir()}")
        _safe_extract_tar(kit_dir / "bin.tar.gz", paths.bin_dir())
        for exe in paths.bin_dir().iterdir():
            if exe.is_file():
                exe.chmod(0o755)
        paths.ensure_runtime_path()

    if (kit_dir / "python.tar.gz").is_file():
        log(f"restoring managed Python runtimes -> {paths.python_dir()}")
        _safe_extract_tar(kit_dir / "python.tar.gz", paths.python_dir())

    # Ensure detectors/ code is unpacked and weights symlinks are wired
    ddir = paths.detectors_dir()

    if (kit_dir / "weights.tar.gz").is_file():
        wroot = paths.weights_root()
        if wroot is not None:
            log(f"restoring checkpoints -> {wroot}")
            _safe_extract_tar(kit_dir / "weights.tar.gz", wroot)
        else:
            log(f"restoring checkpoints into {ddir}/*/weights")
            with tarfile.open(kit_dir / "weights.tar.gz", "r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    parts = Path(member.name).parts
                    if len(parts) < 2:
                        continue
                    det_id, rel = parts[0], Path(*parts[1:])
                    dest = ddir / det_id / "weights" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    if src is not None:
                        with open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)

    if (kit_dir / "cache-uv.tar.gz").is_file():
        log(f"restoring uv wheel cache -> {paths.uv_cache_dir()}")
        _safe_extract_tar(kit_dir / "cache-uv.tar.gz", paths.uv_cache_dir())

    if (kit_dir / "cache-models.tar.gz").is_file():
        log(f"restoring HF / Torch Hub model caches -> {paths.cache_dir()}")
        _safe_extract_tar(kit_dir / "cache-models.tar.gz", paths.cache_dir())

    if (kit_dir / "locks.tar.gz").is_file():
        log(f"restoring detector lockfiles -> {paths.envs_root()}")
        _safe_extract_tar(kit_dir / "locks.tar.gz", paths.envs_root())

    # Persist offline mode & hardware profile in user config
    paths.save_user_config({"offline": True, "hw_profile": profile})
    os.environ["VERITY_OFFLINE"] = "1"

    # Rebuild all detector virtualenvs locally via hardlinks from local uv cache (ZERO network!)
    sync_rc = 0
    if detectors:
        log(f"rebuilding {len(detectors)} detector virtualenvs offline via hardlinks...")
        sync_rc = env_manager.setup_detectors(
            detector_ids_list=detectors,
            profile=profile,
            force=True,
            concurrency=concurrency,
            offline=True,
            detectors_dir=ddir,
            log=log,
        )

    return {
        "profile": profile,
        "detectors_restored": len(detectors),
        "sync_returncode": sync_rc,
        "offline_sealed": True,
        "config_file": str(paths.config_file()),
    }
