from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from verity.weights.manifest import FileSpec, Manifest, Target

CHUNK = 1 << 20
USER_AGENT = "verity-weights/2 (+https://github.com/ADPer0705/verity)"

Log = Callable[[str], None]


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- #
#  Locations                                                             #
# --------------------------------------------------------------------- #


def weights_dir(detectors_dir: Path, detector_id: str) -> Path:
    """Where ``detectors/<id>/weights`` really lives.

    With ``VERITY_WEIGHTS_ROOT`` set (the container), weights persist under
    ``$VERITY_WEIGHTS_ROOT/<id>`` and the package directory is a symlink to
    it, exactly as ``scripts/verity.py`` arranges. Local dev keeps them in
    the package.
    """
    pkg = detectors_dir / detector_id / "weights"
    root = os.environ.get("VERITY_WEIGHTS_ROOT", "")
    if not root:
        pkg.mkdir(parents=True, exist_ok=True)
        return pkg
    target = Path(root) / detector_id
    target.mkdir(parents=True, exist_ok=True)
    if pkg.is_symlink():
        return target
    if pkg.exists():
        readme = pkg / "README.md"
        if readme.exists() and not (target / "README.md").exists():
            shutil.copy2(readme, target / "README.md")
        shutil.rmtree(pkg)
    pkg.parent.mkdir(parents=True, exist_ok=True)
    pkg.symlink_to(target, target_is_directory=True)
    return target


def dest_path(detectors_dir: Path, target: Target) -> Path:
    return weights_dir(detectors_dir, target.detector_id) / target.dest


# --------------------------------------------------------------------- #
#  Status / verify                                                       #
# --------------------------------------------------------------------- #


@dataclass
class FileStatus:
    target: Target
    path: Path
    present: bool
    verified: Optional[bool]  # None when not checked
    size: int = 0

    @property
    def ok(self) -> bool:
        return self.present and self.verified is not False


def status(manifest: Manifest, detectors_dir: Path,
           detector_ids: Optional[List[str]] = None, check: bool = False) -> List[FileStatus]:
    out = []
    for target in manifest.targets_for(detector_ids):
        path = dest_path(detectors_dir, target)
        present = path.is_file()
        verified = None
        size = path.stat().st_size if present else 0
        if present and check:
            verified = size == target.spec.size and sha256_of(path) == target.spec.sha256
        out.append(FileStatus(target=target, path=path, present=present, verified=verified, size=size))
    return out


def verify(manifest: Manifest, detectors_dir: Path,
           detector_ids: Optional[List[str]] = None) -> List[FileStatus]:
    return status(manifest, detectors_dir, detector_ids, check=True)


# --------------------------------------------------------------------- #
#  Pull                                                                  #
# --------------------------------------------------------------------- #


@dataclass
class PullResult:
    fetched: List[Target] = field(default_factory=list)
    linked: List[Target] = field(default_factory=list)
    already: List[Target] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)  # "<det>/<dest>" -> reason

    @property
    def ok(self) -> bool:
        return not self.failed


def _download(url: str, dest: Path, expected_size: Optional[int], log: Log) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = expected_size or int(resp.headers.get("Content-Length") or 0)
        done = 0
        last = -1
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 // total
                if pct // 10 != last // 10:
                    log(f"    {pct:3d}%  {done / 1e6:8.1f} / {total / 1e6:.1f} MB")
                    last = pct
    tmp.replace(dest)


def _verified(path: Path, spec_sha: str, spec_size: int) -> bool:
    return path.is_file() and path.stat().st_size == spec_size and sha256_of(path) == spec_sha


def _fetch_file(manifest: Manifest, spec: FileSpec, dest: Path, source: str, log: Log,
                archive_cache: Dict[str, Path]) -> str:
    """Download ``spec`` into ``dest`` (verified). Returns the URL used."""
    candidates: List[str] = []
    hub = manifest.hub_url(spec)
    if source in ("auto", "hub") and hub:
        candidates.append(hub)
    if source in ("auto", "origin") and spec.origin:
        candidates.append(spec.origin)
    if not candidates:
        raise RuntimeError(
            "no download source: "
            + ("the Verity hub is not configured (set hub.huggingface_repo or "
               "VERITY_WEIGHTS_HUB)" if source == "hub" else "no origin URL in the manifest"))

    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in candidates:
        try:
            if spec.archive is not None and url == spec.origin:
                archive_path = archive_cache.get(url)
                if archive_path is None or not _verified(archive_path, spec.archive.sha256, spec.archive.size):
                    archive_path = Path(tempfile.gettempdir()) / f"verity-{spec.archive.sha256[:16]}.zip"
                    if not _verified(archive_path, spec.archive.sha256, spec.archive.size):
                        log(f"  archive {url}")
                        _download(url, archive_path, spec.archive.size, log)
                        if not _verified(archive_path, spec.archive.sha256, spec.archive.size):
                            raise RuntimeError("archive checksum mismatch")
                    archive_cache[url] = archive_path
                with zipfile.ZipFile(archive_path) as zf, zf.open(spec.archive.member) as src, \
                        open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, CHUNK)
            else:
                log(f"  {url}")
                _download(url, dest, spec.size, log)
            if not _verified(dest, spec.sha256, spec.size):
                dest.unlink(missing_ok=True)
                raise RuntimeError("checksum mismatch after download")
            return url
        except Exception as exc:  # noqa: BLE001 - try the next source
            errors.append(f"{url}: {exc}")
    raise RuntimeError("; ".join(errors))


def pull(manifest: Manifest, detectors_dir: Path, detector_ids: Optional[List[str]] = None,
         source: str = "auto", force: bool = False, log: Log = _log_stderr) -> PullResult:
    """Fetch every checkpoint the given detectors need.

    Identical files needed by several detectors (the I3D backbone, for one)
    are downloaded once and hard-linked, falling back to a copy across
    filesystems.
    """
    result = PullResult()
    by_sha: Dict[str, Path] = {}
    archive_cache: Dict[str, Path] = {}

    # Anything already verified locally is a link source for its twins.
    targets = manifest.targets_for(detector_ids)
    for t in targets:
        p = dest_path(detectors_dir, t)
        if not force and _verified(p, t.spec.sha256, t.spec.size):
            by_sha.setdefault(t.spec.sha256, p)

    for t in targets:
        key = f"{t.detector_id}/{t.dest}"
        p = dest_path(detectors_dir, t)
        if not force and _verified(p, t.spec.sha256, t.spec.size):
            result.already.append(t)
            continue
        twin = by_sha.get(t.spec.sha256)
        if twin is not None and twin != p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.unlink(missing_ok=True)
            try:
                os.link(twin, p)
            except OSError:
                shutil.copy2(twin, p)
            log(f"[weights] {key}: linked from {twin}")
            result.linked.append(t)
            continue
        log(f"[weights] {key} ({t.spec.size / 1e6:.1f} MB)")
        try:
            _fetch_file(manifest, t.spec, p, source, log, archive_cache)
        except Exception as exc:  # noqa: BLE001
            result.failed[key] = str(exc)
            log(f"[weights] FAILED {key}: {exc}")
            continue
        by_sha[t.spec.sha256] = p
        result.fetched.append(t)
    return result
