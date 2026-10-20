from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

MANIFEST_NAME = "weights_manifest.yaml"


@dataclass(frozen=True)
class Archive:
    sha256: str
    size: int
    member: str


@dataclass(frozen=True)
class FileSpec:
    name: str
    sha256: str
    size: int
    source: str
    origin: str
    redistributable: bool
    hub_path: str = ""
    description: str = ""
    license: str = ""          # overrides the source's license when set
    archive: Optional[Archive] = None

    def effective_license(self, manifest: "Manifest") -> str:
        if self.license:
            return self.license
        return manifest.sources.get(self.source, {}).get("license", "")


@dataclass(frozen=True)
class Target:
    """One file a detector needs, and where it must land."""
    detector_id: str
    spec: FileSpec
    dest: str  # relative to detectors/<id>/weights/


@dataclass
class Manifest:
    path: Path
    version: int
    hub_repo: str
    hub_revision: str
    sources: Dict[str, dict]
    files: Dict[str, FileSpec]
    detectors: Dict[str, List[Target]]
    runtime_fetched: Dict[str, str] = field(default_factory=dict)
    no_checkpoint: Dict[str, str] = field(default_factory=dict)

    # --------------------------------------------------------------- #

    def hub_url(self, spec: FileSpec) -> Optional[str]:
        if not (self.hub_repo and spec.redistributable and spec.hub_path):
            return None
        return (f"https://huggingface.co/{self.hub_repo}/resolve/"
                f"{self.hub_revision}/{spec.hub_path}")

    def targets_for(self, detector_ids: Optional[List[str]] = None) -> List[Target]:
        ids = sorted(self.detectors) if detector_ids is None else list(detector_ids)
        out: List[Target] = []
        for det_id in ids:
            out.extend(self.detectors.get(det_id, []))
        return out

    def redistributable_files(self) -> List[FileSpec]:
        return sorted((f for f in self.files.values() if f.redistributable),
                      key=lambda f: f.hub_path)


def load_manifest(path: Path) -> Manifest:
    data = yaml.safe_load(path.read_text()) or {}
    if int(data.get("version", 1)) < 2:
        raise ValueError(f"{path}: manifest version 2 required")

    files: Dict[str, FileSpec] = {}
    for name, raw in (data.get("files") or {}).items():
        archive = None
        if raw.get("archive"):
            a = raw["archive"]
            archive = Archive(sha256=a["sha256"], size=int(a["size"]), member=a["member"])
        files[name] = FileSpec(
            name=name,
            sha256=raw["sha256"],
            size=int(raw["size"]),
            source=raw.get("source", ""),
            origin=raw.get("origin", ""),
            redistributable=bool(raw.get("redistributable", False)),
            hub_path=raw.get("hub_path", ""),
            description=raw.get("description", ""),
            license=raw.get("license", ""),
            archive=archive,
        )

    detectors: Dict[str, List[Target]] = {}
    for det_id, entries in (data.get("detectors") or {}).items():
        targets = []
        for entry in entries or []:
            if isinstance(entry, str):
                name, dest = entry, entry
            else:
                name, dest = entry["file"], entry.get("dest", entry["file"])
            if name not in files:
                raise ValueError(f"{path}: detector {det_id} references unknown file {name}")
            targets.append(Target(detector_id=det_id, spec=files[name], dest=dest))
        detectors[det_id] = targets

    hub = data.get("hub") or {}
    return Manifest(
        path=path,
        version=int(data.get("version", 2)),
        hub_repo=os.environ.get("VERITY_WEIGHTS_HUB", "") or hub.get("huggingface_repo", "") or "",
        hub_revision=hub.get("revision", "main") or "main",
        sources=data.get("sources") or {},
        files=files,
        detectors=detectors,
        runtime_fetched=data.get("runtime_fetched") or {},
        no_checkpoint=data.get("no_checkpoint") or {},
    )
