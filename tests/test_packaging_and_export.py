"""Unit tests for platformdirs paths, evidence preservation bundle, and weights bundling."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from verity import paths
from verity.export import export_evidence_bundle
from verity.models.findings import Classification, Finding, Reasoning
from verity.weights import build_bundle, load_manifest
from verity_api.main import app


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (80, 120, 200)).save(path)


def test_platformdirs_verity_home_and_offline_config(tmp_path, monkeypatch):
    home = tmp_path / "verity-home"
    monkeypatch.setenv("VERITY_HOME", str(home))
    monkeypatch.delenv("VERITY_OFFLINE", raising=False)

    assert paths.is_standalone_mode() is True
    assert paths.data_dir() == home / "data"
    assert paths.cache_dir() == home / "cache"
    assert paths.config_dir() == home / "config"
    assert paths.is_offline() is False

    cfg_path = paths.save_user_config({"offline": True, "hw_profile": "cpu", "max_frames": 16})
    assert cfg_path.is_file()
    assert paths.is_offline() is True
    loaded = paths.load_user_config()
    assert loaded["hw_profile"] == "cpu"
    assert loaded["max_frames"] == 16


def test_export_evidence_bundle_and_checksums(tmp_path):
    ws = tmp_path / "ws"
    media_dir = ws / "media"
    media_dir.mkdir(parents=True)

    media_id = "abc123def456"
    raw_bytes = b"synthetic-test-media-bytes"
    digest = hashlib.sha256(raw_bytes).hexdigest()
    (media_dir / f"{media_id}.mp4").write_bytes(raw_bytes)
    (media_dir / f".{media_id}.json").write_text(
        json.dumps(
            {
                "original_path": "/evidence/suspect.mp4",
                "media_type": "video",
                "checksum": digest,
                "size_bytes": len(raw_bytes),
                "mime_type": "video/mp4",
                "metadata": {"original_name": "suspect.mp4"},
            }
        )
    )

    # Report + forensics + detector outputs
    report_dir = ws / "output" / media_id
    report_dir.mkdir(parents=True)
    (report_dir / "report.html").write_text("<html><body>Verity Report</body></html>")

    forensics_dir = ws / "artifacts" / media_id / "forensics"
    forensics_dir.mkdir(parents=True)
    (forensics_dir / "forensics.json").write_text(
        json.dumps({"media_id": media_id, "media_type": "video", "results": []})
    )
    _png(forensics_dir / "ela.png")

    det_dir = ws / "output" / f"{media_id}_xception"
    det_dir.mkdir(parents=True)
    _png(det_dir / "timeline.png")
    _png(det_dir / "full_frames" / "f00000_face0_frame.png")
    finding = Finding(
        detector_id="xception",
        detector_version="1.0.0",
        classification=Classification.FAKE,
        confidence=0.91,
        reasoning=Reasoning(summary="Xception FAKE"),
        raw_output={"prediction": 0.91},
    )
    (det_dir / "finding.json").write_text(finding.model_dump_json(indent=2))

    res = export_evidence_bundle(
        workspace=ws,
        output_dir=ws / "output",
        detectors_dir=Path("detectors"),
        media_id=media_id,
    )
    assert res["integrity_verified"] is True
    zip_path = Path(res["bundle_path"])
    assert zip_path.is_file()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "MANIFEST.json" in names
        assert "SHA256SUMS.txt" in names
        assert "media/suspect.mp4" in names
        assert "report/report.html" in names
        assert "forensics/ela.png" in names
        assert "detectors/xception/finding.json" in names
        assert "detectors/xception/full_frames/f00000_face0_frame.png" in names

        # Every checksum in SHA256SUMS.txt must match the archived bytes
        sums = zf.read("SHA256SUMS.txt").decode().strip().splitlines()
        assert len(sums) >= 7
        for line in sums:
            expected_sha, arcname = line.split("  ", 1)
            actual_sha = hashlib.sha256(zf.read(arcname)).hexdigest()
            assert actual_sha == expected_sha, f"Mismatch for {arcname}"


def test_weights_bundle_gitattributes_and_model_card(tmp_path):
    manifest = load_manifest(Path("detectors/weights_manifest.yaml"))
    out_dir = tmp_path / "hf-bundle"
    build_bundle(manifest, Path("detectors"), out_dir, "ADPer/verity-weights")

    attrs = (out_dir / ".gitattributes").read_text()
    for ext in ("*.pth", "*.safetensors", "*.onnx", "*.model", "*.tar"):
        assert ext in attrs

    card = (out_dir / "README.md").read_text()
    assert "re-hosts other people's work" in card
    assert "DF40" in card or "NPR" in card


def test_api_export_404_on_unknown_case():
    client = TestClient(app)
    resp = client.get("/api/media/nonexistent_case_id/export")
    assert resp.status_code == 404
