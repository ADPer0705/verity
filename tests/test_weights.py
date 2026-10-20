"""Checkpoint manifest, pull/verify and bundle — all offline via file:// URLs."""
import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import pytest
import yaml

from verity.weights import build_bundle, load_manifest, pull, status, verify


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp: Path, origin_dir: Path, hub_dir: Path | None = None) -> Path:
    a = b"alpha-weights" * 100
    b = b"beta-weights" * 50
    (origin_dir / "a.pth").write_bytes(a)
    (origin_dir / "b.pth").write_bytes(b)
    with zipfile.ZipFile(origin_dir / "pre.zip", "w") as zf:
        zf.writestr("pretrained/c.pth", b"gamma" * 20)
    zip_bytes = (origin_dir / "pre.zip").read_bytes()
    data = {
        "version": 2,
        "hub": {"huggingface_repo": "", "revision": "main"},
        "sources": {"src": {"name": "Src", "url": "https://example.org", "license": "MIT",
                            "copyright": "(c) Src", "citation": "Src et al. 2024"}},
        "files": {
            "a.pth": {"sha256": _sha(a), "size": len(a), "source": "src",
                      "origin": (origin_dir / "a.pth").as_uri(), "redistributable": True,
                      "hub_path": "src/a.pth", "description": "A"},
            "b.pth": {"sha256": _sha(b), "size": len(b), "source": "src",
                      "origin": (origin_dir / "b.pth").as_uri(), "redistributable": False,
                      "description": "B"},
            "c.pth": {"sha256": _sha(b"gamma" * 20), "size": 100, "source": "src",
                      "origin": (origin_dir / "pre.zip").as_uri(), "redistributable": True,
                      "hub_path": "src/c.pth",
                      "archive": {"sha256": _sha(zip_bytes), "size": len(zip_bytes),
                                  "member": "pretrained/c.pth"}},
        },
        "detectors": {
            "det_a": ["a.pth"],
            "det_a2": ["a.pth"],                      # shares a.pth with det_a
            "det_b": ["b.pth"],
            "det_c": [{"file": "c.pth", "dest": "pretrained/c.pth"}],
        },
        "runtime_fetched": {"det_r": "something from HF"},
    }
    path = tmp / "weights_manifest.yaml"
    path.write_text(yaml.safe_dump(data))
    for det in data["detectors"]:
        (tmp / det).mkdir()
        (tmp / det / "detector.yaml").write_text(yaml.safe_dump(
            {"detector_id": det, "version": "1.0.0",
             "artifacts": {"required": [{"type": "face_crop", "params": {}}]}}))
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("VERITY_WEIGHTS_ROOT", raising=False)
    monkeypatch.delenv("VERITY_WEIGHTS_HUB", raising=False)
    detectors = tmp_path / "detectors"
    detectors.mkdir()
    origin = tmp_path / "origin"
    origin.mkdir()
    path = _manifest(detectors, origin)
    return detectors, origin, load_manifest(path)


class TestManifest:
    def test_load(self, env):
        detectors, origin, m = env
        assert set(m.files) == {"a.pth", "b.pth", "c.pth"}
        assert [t.dest for t in m.detectors["det_c"]] == ["pretrained/c.pth"]
        assert m.hub_url(m.files["a.pth"]) is None            # hub not configured
        assert [f.name for f in m.redistributable_files()] == ["a.pth", "c.pth"]

    def test_hub_url_and_env_override(self, env, monkeypatch):
        detectors, origin, m = env
        monkeypatch.setenv("VERITY_WEIGHTS_HUB", "someone/verity-weights")
        m = load_manifest(m.path)
        assert m.hub_url(m.files["a.pth"]) == \
            "https://huggingface.co/someone/verity-weights/resolve/main/src/a.pth"
        assert m.hub_url(m.files["b.pth"]) is None            # not redistributable

    def test_unknown_file_rejected(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump({"version": 2, "files": {}, "detectors": {"x": ["nope.pth"]}}))
        with pytest.raises(ValueError):
            load_manifest(p)

    def test_v1_rejected(self, tmp_path):
        p = tmp_path / "m.yaml"
        p.write_text(yaml.safe_dump({"release_v101": "x", "detectors": {}}))
        with pytest.raises(ValueError):
            load_manifest(p)


class TestPull:
    def test_pull_links_and_extracts(self, env):
        detectors, origin, m = env
        result = pull(m, detectors, None, source="origin", log=lambda _m: None)
        assert result.ok
        assert sorted(t.detector_id for t in result.fetched) == ["det_a", "det_b", "det_c"]
        assert [t.detector_id for t in result.linked] == ["det_a2"]
        a1 = detectors / "det_a" / "weights" / "a.pth"
        a2 = detectors / "det_a2" / "weights" / "a.pth"
        assert a1.read_bytes() == a2.read_bytes()
        assert os.stat(a1).st_ino == os.stat(a2).st_ino          # hard-linked, stored once
        assert (detectors / "det_c" / "weights" / "pretrained" / "c.pth").read_bytes() == b"gamma" * 20
        assert all(r.verified for r in verify(m, detectors))

    def test_second_pull_is_a_noop(self, env):
        detectors, origin, m = env
        pull(m, detectors, None, source="origin", log=lambda _m: None)
        again = pull(m, detectors, None, source="origin", log=lambda _m: None)
        assert not again.fetched and not again.linked and len(again.already) == 4

    def test_checksum_mismatch_is_rejected(self, env):
        detectors, origin, m = env
        (origin / "a.pth").write_bytes(b"tampered")
        result = pull(m, detectors, ["det_a"], source="origin", log=lambda _m: None)
        assert "det_a/a.pth" in result.failed
        assert "checksum" in result.failed["det_a/a.pth"] or "size" in result.failed["det_a/a.pth"]
        assert not (detectors / "det_a" / "weights" / "a.pth").exists()

    def test_hub_source_without_hub_fails_clearly(self, env):
        detectors, origin, m = env
        result = pull(m, detectors, ["det_a"], source="hub", log=lambda _m: None)
        assert "hub is not configured" in result.failed["det_a/a.pth"]

    def test_corrupt_local_file_is_detected_and_replaced(self, env):
        detectors, origin, m = env
        pull(m, detectors, ["det_b"], source="origin", log=lambda _m: None)
        target = detectors / "det_b" / "weights" / "b.pth"
        target.write_bytes(b"corrupt")
        assert [r.verified for r in verify(m, detectors, ["det_b"])] == [False]
        result = pull(m, detectors, ["det_b"], source="origin", log=lambda _m: None)
        assert result.ok and len(result.fetched) == 1
        assert [r.verified for r in verify(m, detectors, ["det_b"])] == [True]

    def test_weights_root_symlinks_package_dir(self, env, monkeypatch, tmp_path):
        detectors, origin, m = env
        root = tmp_path / "wroot"
        monkeypatch.setenv("VERITY_WEIGHTS_ROOT", str(root))
        pull(m, detectors, ["det_a"], source="origin", log=lambda _m: None)
        pkg = detectors / "det_a" / "weights"
        assert pkg.is_symlink() and pkg.resolve() == (root / "det_a").resolve()
        assert (root / "det_a" / "a.pth").is_file()

    def test_status_reports_missing(self, env):
        detectors, origin, m = env
        rows = status(m, detectors)
        assert all(not r.present for r in rows)


class TestBundle:
    def test_bundle_layout(self, env, tmp_path):
        detectors, origin, m = env
        pull(m, detectors, None, source="origin", log=lambda _m: None)
        out = tmp_path / "bundle"
        info = build_bundle(m, detectors, out, "someone/verity-weights")
        assert info["missing"] == []
        assert sorted(info["copied"]) == ["src/a.pth", "src/c.pth"]
        assert (out / "src" / "a.pth").is_file()
        assert not (out / "b.pth").exists()                      # not redistributable
        card = (out / "README.md").read_text()
        assert card.startswith("---\nlicense: cc-by-nc-4.0")
        assert "someone/verity-weights" in card and "Src et al. 2024" in card
        notice = (out / "NOTICE.md").read_text()
        assert "## src/a.pth" in notice and "(c) Src" in notice
        sums = [line.split("  ")[1] for line in (out / "sha256sums.txt").read_text().splitlines()]
        assert sums == ["src/a.pth", "src/c.pth"]
        assert (out / "manifest.json").exists() and (out / ".gitattributes").exists()

    def test_bundle_reports_missing(self, env, tmp_path):
        detectors, origin, m = env
        info = build_bundle(m, detectors, tmp_path / "b", "x/y")
        assert sorted(info["missing"]) == ["src/a.pth", "src/c.pth"]
