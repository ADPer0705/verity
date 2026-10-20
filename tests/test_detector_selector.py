import tempfile
import yaml
from pathlib import Path
from verity.analyze.detector_selector import DetectorSelector
from verity.analyze.models import MediaAnalysis, MediaCharacteristics
from verity.detect.detector_registry import DetectorRegistry
from verity.models.media import MediaType


def _write_detector(tmp: Path, detector_id: str, required: list[str],
                    clip_size: int | None = None, min_face: int = 30,
                    weights: dict | None = None):
    det_dir = tmp / detector_id
    det_dir.mkdir(parents=True)
    config = {
        "detector_id": detector_id,
        "version": "1.0.0",
        "artifacts": {"required": [{"type": t, "params": {}} for t in required]},
        "output": {"format": "json", "schema": {"prediction": "float"}},
    }
    if clip_size is not None:
        config["clip_size"] = clip_size
    if min_face != 30:
        config["min_face"] = min_face
    if weights is not None:
        config["weights"] = weights
    (det_dir / "detector.yaml").write_text(yaml.safe_dump(config))
    if weights and weights.get("file"):
        wdir = det_dir / "weights"
        wdir.mkdir()
        (wdir / weights["file"]).write_bytes(b"x")


def _analysis(**kwargs) -> MediaAnalysis:
    defaults = dict(
        media_id="abc123", media_type="video", face_present=True, face_count=1,
        sampled_frames=30, max_face_size=120, frame_count=460,
        duration_seconds=18.4, fps=25.0,
    )
    defaults.update(kwargs)
    return MediaAnalysis(media=MediaCharacteristics(
        media_id=defaults.pop("media_id"),
        media_type=MediaType(defaults.pop("media_type")),
        **defaults,
    ))


def _selector(tmp: Path, offline: bool = False,
              include_untrained: bool = False) -> DetectorSelector:
    return DetectorSelector(DetectorRegistry(Path(tmp)), offline=offline,
                            include_untrained=include_untrained)


class TestSelectionRules:
    def test_media_type_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            _write_detector(Path(tmp), "videoey", ["video"])
            selection = _selector(tmp).select(_analysis(media_type="audio"))
            assert selection.decision("facey").included is False
            assert "media is audio" in selection.decision("facey").reasons[0]
            assert selection.decision("videoey").included is False
            assert selection.decision("videoey").reasons[0] == \
                "requires video, media is audio"

    def test_faces_required_none_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            decision = _selector(tmp).select(
                _analysis(face_present=False, face_count=0, max_face_size=0)
            ).decision("facey")
            assert decision.included is False
            assert "requires faces" in decision.reasons[0]

    def test_faces_too_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"], min_face=100)
            decision = _selector(tmp).select(
                _analysis(max_face_size=60)
            ).decision("facey")
            assert decision.included is False
            assert "faces too small" in decision.reasons[0]
            assert "60px" in decision.reasons[0] and "100px" in decision.reasons[0]

    def test_video_too_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"], clip_size=16)
            decision = _selector(tmp).select(
                _analysis(frame_count=8, duration_seconds=0.3, fps=25)
            ).decision("videoey")
            assert decision.included is False
            assert "video too short" in decision.reasons[0]
            assert "8 frames" in decision.reasons[0] and "clip size 16" in decision.reasons[0]

    def test_video_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"], clip_size=16)
            decision = _selector(tmp).select(
                _analysis(frame_count=460)
            ).decision("videoey")
            assert decision.included is True
            assert any(">= clip size 16" in r for r in decision.reasons)

    def test_short_video_without_clip_size(self):
        # recce-style: video artifact, no clip_size declared (min 2 frames)
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"])
            decision = _selector(tmp).select(
                _analysis(frame_count=1, duration_seconds=0.04, fps=25)
            ).decision("videoey")
            assert decision.included is False
            assert "video too short" in decision.reasons[0]

    def test_frame_count_falls_back_to_duration_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"], clip_size=32)
            decision = _selector(tmp).select(
                _analysis(frame_count=None, duration_seconds=2.0, fps=25)
            ).decision("videoey")
            # 50 estimated frames >= clip size 32
            assert decision.included is True


class TestWeightRules:
    def test_checkpoint_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = {"file": "model.pth", "kind": "official",
                       "url": "https://github.com/SCLBD/DeepfakeBench"}
            _write_detector(Path(tmp), "facey", ["face_crop"], weights=weights)
            (Path(tmp) / "facey" / "weights" / "model.pth").unlink()
            decision = _selector(tmp).select(_analysis()).decision("facey")
            assert decision.included is False
            assert "checkpoint missing: weights/model.pth" in decision.reasons[0]
            assert "github.com" in decision.reasons[0]

    def test_checkpoint_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = {"file": "model.pth", "kind": "official"}
            _write_detector(Path(tmp), "facey", ["face_crop"], weights=weights)
            decision = _selector(tmp).select(_analysis()).decision("facey")
            assert decision.included is True
            assert any("weights available" in r for r in decision.reasons)

    def test_random_weights_only_on_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"],
                            weights={"kind": "random"})
            decision = _selector(tmp, include_untrained=True).select(_analysis()).decision("facey")
            assert decision.included is True
            assert any("random initialization" in r for r in decision.reasons)
            assert any("not evidence-grade" in r for r in decision.reasons)

    def test_runtime_download_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            decision = _selector(tmp).select(_analysis()).decision("facey")
            assert decision.included is True
            assert any("downloaded at first run" in r for r in decision.reasons)

    def test_offline_mode_excludes_runtime_downloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            _write_detector(Path(tmp), "randomy", ["face_crop"],
                            weights={"kind": "random"})
            weights = {"file": "model.pth", "kind": "official"}
            _write_detector(Path(tmp), "localy", ["face_crop"], weights=weights)
            selection = _selector(tmp, offline=True, include_untrained=True).select(_analysis())
            assert selection.decision("facey").included is False
            assert "offline mode" in selection.decision("facey").reasons[0]
            assert selection.decision("randomy").included is True
            assert selection.decision("localy").included is True


class TestTrainedHeadGate:
    """Untrained heads are research artifacts: never auto-selected, always
    labelled when run on request."""

    def _write_all(self, tmp: Path):
        _write_detector(tmp, "official", ["face_crop"],
                        weights={"file": "m.pth", "kind": "official"})
        _write_detector(tmp, "randomy", ["face_crop"], weights={"kind": "random"})
        _write_detector(tmp, "backboney", ["face_crop"],
                        weights={"file": "bb.pth", "kind": "backbone"})
        _write_detector(tmp, "external_untrained", ["face_crop"],
                        weights={"kind": "external"})
        _write_detector(tmp, "external_trained", ["face_crop"],
                        weights={"kind": "external", "trained": True})

    def test_default_excludes_untrained(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(Path(tmp))
            selection = _selector(tmp).select(_analysis())
            assert selection.selected == ["external_trained", "official"]
            for det in ("randomy", "backboney", "external_untrained"):
                d = selection.decision(det)
                assert d.included is False
                assert "excluded by default" in d.reasons[0]
            assert "random-init" in selection.decision("randomy").reasons[0]
            assert "backbone-only" in selection.decision("backboney").reasons[0]

    def test_include_untrained_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(Path(tmp))
            selection = _selector(tmp, include_untrained=True).select(_analysis())
            assert set(selection.selected) == {
                "official", "randomy", "backboney", "external_untrained", "external_trained"}

    def test_env_flag(self, monkeypatch):
        monkeypatch.setenv("VERITY_INCLUDE_UNTRAINED", "1")
        with tempfile.TemporaryDirectory() as tmp:
            self._write_all(Path(tmp))
            assert "randomy" in _selector(tmp).select(_analysis()).selected

    def test_tier_derivation(self):
        from verity.detect.base_detector import Weights
        assert Weights(kind="official").tier == "trained"
        assert Weights(kind="backbone").tier == "backbone-only"
        assert Weights(kind="random").tier == "random-init"
        assert Weights(kind="external").tier == "untrained"
        assert Weights(kind="external", trained=True).tier == "trained"
        assert Weights(kind="official", trained=False).tier == "untrained"


class TestSelectorDeterminism:
    def test_same_analysis_same_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"], clip_size=4)
            _write_detector(Path(tmp), "videoey", ["video"], clip_size=16)
            selector = _selector(tmp)
            a = selector.select(_analysis())
            b = selector.select(_analysis())
            assert a.model_dump_json() == b.model_dump_json()

    def test_sorted_deterministic_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("zeta", "alpha", "mike"):
                _write_detector(Path(tmp), name, ["face_crop"])
            selection = _selector(tmp).select(_analysis())
            ids = [d.detector_id for d in selection.decisions]
            assert ids == sorted(ids)
