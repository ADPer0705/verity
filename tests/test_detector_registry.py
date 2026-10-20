import tempfile
import yaml
from pathlib import Path
from verity.detect.detector_registry import DetectorRegistry
from verity.detect.base_detector import DetectorDefinition, DetectorCapabilities


def _write_detector(tmp: Path, detector_id: str, required: list[str],
                    weights_file: str = "", weights_kind: str = "official",
                    name: str = "", clip_size: int | None = None,
                    min_face: int = 30, video_mode: bool = False):
    det_dir = tmp / detector_id
    det_dir.mkdir(parents=True)
    config = {
        "detector_id": detector_id,
        "version": "1.0.0",
        "artifacts": {"required": [{"type": t, "params": {}} for t in required]},
        "output": {"format": "json", "schema": {"prediction": "float"}},
    }
    if name:
        config["name"] = name
    if clip_size is not None:
        config["clip_size"] = clip_size
    if min_face != 30:
        config["min_face"] = min_face
    if video_mode:
        config["video_mode"] = True
    if weights_file or weights_kind:
        config["weights"] = {"file": weights_file, "kind": weights_kind}
    (det_dir / "detector.yaml").write_text(yaml.safe_dump(config))
    if weights_file:
        wdir = det_dir / "weights"
        wdir.mkdir()
        (wdir / weights_file).write_bytes(b"x")


class TestCapabilities:
    def test_face_crop_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            definition = DetectorRegistry(Path(tmp)).get("facey")
            caps = definition.capabilities
            assert caps.media_types == ["image", "video"]
            assert caps.faces_required is True
            assert caps.accepts("image") and caps.accepts("video")
            assert not caps.accepts("audio")

    def test_video_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"])
            caps = DetectorRegistry(Path(tmp)).get("videoey").capabilities
            assert caps.media_types == ["video"]
            assert caps.faces_required is True

    def test_audio_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "audioey", ["audio"])
            caps = DetectorRegistry(Path(tmp)).get("audioey").capabilities
            # audio detectors consume standalone audio and video soundtracks
            assert caps.media_types == ["audio", "video"]
            assert caps.audio_required is True
            assert caps.faces_required is False


class TestWeightsAvailability:
    def test_file_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "d", ["face_crop"], weights_file="model.pth")
            registry = DetectorRegistry(Path(tmp))
            assert registry.weights_available("d") is True

    def test_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "d", ["face_crop"], weights_file="model.pth")
            (Path(tmp) / "d" / "weights" / "model.pth").unlink()
            assert DetectorRegistry(Path(tmp)).weights_available("d") is False

    def test_self_contained(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "d", ["face_crop"],
                            weights_file="", weights_kind="external")
            assert DetectorRegistry(Path(tmp)).weights_available("d") is True


class TestRegistry:
    def test_empty_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = DetectorRegistry(Path(tmp))
            assert len(registry.list_detectors()) == 0

    def test_get_unknown_detector(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert DetectorRegistry(Path(tmp)).get("nonexistent") is None

    def test_scan_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("zeta", "alpha", "mike"):
                _write_detector(Path(tmp), name, ["face_crop"])
            registry = DetectorRegistry(Path(tmp))
            assert list(registry.list_detectors()) == ["alpha", "mike", "zeta"]

    def test_name_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"], name="Facey")
            definition = DetectorRegistry(Path(tmp)).get("facey")
            assert definition.name == "Facey"

    def test_name_defaults_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            definition = DetectorRegistry(Path(tmp)).get("facey")
            assert definition.name == ""

    def test_runtime_fields_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "videoey", ["video"], clip_size=16,
                            min_face=40, video_mode=True)
            definition = DetectorRegistry(Path(tmp)).get("videoey")
            assert definition.clip_size == 16
            assert definition.min_face == 40
            assert definition.video_mode is True

    def test_runtime_fields_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_detector(Path(tmp), "facey", ["face_crop"])
            definition = DetectorRegistry(Path(tmp)).get("facey")
            assert definition.clip_size is None
            assert definition.min_face == 30
            assert definition.video_mode is False
            assert definition.cam is True
