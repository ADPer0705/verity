"""Model-free forensic analyses: every analyzer writes its files, the
combined report is deterministic, and failures stay soft."""
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from verity.forensics import ForensicsRunner, load_report
from verity.forensics.compression_analyzer import (CompressionAnalyzer, estimate_quality, ijg_table,
                                                   periodicity_score, block_dct)
from verity.forensics.models import ForensicsReport, Observation
from verity.forensics.runner import ANALYZERS
from verity.ingest.media_ingestor import MediaIngestor
from verity.models.media import MediaType

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _synthetic_image(w: int = 512, h: int = 384, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 128 + 60 * np.sin(x / 40.0) * np.cos(y / 55.0) + 30 * np.sin((x + y) / 13.0)
    base = base + rng.normal(0, 6, size=base.shape)
    base = np.clip(base, 0, 255).astype(np.uint8)
    rgb = np.stack([base, np.roll(base, 7, axis=1), np.roll(base, 3, axis=0)], axis=2)
    return Image.fromarray(rgb)


def _save_double_jpeg(path: Path, q1: int = 75, q2: int = 95) -> None:
    """A JPEG saved twice with different qualities: the classic double-compression trace."""
    buf = io.BytesIO()
    _synthetic_image().save(buf, format="JPEG", quality=q1)
    buf.seek(0)
    Image.open(buf).save(path, format="JPEG", quality=q2)


def _make_video(path: Path) -> bool:
    cmd = ["ffmpeg", "-v", "quiet", "-y",
           "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
           "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0 and path.exists()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _comparable(workspace: Path, media_id: str) -> dict:
    data = json.loads((workspace / "artifacts" / media_id / "forensics" / "forensics.json").read_text())
    return data


class TestModels:
    def test_flat_observations_sorted_by_severity(self):
        from verity.forensics.models import ForensicResult
        results = [
            ForensicResult(analyzer="b", label="B", observations=[
                Observation(analyzer="b", severity="info", statement="z"),
                Observation(analyzer="b", severity="warning", statement="a"),
            ]),
            ForensicResult(analyzer="a", label="A", observations=[
                Observation(analyzer="a", severity="notice", statement="m"),
            ]),
        ]
        report = ForensicsReport.from_results("id", "image", results)
        assert [r.analyzer for r in report.results] == ["a", "b"]
        assert [o.severity for o in report.observations] == ["warning", "notice", "info"]
        assert report.counts == {"warning": 1, "notice": 1, "info": 1}


class TestCompressionPrimitives:
    def test_quality_estimate_roundtrips_standard_tables(self):
        for q in (50, 75, 90, 95):
            est, dist = estimate_quality(ijg_table(q))
            assert est == q and dist == 0.0

    def test_double_jpeg_more_periodic_than_single(self):
        with tempfile.TemporaryDirectory() as tmp:
            single = Path(tmp) / "single.jpg"
            double = Path(tmp) / "double.jpg"
            _synthetic_image().save(single, format="JPEG", quality=95)
            _save_double_jpeg(double, 75, 95)

            def score(path):
                img = Image.open(path)
                table = np.array(img.quantization[0], dtype=np.float64).reshape(8, 8)
                luma = np.asarray(img.convert("L"))
                quantized = np.round(block_dct(luma) / table[None, None])
                s, _, _ = periodicity_score(quantized[:, :, 0, 1].ravel())
                return s

            assert score(double) > score(single)


class TestImageForensics:
    def test_image_analyzers_write_files_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            src = Path(tmp) / "photo.jpg"
            _save_double_jpeg(src)
            copy = MediaIngestor(ws).ingest(src)

            runner = ForensicsRunner(ws)
            first = runner.run(copy.media_id, MediaType.IMAGE)
            first_json = _comparable(ws, copy.media_id)
            first_bytes = {
                p.relative_to(ws).as_posix(): p.read_bytes()
                for p in (ws / "artifacts" / copy.media_id / "forensics").rglob("*.png")
            }
            second = runner.run(copy.media_id, MediaType.IMAGE)
            second_json = _comparable(ws, copy.media_id)

            assert first_json == second_json
            assert first.model_dump() == second.model_dump()
            for rel, data in first_bytes.items():
                assert (ws / rel).read_bytes() == data, f"{rel} differs between runs"

            by_id = {r.analyzer: r for r in first.results}
            assert set(by_id) == {"metadata", "compression", "noise"}
            assert all(r.ok for r in by_id.values()), [r.observations for r in by_id.values() if not r.ok]
            fdir = ws / "artifacts" / copy.media_id / "forensics"
            for rel in ("metadata/metadata.json", "metadata/observations.json",
                        "compression/ela.png", "compression/ela_overlay.png",
                        "compression/blockiness.png", "compression/compression.json",
                        "noise/noise_residual.png", "noise/noise_variance.png",
                        "noise/spectrum.png", "noise/noise.json", "forensics.json"):
                assert (fdir / rel).exists(), rel

            comp = json.loads((fdir / "compression" / "compression.json").read_text())
            assert comp["jpeg"]["estimated_quality"] == 95
            assert comp["jpeg"]["periodicity_score"] >= comp["jpeg"]["periodicity_threshold"]
            statements = " ".join(o.statement for o in by_id["compression"].observations)
            assert "saved at least twice" in statements

            # No EXIF on a synthetic JPEG: the metadata analyzer must say so.
            statements = " ".join(o.statement for o in by_id["metadata"].observations)
            assert "No EXIF" in statements

            # Never a verdict.
            for obs in first.observations:
                assert obs.severity in ("info", "notice", "warning")
                assert "fake" not in obs.statement.lower().split()
                assert "real" not in obs.statement.lower().split()

            loaded = load_report(ws, copy.media_id)
            assert loaded is not None and loaded.counts == first.counts

    def test_png_skips_jpeg_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            src = Path(tmp) / "picture.png"
            _synthetic_image().save(src)
            copy = MediaIngestor(ws).ingest(src)
            report = ForensicsRunner(ws).run(copy.media_id, MediaType.IMAGE)
            comp = next(r for r in report.results if r.analyzer == "compression")
            assert comp.ok and comp.data["jpeg_quality"] is None
            assert any("Not a JPEG" in o.statement for o in comp.observations)


class TestFailSoft:
    def test_analyzer_exception_becomes_notice(self):
        from verity.forensics.base import ForensicAnalyzer

        class Broken(ForensicAnalyzer):
            analyzer_id = "broken"
            label = "Broken"
            media_types = ("image",)

            def analyze(self, media_id, media_type):
                raise RuntimeError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            src = Path(tmp) / "picture.png"
            _synthetic_image(64, 64).save(src)
            copy = MediaIngestor(ws).ingest(src)
            report = ForensicsRunner(ws, analyzers=[Broken]).run(copy.media_id, MediaType.IMAGE)
            assert len(report.results) == 1
            result = report.results[0]
            assert not result.ok
            assert result.observations[0].severity == "notice"
            assert "analyzer broken failed: RuntimeError: boom" in result.observations[0].statement
            assert (ws / "artifacts" / copy.media_id / "forensics" / "forensics.json").exists()

    def test_missing_media_is_soft(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            (ws / "media").mkdir(parents=True)
            report = ForensicsRunner(ws).run("nope", MediaType.IMAGE)
            assert report.results and all(not r.ok for r in report.results)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")
class TestVideoForensics:
    def test_video_analyzers_write_files_and_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            src = Path(tmp) / "clip.mp4"
            if not _make_video(src):
                pytest.skip("ffmpeg could not synthesise the test clip (lavfi/libx264/aac missing)")
            copy = MediaIngestor(ws).ingest(src)
            assert copy.original.media_type == MediaType.VIDEO

            runner = ForensicsRunner(ws)
            first = runner.run(copy.media_id, MediaType.VIDEO)
            first_json = _comparable(ws, copy.media_id)
            second = runner.run(copy.media_id, MediaType.VIDEO)
            assert first_json == _comparable(ws, copy.media_id)
            assert first.model_dump() == second.model_dump()

            by_id = {r.analyzer: r for r in first.results}
            assert set(by_id) == {"metadata", "compression", "noise", "video_temporal", "audio"}
            assert all(r.ok for r in by_id.values()), [r.observations for r in by_id.values() if not r.ok]
            fdir = ws / "artifacts" / copy.media_id / "forensics"
            for rel in ("metadata/metadata.json", "compression/ela.png", "compression/blockiness.png",
                        "noise/spectrum.png", "noise/noise_variance.png",
                        "video_temporal/timeline.png", "video_temporal/temporal.json",
                        "audio/spectrogram.png", "audio/audio.json"):
                assert (fdir / rel).exists(), rel

            temporal = json.loads((fdir / "video_temporal" / "temporal.json").read_text())
            assert temporal["frames_processed"] >= 15
            assert temporal["faces"]["available"] is False  # no face extractor ran

            audio = json.loads((fdir / "audio" / "audio.json").read_text())
            assert audio["has_audio"] is True
            assert 1.5 <= audio["duration"] <= 2.5
            # A pure 440 Hz sine has nothing above 8 kHz.
            if audio["high_band_ratio_db"] is not None:
                assert audio["high_band_ratio_db"] < audio["band_ratio_threshold_db"]

            meta = json.loads((fdir / "metadata" / "metadata.json").read_text())
            assert meta["encoder_tags"].get("container", "").startswith("Lavf")
            statements = " ".join(o.statement for o in by_id["metadata"].observations)
            assert "transcoder or editor" in statements

    def test_runner_skips_video_only_analyzers_for_images(self):
        assert any(a.analyzer_id == "video_temporal" and "image" not in a.media_types for a in ANALYZERS)
