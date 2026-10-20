import json
import os
import subprocess
import tempfile
from pathlib import Path
import pytest
from PIL import Image
from verity.ingest.media_ingestor import MediaIngestor, MEDIA_ID_LENGTH
from verity.analyze.media_analyzer import MediaAnalyzer
from verity.models.media import MediaType


class TestDeterministicMediaId:
    def test_same_file_same_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img = ws / "a.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            ingestor = MediaIngestor(ws)
            first = ingestor.ingest(img)
            second = ingestor.ingest(img)
            assert first.media_id == second.media_id

    def test_media_id_is_checksum_prefix(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img = ws / "a.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            ingestor = MediaIngestor(ws)
            copy = ingestor.ingest(img)
            checksum = hashlib.sha256(img.read_bytes()).hexdigest()
            assert copy.media_id == checksum[:MEDIA_ID_LENGTH]

    def test_different_files_different_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            a = ws / "a.png"
            b = ws / "b.png"
            a.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            b.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x01" * 64)
            ingestor = MediaIngestor(ws)
            assert ingestor.ingest(a).media_id != ingestor.ingest(b).media_id


class TestMediaAnalyzer:
    def test_analysis_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img = ws / "a.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
            ingestor = MediaIngestor(ws)
            copy = ingestor.ingest(img)
            analyzer = MediaAnalyzer(ws / "media")
            a = analyzer.analyze(copy.media_id, MediaType.IMAGE)
            b = analyzer.analyze(copy.media_id, MediaType.IMAGE)
            assert a.model_dump_json() == b.model_dump_json()

    def test_save_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img = ws / "a.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 512)
            copy = MediaIngestor(ws).ingest(img)
            analyzer = MediaAnalyzer(ws / "media")
            analysis = analyzer.analyze(copy.media_id, MediaType.IMAGE)
            out = analyzer.save(analysis, ws / "out")
            assert out.exists()
            loaded = json.loads(out.read_text())
            assert loaded["media"]["media_id"] == copy.media_id


class TestDetectorEngineDeterminism:
    """End-to-end determinism of the shared detector engine (common/engine.py).

    Runs the real `core` detector in its own provisioned environment twice on
    the same input and requires byte-identical outputs (wall-clock timing
    excluded). Skipped when the environment is not provisioned.
    """

    @staticmethod
    def _env_python() -> Path | None:
        repo = Path(__file__).resolve().parents[1]
        for profile in ("cpu", "cuda"):
            py = repo / ".detectorenvs" / profile / "core" / ".venv" / "bin" / "python"
            if py.exists():
                return py
        return None

    def _run(self, python: Path, inference: Path, input_dir: Path,
             output_dir: Path) -> None:
        env = dict(os.environ,
                   INPUT_DIR=str(input_dir), OUTPUT_DIR=str(output_dir))
        subprocess.run(
            [str(python), str(inference)],
            env=env, check=True, capture_output=True, timeout=600,
        )

    @staticmethod
    def _comparable(path: Path) -> dict:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data.pop("timing_seconds", None)
        return data

    def test_core_run_is_deterministic(self):
        python = self._env_python()
        if python is None:
            pytest.skip("core detector environment not provisioned")
        repo = Path(__file__).resolve().parents[1]
        inference = repo / "detectors" / "core" / "inference.py"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "in"
            (input_dir / "media").mkdir(parents=True)
            img = Image.new("RGB", (64, 64), (90, 90, 90))
            img.save(input_dir / "media" / "probe.png")

            outputs = []
            for i in range(2):
                out_dir = root / f"out{i}"
                self._run(python, inference, input_dir, out_dir)
                outputs.append(out_dir)

            for name in ("result.json", "per_face_results.json"):
                a = self._comparable(outputs[0] / name)
                b = self._comparable(outputs[1] / name)
                assert a == b, f"{name} differs between identical runs"

            artifacts_a = sorted(
                p.relative_to(outputs[0]).as_posix()
                for p in outputs[0].rglob("*") if p.is_file()
            )
            artifacts_b = sorted(
                p.relative_to(outputs[1]).as_posix()
                for p in outputs[1].rglob("*") if p.is_file()
            )
            assert artifacts_a == artifacts_b
            for rel in artifacts_a:
                if not rel.endswith((".json")):
                    assert (outputs[0] / rel).read_bytes() == \
                        (outputs[1] / rel).read_bytes(), \
                        f"artifact {rel} differs between identical runs"
