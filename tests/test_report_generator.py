import json
import tempfile
from pathlib import Path
from PIL import Image
from verity.analyze.models import DetectorSelection, SelectionDecision
from verity.detect.base_detector import DetectorDefinition
from verity.models.findings import Classification, Finding, Reasoning
from verity.models.media import MediaType
from verity.report.report_generator import ReportGenerator


def _png(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (120, 60, 200)).save(path)


def _make_workspace(tmp: Path, media_id: str) -> Path:
    ws = tmp / "ws"
    media_dir = ws / "media"
    media_dir.mkdir(parents=True)
    info = {
        "original_path": f"/input/suspect.mp4",
        "media_type": "video",
        "checksum": "ab" * 32,
        "size_bytes": 1234,
        "mime_type": "video/mp4",
        "metadata": {"original_name": "suspect.mp4"},
    }
    (media_dir / f".{media_id}.json").write_text(json.dumps(info))
    (media_dir / f"{media_id}.mp4").write_bytes(b"\x00" * 64)

    out = ws / "output" / f"{media_id}_recce"
    out.mkdir(parents=True)
    _png(out / "timeline.png")
    for i in range(8):
        _png(out / "faces" / f"f0000{i}_face0.png")
        _png(out / "attention_maps" / f"f0000{i}_face0_attention.png")

    finding = Finding(
        detector_id="recce",
        detector_version="1.0.0",
        classification=Classification.FAKE,
        confidence=0.95,
        reasoning=Reasoning(
            summary="Media shows signs of manipulation.",
            confidence_factors=["factor one", "factor two"],
        ),
        raw_output={
            "prediction": 0.98,
            "stats": {"count": 3, "min": 0.9, "max": 0.99},
            "per_face": [
                {"face_id": "f00001_face0", "frame_idx": 1, "fake_prob": 0.95},
                {"face_id": "f00002_face0", "frame_idx": 2, "fake_prob": 0.99},
                {"face_id": "f00000_face0", "frame_idx": 0, "fake_prob": 0.90},
            ],
        },
    )
    (out / "finding.json").write_text(finding.model_dump_json(indent=2))
    return ws


def _generator(tmp: str, definitions=None) -> ReportGenerator:
    ws = _make_workspace(Path(tmp), "d425d169a201")
    return ReportGenerator(ws, definitions=definitions)


class TestReportGenerator:
    def test_trimmed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp)
            report = generator.generate("d425d169a201")
            html = generator._to_html(report)

            # neutral findings summary with per-detector metrics
            assert "Findings Summary" in html
            assert "Method" in html
            assert "Faces Analyzed" in html
            assert "Mean Fake Probability" in html
            assert "Confidence" in html
            assert "0.9467" in html  # mean of per_face probabilities

            # per-finding section: reasoning, factors, key figures
            assert "Media shows signs of manipulation." in html
            assert "factor one" in html
            assert "Key Figures" in html
            assert "Score Range" in html

            # explainability artifacts with dropdowns
            assert "Prediction Timeline" in html
            assert "Top suspicious faces" in html
            assert "<details>" in html
            assert "All Faces" in html
            assert "All Attention Maps" in html
            assert "Per-Face Results" in html

            # no verdicts or judgements rendered
            assert "verdict-box" not in html
            assert "FAKE" not in html
            assert "REAL" not in html
            assert "UNCERTAIN" not in html

            # trimmed: no detector metadata / timing / benchmark noise
            for noise in ("About this detector", "Verity Version", "Verity v",
                          "Inference Timing", "Detector Metadata",
                          "Evidence Inventory", "Benchmark Performance",
                          "timing_seconds", "Debug Logs", "execution_context"):
                assert noise not in html

    def test_human_method_names(self):
        definitions = {
            "recce": DetectorDefinition(detector_id="recce", name="RECCE",
                                        version="1.0.0"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp, definitions=definitions)
            html = generator._to_html(generator.generate("d425d169a201"))
            assert "<h2>RECCE" in html
            assert html.count("RECCE") >= 2  # summary row + section heading
            assert "<code>recce</code> v1.0.0" in html  # muted id/version

    def test_name_falls_back_to_detector_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp)
            html = generator._to_html(generator.generate("d425d169a201"))
            assert ">recce<" in html

    def test_selection_present_summary_and_filtering(self):
        selection = DetectorSelection(
            media_id="d425d169a201",
            media_type=MediaType.VIDEO,
            decisions=[
                SelectionDecision(detector_id="recce", included=True,
                                  reasons=[]),
                SelectionDecision(detector_id="core", included=False,
                                  reasons=["requires faces; none detected"]),
                SelectionDecision(detector_id="xception", included=True,
                                  reasons=[]),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp)
            html = generator._to_html(
                generator.generate("d425d169a201", selection=selection)
            )
            # summary row for the included detector with a finding
            assert "<td>recce" in html
            assert "0.9467" in html
            # excluded detector listed with its reason
            assert "Excluded Detectors" in html
            assert "requires faces; none detected" in html
            # included detector without a finding -> muted placeholder
            assert "no finding" in html
            # filtering: only included findings get sections
            assert "<h2>recce" in html
            assert html.count("<h2>") == 3  # Coverage + Findings Summary + recce only
            assert "<h2>Coverage</h2>" in html

    def test_report_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp)
            a = generator._to_html(generator.generate("d425d169a201"))
            b = generator._to_html(generator.generate("d425d169a201"))
            assert a == b

    def test_per_face_sorted_by_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            generator = _generator(tmp)
            html = generator._to_html(generator.generate("d425d169a201"))
            first = html.index("f00002_face0")
            second = html.index("f00001_face0")
            third = html.index("f00000_face0")
            assert first < second < third
