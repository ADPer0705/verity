import pytest
import tempfile
import json
from pathlib import Path
from verity.compile.findings_compiler import FindingsCompiler
from verity.models.findings import Classification, Reasoning, Finding


class TestFindingsCompiler:
    def test_compile_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            compiler = FindingsCompiler(ws)
            collection = compiler.compile("nonexistent")
            assert len(collection.findings) == 0
            assert collection.aggregated_score is None

    def test_compile_with_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            output_dir = ws / "output"
            output_dir.mkdir()

            finding = Finding(
                detector_id="xception",
                detector_version="1.0.0",
                classification=Classification.FAKE,
                confidence=0.95,
                reasoning=Reasoning(summary="Test"),
            )

            subdir = output_dir / "abc_xception"
            subdir.mkdir()
            (subdir / "finding.json").write_text(finding.model_dump_json(indent=2))

            compiler = FindingsCompiler(ws)
            collection = compiler.compile("abc")
            assert len(collection.findings) == 1
            assert collection.findings[0].detector_id == "xception"
            assert collection.aggregated_score is not None


class TestTierAggregation:
    def _write(self, output_dir: Path, media_id: str, det: str, tier: str,
               classification: Classification, confidence: float) -> None:
        finding = Finding(
            detector_id=det, detector_version="1.0.0", classification=classification,
            confidence=confidence, reasoning=Reasoning(summary="t"), tier=tier,
        )
        subdir = output_dir / f"{media_id}_{det}"
        subdir.mkdir()
        (subdir / "finding.json").write_text(finding.model_dump_json(indent=2))

    def test_untrained_findings_never_move_the_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            out = ws / "output"
            out.mkdir()
            self._write(out, "abc", "real_one", "trained", Classification.REAL, 0.9)
            self._write(out, "abc", "noise", "random-init", Classification.FAKE, 1.0)
            collection = FindingsCompiler(ws).compile("abc")
            assert len(collection.findings) == 2          # still reported ...
            assert collection.aggregated_score == 0.0     # ... but not counted
            tiers = {f.detector_id: f.tier for f in collection.findings}
            assert tiers == {"real_one": "trained", "noise": "random-init"}

    def test_only_untrained_gives_no_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            out = ws / "output"
            out.mkdir()
            self._write(out, "abc", "noise", "backbone-only", Classification.FAKE, 1.0)
            assert FindingsCompiler(ws).compile("abc").aggregated_score is None

    def test_legacy_finding_without_tier_is_trained(self):
        data = {"detector_id": "x", "detector_version": "1", "classification": "REAL",
                "confidence": 0.5, "reasoning": {"summary": "s"}}
        assert Finding(**data).tier == "trained"

    def test_legacy_finding_gets_tier_from_registry(self):
        from verity.detect.base_detector import DetectorDefinition
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            out = ws / "output"
            out.mkdir()
            legacy = {"detector_id": "oldrandom", "detector_version": "1", "classification": "FAKE",
                      "confidence": 0.9, "reasoning": {"summary": "s"}}
            (out / "abc_oldrandom").mkdir()
            (out / "abc_oldrandom" / "finding.json").write_text(json.dumps(legacy))
            defs = {"oldrandom": DetectorDefinition(detector_id="oldrandom", version="1",
                                                    weights={"kind": "random"})}
            collection = FindingsCompiler(ws, defs).compile("abc")
            assert collection.findings[0].tier == "random-init"
            assert collection.aggregated_score is None
            assert FindingsCompiler(ws).compile("abc").findings[0].tier == "trained"
