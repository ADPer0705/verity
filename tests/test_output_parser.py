import pytest
import tempfile
from pathlib import Path
from verity.detect.output_parser import parse


class TestOutputParser:
    def test_parse_fake(self):
        raw = {
            "prediction": 0.92,
            "confidence": 0.88,
            "reasoning": "Strong GAN artifacts detected.",
            "confidence_factors": ["Upsampling grid", "Color anomaly"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            result = parse(raw, Path(tmp), "1.0.0", "xception")
            assert result.classification.value == "FAKE"
            assert result.confidence == 0.88
            assert len(result.reasoning.confidence_factors) == 2
            assert result.detector_id == "xception"

    def test_parse_real(self):
        raw = {"prediction": 0.12, "confidence": 0.95, "reasoning": "No artifacts found."}
        with tempfile.TemporaryDirectory() as tmp:
            result = parse(raw, Path(tmp), "1.0.0", "meso4")
            assert result.classification.value == "REAL"

    def test_parse_uncertain(self):
        raw = {"prediction": 0.5, "confidence": 0.3, "reasoning": "Inconclusive."}
        with tempfile.TemporaryDirectory() as tmp:
            result = parse(raw, Path(tmp), "1.0.0", "test")
            assert result.classification.value == "UNCERTAIN"
