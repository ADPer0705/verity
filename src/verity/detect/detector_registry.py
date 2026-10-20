from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Optional
from verity.detect.base_detector import DetectorDefinition


class DetectorRegistry:
    def __init__(self, detectors_dir: Path):
        self.detectors_dir = detectors_dir
        self._definitions: Dict[str, DetectorDefinition] = {}
        self._scan()

    def _scan(self):
        if not self.detectors_dir.exists():
            return
        for detector_dir in sorted(self.detectors_dir.iterdir()):
            if not detector_dir.is_dir():
                continue
            yaml_path = detector_dir / "detector.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                definition = DetectorDefinition(**data)
                self._definitions[definition.detector_id] = definition

    def get(self, detector_id: str) -> Optional[DetectorDefinition]:
        return self._definitions.get(detector_id)

    def list_detectors(self) -> Dict[str, DetectorDefinition]:
        return dict(self._definitions)

    def register(self, definition: DetectorDefinition):
        self._definitions[definition.detector_id] = definition

    def weights_available(self, detector_id: str) -> bool:
        """True when the detector can run without a local checkpoint, or when
        its checkpoint file is present under detectors/<id>/weights/."""
        definition = self.get(detector_id)
        if definition is None:
            return False
        if definition.weights.self_contained or not definition.weights.file:
            return True
        weights_dir = self.detectors_dir / detector_id / "weights"
        return (weights_dir / definition.weights.file).exists()
