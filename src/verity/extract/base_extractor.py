from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from verity.models.artifacts import ArtifactCollection


class BaseExtractor(ABC):
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.artifact_dir = workspace / "artifacts"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(self, media_id: str) -> ArtifactCollection: ...
