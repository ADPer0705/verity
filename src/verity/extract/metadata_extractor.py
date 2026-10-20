from __future__ import annotations
import subprocess
import json
from pathlib import Path
from verity.models.artifacts import ArtifactType, Artifact, ArtifactCollection
from verity.extract.base_extractor import BaseExtractor


class MetadataExtractor(BaseExtractor):
    def extract(self, media_id: str) -> ArtifactCollection:
        media_dir = self.workspace / "media"
        media_path = None
        for p in media_dir.iterdir():
            if p.stem == media_id:
                media_path = p
                break

        if media_path is None:
            raise FileNotFoundError(f"No media found for id: {media_id}")

        out_dir = self.artifact_dir / media_id / "metadata"
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {"file": {"name": media_path.name, "size_bytes": media_path.stat().st_size}}

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", str(media_path)],
                capture_output=True, text=True, check=True,
            )
            metadata["ffprobe"] = json.loads(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            metadata["ffprobe"] = None

        meta_path = out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        artifact = Artifact(
            artifact_type=ArtifactType.METADATA,
            path=meta_path,
            source_media_id=media_id,
            params={},
        )

        return ArtifactCollection(media_id=media_id, artifacts=[artifact])
