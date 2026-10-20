from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
from verity.models.artifacts import ArtifactType, Artifact, ArtifactCollection
from verity.extract.base_extractor import BaseExtractor


class AudioExtractor(BaseExtractor):
    def __init__(self, workspace: Path, sample_rate: Optional[int] = None):
        super().__init__(workspace)
        self.sample_rate = sample_rate

    def extract(self, media_id: str) -> ArtifactCollection:
        media_dir = self.workspace / "media"
        media_path = None
        for p in media_dir.iterdir():
            if p.stem == media_id:
                media_path = p
                break

        if media_path is None:
            raise FileNotFoundError(f"No media found for id: {media_id}")

        out_dir = self.artifact_dir / media_id / "audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        audio_path = out_dir / "audio.wav"
        cmd = ["ffmpeg", "-i", str(media_path), "-vn",
               "-acodec", "pcm_s16le", "-ar",
               str(self.sample_rate or 16000),
               "-ac", "1", str(audio_path), "-y"]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            # Media carries no audio track, or ffmpeg refused it. Either way
            # this extractor simply contributes nothing.
            return ArtifactCollection(media_id=media_id)
        except (FileNotFoundError, OSError):
            # ffmpeg is not installed. Audio extraction is optional, so this
            # must not abort the whole examination; detectors that need audio
            # are excluded by the selector for want of the artifact.
            return ArtifactCollection(media_id=media_id)

        artifact = Artifact(
            artifact_type=ArtifactType.AUDIO_CLIP,
            path=audio_path,
            source_media_id=media_id,
            params={"sample_rate": self.sample_rate or 16000},
        )

        return ArtifactCollection(media_id=media_id, artifacts=[artifact])
