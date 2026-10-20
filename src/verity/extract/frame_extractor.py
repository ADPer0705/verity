from __future__ import annotations
import subprocess
from pathlib import Path
from typing import Optional
from verity.models.artifacts import ArtifactType, Artifact, ArtifactCollection
from verity.extract.base_extractor import BaseExtractor


class FrameExtractor(BaseExtractor):
    def __init__(self, workspace: Path, fps: Optional[float] = None):
        super().__init__(workspace)
        self.fps = fps

    def extract(self, media_id: str) -> ArtifactCollection:
        media_dir = self.workspace / "media"
        video_path = None
        for p in media_dir.iterdir():
            if p.stem == media_id:
                video_path = p
                break

        if video_path is None:
            raise FileNotFoundError(f"No media found for id: {media_id}")

        out_dir = self.artifact_dir / media_id / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []

        mime = self._guess_mime(video_path)
        if mime and mime.startswith("video/"):
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
                   "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                   str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            has_video = "video" in result.stdout

            if has_video:
                fps_val = self.fps or 1
                frame_pattern = str(out_dir / "frame_%05d.png")
                subprocess.run(
                    ["ffmpeg", "-i", str(video_path),
                     "-vf", f"fps={fps_val}",
                     "-frame_pts", "1",
                     frame_pattern, "-y"],
                    capture_output=True, text=True, check=True,
                )

                for frame_path in sorted(out_dir.glob("frame_*.png")):
                    artifacts.append(Artifact(
                        artifact_type=ArtifactType.FRAME,
                        path=frame_path,
                        source_media_id=media_id,
                        params={"fps": fps_val, "frame": frame_path.stem},
                    ))

        return ArtifactCollection(media_id=media_id, artifacts=artifacts)

    @staticmethod
    def _guess_mime(path: Path) -> str:
        import filetype
        kind = filetype.guess(str(path))
        return kind.mime if kind else ""
