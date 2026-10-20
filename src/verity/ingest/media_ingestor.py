from __future__ import annotations
import json
import hashlib
import shutil
import filetype
from pathlib import Path
from datetime import datetime, timezone
from verity.models.media import MediaType, MediaInfo, MediaCopy

MEDIA_ID_LENGTH = 12


class MediaIngestor:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.media_dir = workspace / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _detect_type(self, path: Path) -> MediaType:
        kind = filetype.guess(str(path))
        if kind is None:
            return MediaType.UNKNOWN
        mime = kind.mime
        if mime.startswith("image/"):
            return MediaType.IMAGE
        if mime.startswith("video/"):
            return MediaType.VIDEO
        if mime.startswith("audio/"):
            return MediaType.AUDIO
        return MediaType.UNKNOWN

    def ingest(self, media_path: Path) -> MediaCopy:
        media_path = media_path.resolve()
        if not media_path.exists():
            raise FileNotFoundError(f"Media not found: {media_path}")

        checksum = self._compute_checksum(media_path)
        media_id = checksum[:MEDIA_ID_LENGTH]
        mime_type = filetype.guess_mime(str(media_path)) or ""
        media_type = self._detect_type(media_path)

        copy_path = self.media_dir / f"{media_id}{media_path.suffix}"
        if not copy_path.exists():
            shutil.copy2(media_path, copy_path)
            copy_path.chmod(0o444)

        info = MediaInfo(
            original_path=media_path,
            media_type=media_type,
            checksum=checksum,
            size_bytes=media_path.stat().st_size,
            mime_type=mime_type,
            metadata={
                "original_name": media_path.name,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        meta_path = self.media_dir / f".{media_id}.json"
        meta_path.write_text(info.model_dump_json(indent=2))

        return MediaCopy(
            media_id=media_id,
            original=info,
            copy_path=copy_path,
            verified=True,
        )
