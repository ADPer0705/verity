"""Safe file serving.

The UI needs to display artifacts that live on disk (frames, face crops,
attention maps, reports, logs). Rather than expose a directory-walking
endpoint, every served path is resolved and then checked to be inside one of
the configured roots. Anything else is a 403 — an evidence tool must not be
turned into an arbitrary-file reader by a crafted query string.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse

from verity_api.config import Settings

# Extensions we are willing to hand back. Everything the pipeline emits is on
# this list; anything else is treated as out of scope rather than guessed at.
ALLOWED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi",
    ".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg",
    ".json", ".yaml", ".yml", ".txt", ".log", ".csv",
    ".html", ".pdf",
}


def resolve_within_roots(settings: Settings, raw_path: str) -> Path:
    """Resolve ``raw_path`` and assert it sits inside an allowed root."""
    try:
        candidate = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise HTTPException(status_code=404, detail="File not found") from None

    for root in settings.readable_roots:
        try:
            resolved_root = root.resolve()
        except OSError:
            continue
        if candidate == resolved_root or resolved_root in candidate.parents:
            break
    else:
        raise HTTPException(
            status_code=403,
            detail="Path is outside the workspace, output and detector directories",
        )

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if candidate.suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: {candidate.suffix or 'none'}"
        )
    return candidate


def file_response(
    path: Path,
    download_name: Optional[str] = None,
    inline: bool = True,
) -> FileResponse:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {}
    if download_name and not inline:
        headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
    return FileResponse(path, media_type=media_type, headers=headers)
