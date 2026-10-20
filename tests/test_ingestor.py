import pytest
import tempfile
from pathlib import Path
from verity.ingest.media_ingestor import MediaIngestor


class TestMediaIngestor:
    def test_ingest_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img_path = ws / "test.png"
            img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

            ingestor = MediaIngestor(ws)
            copy = ingestor.ingest(img_path)

            assert copy.media_id
            assert copy.original.media_type.value == "image"
            assert copy.original.checksum
            assert copy.copy_path.exists()
            assert copy.copy_path.stat().st_mode & 0o444
            assert copy.verified is True

    def test_ingest_file_not_found(self):
        ingestor = MediaIngestor(Path("/tmp"))
        with pytest.raises(FileNotFoundError):
            ingestor.ingest(Path("/nonexistent/file.mp4"))

    def test_ingest_persists_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            img_path = ws / "photo.jpg"
            img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

            ingestor = MediaIngestor(ws)
            copy = ingestor.ingest(img_path)

            meta_path = ws / "media" / f".{copy.media_id}.json"
            assert meta_path.exists()
