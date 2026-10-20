from __future__ import annotations

from PIL import Image


class FaceDetector:
    """MTCNN-backed face detection used by every detector.

    Deliberately tiny wrapper so detector ``inference.py`` scripts do not need
    to touch MTCNN directly; behavior mirrors the RECCE detector (detect all
    faces, drop tiny boxes).
    """

    def __init__(self, device: str = "cpu", min_face: int = 30):
        from facenet_pytorch import MTCNN

        self._mtcnn = MTCNN(keep_all=False, device="cpu")
        self.min_face = min_face

    def detect(self, pil: Image.Image) -> list[tuple[int, int, int, int]]:
        """Return [(x, y, w, h)] boxes in the image's pixel space."""
        boxes, _ = self._mtcnn.detect(pil)
        if boxes is None:
            return []
        out = []
        for b in boxes:
            x1, y1, x2, y2 = [max(0, int(v)) for v in b]
            if x2 - x1 < self.min_face or y2 - y1 < self.min_face:
                continue
            out.append((x1, y1, x2 - x1, y2 - y1))
        return out
