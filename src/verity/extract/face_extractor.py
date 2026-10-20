from __future__ import annotations
import cv2
from pathlib import Path
from typing import Optional
from verity.models.artifacts import ArtifactType, Artifact, ArtifactCollection
from verity.extract.base_extractor import BaseExtractor


class FaceExtractor(BaseExtractor):
    def __init__(self, workspace: Path, cascade_path: Optional[Path] = None):
        super().__init__(workspace)
        self._detector = None
        self._cascade_path = cascade_path
        self._init_detector()

    def _init_detector(self):
        try:
            cascade = self._cascade_path or Path(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            if hasattr(cv2, "CascadeClassifier"):
                self._detector = cv2.CascadeClassifier(str(cascade))
            else:
                self._detector = None
        except Exception:
            self._detector = None

    def extract(self, media_id: str) -> ArtifactCollection:
        frames_dir = self.artifact_dir / media_id / "frames"
        if not frames_dir.exists() or self._detector is None:
            return ArtifactCollection(media_id=media_id)

        out_dir = self.artifact_dir / media_id / "faces"
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts = []

        for frame_path in sorted(frames_dir.glob("frame_*.png")):
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            for i, (x, y, w, h) in enumerate(faces):
                x, y, w, h = max(0, x), max(0, y), min(w, img.shape[1] - x), min(h, img.shape[0] - y)
                if w < 20 or h < 20:
                    continue
                face_img = img[y : y + h, x : x + w]
                face_path = out_dir / f"{frame_path.stem}_face_{i}.png"
                cv2.imwrite(str(face_path), face_img)
                artifacts.append(Artifact(
                    artifact_type=ArtifactType.FACE_CROP,
                    path=face_path,
                    source_media_id=media_id,
                    params={"source_frame": frame_path.name, "bbox": [int(x), int(y), int(w), int(h)]},
                ))

        return ArtifactCollection(media_id=media_id, artifacts=artifacts)
