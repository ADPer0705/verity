from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectorSpec:
    """Static configuration for one detector, mirroring ``detector.yaml``.

    All fields default to the DeepfakeBench image-detector baseline
    (256x256 RGB, mean/std 0.5); individual detectors override the fields
    that differ.
    """

    detector_id: str
    version: str = "1.0.0"
    name: str = ""
    description: str = ""
    architecture: str = ""
    parameters: int | None = None
    pretrained: str = ""
    resolution: int = 256
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    video_mode: bool = False
    clip_size: int | None = None
    sample_interval: int = 1
    weights_file: str | None = None
    weights_url: str | None = None
    weights_size: int | None = None
    weights_md5: str | None = None
    weights_kind: str = "official"  # official | backbone | external | random
    weight_note: str = ""
    cam: bool = True
    min_face: int = 30
    region_analysis: bool = True
    # What the model consumes:
    #   face_crop  one face crop per sampled frame (default; FF++-style detectors)
    #   face_clip  clip_size consecutive face crops (video_mode: true)
    #   full_frame the whole image / sampled frame (AI-generated-image detectors)
    #   full_clip  clip_size consecutive whole frames (generated-video detectors)
    #   audio      mono 16 kHz waveform windows (voice-clone detectors)
    input_mode: str = "face_crop"
    # full_frame / full_clip preprocessing: resize the shorter side to
    # ``resize_short`` (None = resize to resolution x resolution directly),
    # then center-crop ``resolution`` when ``center_crop`` is true.
    resize_short: int | None = None
    center_crop: bool = False
    max_frames: int = 32          # frame budget for full_frame/full_clip modes
    # audio mode: window length / hop in seconds (None = whole signal at once)
    audio_window_seconds: float | None = 4.0
    audio_hop_seconds: float | None = 2.0
    audio_sample_rate: int = 16000

    def to_dict(self) -> dict:
        d = {
            "detector_id": self.detector_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "pretrained": self.pretrained,
            "resolution": self.resolution,
            "mean": list(self.mean),
            "std": list(self.std),
            "video_mode": self.video_mode,
            "clip_size": self.clip_size,
            "sample_interval": self.sample_interval,
            "weights": {
                "file": self.weights_file,
                "url": self.weights_url,
                "size": self.weights_size,
                "md5": self.weights_md5,
                "kind": self.weights_kind,
                "note": self.weight_note,
            },
            "cam": self.cam,
            "input_mode": self.input_mode,
        }
        return d


def load_spec(yaml_path: Path) -> DetectorSpec:
    """Load a spec from a ``detector.yaml``-shaped JSON dict or YAML file."""
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    weights = data.get("weights", {})
    return DetectorSpec(
        detector_id=data["detector_id"],
        version=data.get("version", "1.0.0"),
        name=data.get("name", data["detector_id"]),
        description=data.get("description", ""),
        architecture=data.get("architecture", ""),
        parameters=data.get("parameters"),
        pretrained=data.get("pretrained", ""),
        resolution=data.get("resolution", 256),
        mean=tuple(data.get("mean", [0.5, 0.5, 0.5])),
        std=tuple(data.get("std", [0.5, 0.5, 0.5])),
        video_mode=data.get("video_mode", False),
        clip_size=data.get("clip_size"),
        sample_interval=data.get("sample_interval", 1),
        weights_file=weights.get("file"),
        weights_url=weights.get("url"),
        weights_size=weights.get("size"),
        weights_md5=weights.get("md5"),
        weights_kind=weights.get("kind", "official"),
        weight_note=weights.get("note", ""),
        cam=data.get("cam", True),
        min_face=data.get("min_face", 30),
        region_analysis=data.get("region_analysis", True),
        input_mode=data.get("input_mode", "face_clip" if data.get("video_mode") else "face_crop"),
        resize_short=data.get("resize_short"),
        center_crop=bool(data.get("center_crop", False)),
        max_frames=int(data.get("max_frames", 32)),
        audio_window_seconds=data.get("audio_window_seconds", 4.0),
        audio_hop_seconds=data.get("audio_hop_seconds", 2.0),
        audio_sample_rate=int(data.get("audio_sample_rate", 16000)),
    )
