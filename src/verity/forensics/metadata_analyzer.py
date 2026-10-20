"""Container, stream and EXIF consistency.

Rationale: every encoder, editor and camera leaves fingerprints in the
container (``encoder`` tags, brand codes), the streams (pixel format,
profile, frame-rate mode, GOP structure) and, for photographs, the EXIF
block. A file that claims to come straight from a camera but carries an
ffmpeg encoder tag, a 4:4:4 pixel format or an EXIF modify date earlier than
its capture date is internally inconsistent. Metadata is trivially stripped
or forged, so its *absence* proves nothing and its presence is only as
trustworthy as the chain of custody; what this analyzer surfaces is whether
the pieces agree with each other.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from verity.forensics.base import ForensicAnalyzer, r, write_json
from verity.forensics.models import ForensicResult, Observation

#: Substrings (case-insensitive) in encoder / software tags that identify a
#: transcoder or an editing application rather than a capture device.
SOFTWARE_SIGNATURES = [
    "lavf", "lavc", "ffmpeg", "libav", "x264", "x265", "handbrake", "adobe",
    "photoshop", "lightroom", "premiere", "after effects", "media encoder",
    "gimp", "davinci", "final cut", "imovie", "snapseed", "canva", "picasa",
    "paint.net", "affinity", "capture one", "vegas", "filmora", "capcut",
    "kinemaster", "shotcut", "blender", "openshot", "avidemux", "virtualdub",
    "mencoder", "google", "facebook", "instagram", "tiktok", "whatsapp",
    "telegram", "snapchat", "mp4box", "gpac", "obs", "stable diffusion",
    "midjourney", "dall", "firefly", "comfyui", "automatic1111",
]

#: Common display aspect ratios (w/h); anything else is merely noted.
COMMON_ASPECTS = [16 / 9, 9 / 16, 4 / 3, 3 / 4, 3 / 2, 2 / 3, 1.0, 21 / 9, 5 / 4, 4 / 5, 1.85, 2.39]

#: Pixel formats consumer capture devices actually record.
CAPTURE_PIX_FMTS = {"yuv420p", "yuvj420p", "nv12", "nv21", "yuv420p10le", "p010le"}

#: Frames to inspect for GOP structure (keeps ffprobe decoding bounded).
GOP_PROBE_FRAMES = 600

#: Camera-originated JPEGs carry at least these EXIF fields.
CAMERA_EXIF_FIELDS = ("Make", "Model", "DateTimeOriginal")


def _ffprobe(path: Path) -> Optional[dict]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def _ffprobe_gop(path: Path, max_frames: int = GOP_PROBE_FRAMES) -> Optional[List[str]]:
    """Picture types (I/P/B) of the first ``max_frames`` video frames."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "frame=pict_type", "-of", "csv=p=0",
             "-read_intervals", f"%+#{max_frames}", str(path)],
            capture_output=True, text=True, timeout=300,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    types = [line.strip().strip(",") for line in out.stdout.splitlines() if line.strip()]
    return types[:max_frames] or None


def _fraction(text: Any) -> Optional[float]:
    try:
        num, den = str(text).split("/")
        den = float(den)
        return float(num) / den if den else None
    except (ValueError, AttributeError):
        return None


def _float(text: Any) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _parse_time(text: Any) -> Optional[datetime]:
    """Parse ISO-8601 (ffprobe) and ``YYYY:MM:DD HH:MM:SS`` (EXIF) stamps."""
    if not text:
        return None
    s = str(text).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _software_hits(value: str) -> List[str]:
    low = value.lower()
    return [sig for sig in SOFTWARE_SIGNATURES if sig in low]


def _exif(path: Path) -> Dict[str, Any]:
    """EXIF as {tag name: string value}, merged from IFD0 and the Exif IFD."""
    try:
        from PIL import ExifTags, Image
        with Image.open(path) as img:
            exif = img.getexif()
            out: Dict[str, Any] = {}
            for tag, value in exif.items():
                out[ExifTags.TAGS.get(tag, str(tag))] = _exif_value(value)
            try:
                for tag, value in exif.get_ifd(0x8769).items():
                    out[ExifTags.TAGS.get(tag, str(tag))] = _exif_value(value)
            except Exception:  # noqa: BLE001 - malformed sub-IFD, keep what we have
                pass
            return dict(sorted(out.items()))
    except Exception:  # noqa: BLE001 - not an image Pillow can open
        return {}


def _exif_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value[:64].hex()
    if isinstance(value, (tuple, list)):
        return [_exif_value(v) for v in value[:16]]
    if isinstance(value, (int, float, str)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)[:200]


class MetadataAnalyzer(ForensicAnalyzer):
    analyzer_id = "metadata"
    label = "Metadata consistency"
    media_types = ("image", "video", "audio")
    method = (
        "Reads the container and stream descriptors with ffprobe and the EXIF "
        "block with Pillow, then checks them against each other and against "
        "what a capture device would write. Encoder tags, pixel formats and "
        "timestamps can be stripped or forged at will, so a clean result is not "
        "proof of originality; a contradictory one shows that some processing "
        "took place, without saying what."
    )

    def analyze(self, media_id: str, media_type: str) -> ForensicResult:
        path = self.media_path(media_id)
        out_dir = self.output_dir(media_id)
        probe = _ffprobe(path)
        exif = _exif(path) if media_type == "image" else {}
        normalized = self._normalize(path, probe, exif)
        observations: List[Observation] = []

        if probe is None:
            observations.append(self.observation(
                "notice", "ffprobe could not parse the container, so stream-level checks were skipped."))
        else:
            observations.extend(self._encoder_observations(normalized))
            observations.extend(self._time_observations(normalized, exif))
            observations.extend(self._stream_observations(normalized, media_type))
            if media_type == "video":
                gop = _ffprobe_gop(path)
                normalized["gop"] = self._gop_stats(gop)
                observations.extend(self._gop_observations(normalized["gop"]))

        if media_type == "image":
            observations.extend(self._exif_observations(path, exif))

        meta_path = write_json(out_dir / "metadata.json", normalized)
        obs_path = write_json(out_dir / "observations.json",
                              {"observations": [o.model_dump() for o in observations]})
        return self.result(observations, [meta_path, obs_path], data={
            "container": normalized.get("container", {}).get("format_name"),
            "encoder_tags": normalized.get("encoder_tags", {}),
            "stream_count": len(normalized.get("streams", [])),
            "exif_fields": len(exif),
        })

    # ------------------------------------------------------------------ #
    #  Normalisation                                                       #
    # ------------------------------------------------------------------ #

    def _normalize(self, path: Path, probe: Optional[dict], exif: Dict[str, Any]) -> Dict[str, Any]:
        norm: Dict[str, Any] = {
            "file": {"name": path.name, "size_bytes": path.stat().st_size, "suffix": path.suffix.lower()},
            "container": {},
            "streams": [],
            "encoder_tags": {},
            "exif": exif,
        }
        if probe is None:
            return norm
        fmt = probe.get("format", {}) or {}
        tags = {k.lower(): str(v) for k, v in (fmt.get("tags") or {}).items()}
        norm["container"] = {
            "format_name": fmt.get("format_name"),
            "format_long_name": fmt.get("format_long_name"),
            "duration": r(_float(fmt.get("duration")), 3),
            "bit_rate": int(_float(fmt.get("bit_rate")) or 0) or None,
            "nb_streams": fmt.get("nb_streams"),
            "tags": dict(sorted(tags.items())),
        }
        if "encoder" in tags:
            norm["encoder_tags"]["container"] = tags["encoder"]
        for s in probe.get("streams", []) or []:
            stags = {k.lower(): str(v) for k, v in (s.get("tags") or {}).items()}
            entry = {
                "index": s.get("index"),
                "codec_type": s.get("codec_type"),
                "codec_name": s.get("codec_name"),
                "profile": s.get("profile"),
                "width": s.get("width"),
                "height": s.get("height"),
                "pix_fmt": s.get("pix_fmt"),
                "r_frame_rate": s.get("r_frame_rate"),
                "avg_frame_rate": s.get("avg_frame_rate"),
                "nb_frames": int(_float(s.get("nb_frames")) or 0) or None,
                "duration": r(_float(s.get("duration")), 3),
                "bit_rate": int(_float(s.get("bit_rate")) or 0) or None,
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "has_b_frames": s.get("has_b_frames"),
                "field_order": s.get("field_order"),
                "tags": dict(sorted(stags.items())),
            }
            norm["streams"].append({k: v for k, v in entry.items() if v is not None})
            if "encoder" in stags:
                norm["encoder_tags"][f"stream_{s.get('index')}"] = stags["encoder"]
        if exif.get("Software"):
            norm["encoder_tags"]["exif_software"] = str(exif["Software"])
        return norm

    # ------------------------------------------------------------------ #
    #  Checks                                                              #
    # ------------------------------------------------------------------ #

    def _encoder_observations(self, norm: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        tags = norm["encoder_tags"]
        for where, value in sorted(tags.items()):
            hits = _software_hits(value)
            if hits:
                obs.append(self.observation(
                    "notice",
                    f"The {where.replace('_', ' ')} encoder tag '{value}' names processing "
                    f"software ({', '.join(hits)}), so the file was written by a transcoder or "
                    f"editor rather than directly by a capture device.",
                    where=where, value=value, signatures=hits))
            else:
                obs.append(self.observation(
                    "info", f"The {where.replace('_', ' ')} encoder tag is '{value}'.",
                    where=where, value=value))
        container = tags.get("container")
        streams = {k: v for k, v in tags.items() if k.startswith("stream_")}
        if container and streams:
            differing = {k: v for k, v in streams.items() if v != container}
            if differing:
                obs.append(self.observation(
                    "info",
                    "The container and the stream carry different encoder tags, which is normal "
                    "when a muxer wraps an already encoded stream but also means the file was "
                    "assembled in at least two steps.",
                    container=container, **differing))
        handler = next((s["tags"].get("handler_name") for s in norm["streams"]
                        if s.get("codec_type") == "video" and s["tags"].get("handler_name")), None)
        if handler:
            hits = _software_hits(handler)
            obs.append(self.observation(
                "notice" if hits else "info",
                f"The video track handler name is '{handler}'"
                + (" (a software muxer default, not a device name)." if hits or handler.lower()
                   in ("videohandler", "video handler", "core media video") else "."),
                handler_name=handler))
        return obs

    def _time_observations(self, norm: Dict[str, Any], exif: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        ctags = norm["container"].get("tags", {})
        c_created = _parse_time(ctags.get("creation_time"))
        stream_times = {}
        for s in norm["streams"]:
            t = _parse_time(s["tags"].get("creation_time"))
            if t is not None:
                stream_times[s["index"]] = t
        if c_created is not None and stream_times:
            drift = {idx: abs((t - c_created).total_seconds()) for idx, t in stream_times.items()}
            worst = max(drift.values())
            if worst > 2.0:
                obs.append(self.observation(
                    "warning",
                    f"Container and stream creation times disagree by up to {worst:.0f} s; "
                    "a single capture writes them together, so the streams were combined "
                    "from different sources or re-muxed later.",
                    container_creation_time=ctags.get("creation_time"),
                    max_drift_seconds=r(worst, 1)))
        if c_created is None and norm["streams"] and norm["container"].get("format_name", "").startswith("mov"):
            obs.append(self.observation(
                "info",
                "The MP4/MOV container has no creation_time tag; cameras and phones normally "
                "write one, transcoders often drop it.",
            ))
        original = _parse_time(exif.get("DateTimeOriginal"))
        modified = _parse_time(exif.get("DateTime"))
        digitized = _parse_time(exif.get("DateTimeDigitized"))
        if original is not None and modified is not None:
            delta = (modified - original).total_seconds()
            if delta < 0:
                obs.append(self.observation(
                    "warning",
                    f"EXIF modify time precedes capture time by {-delta:.0f} s, an ordering no "
                    "camera produces; one of the two stamps was rewritten.",
                    DateTimeOriginal=exif.get("DateTimeOriginal"), DateTime=exif.get("DateTime")))
            elif delta > 60:
                obs.append(self.observation(
                    "notice",
                    f"EXIF modify time is {delta / 3600:.1f} h after capture time: the image was "
                    "re-saved after it was taken.",
                    DateTimeOriginal=exif.get("DateTimeOriginal"), DateTime=exif.get("DateTime"),
                    delta_seconds=r(delta, 0)))
        if original is not None and digitized is not None and original != digitized:
            obs.append(self.observation(
                "notice",
                "EXIF DateTimeOriginal and DateTimeDigitized differ; cameras write identical "
                "values, scanners and editors do not.",
                DateTimeOriginal=exif.get("DateTimeOriginal"),
                DateTimeDigitized=exif.get("DateTimeDigitized")))
        return obs

    def _stream_observations(self, norm: Dict[str, Any], media_type: str) -> List[Observation]:
        obs: List[Observation] = []
        streams = norm["streams"]
        video = [s for s in streams if s.get("codec_type") == "video"]
        audio = [s for s in streams if s.get("codec_type") == "audio"]
        other = [s for s in streams if s.get("codec_type") not in ("video", "audio")]
        if media_type == "video":
            if not audio:
                obs.append(self.observation(
                    "info", "The video carries no audio track.", video_streams=len(video), audio_streams=0))
            if len(video) > 1:
                obs.append(self.observation(
                    "notice", f"The container holds {len(video)} video streams.", video_streams=len(video)))
            if other:
                obs.append(self.observation(
                    "info", f"The container holds {len(other)} non-audio/video stream(s) "
                    f"({', '.join(sorted(str(s.get('codec_name')) for s in other))}).",
                    other_streams=len(other)))
        if media_type == "audio" and video:
            obs.append(self.observation(
                "notice", "An audio file that also carries a video stream (cover art or an embedded picture).",
                video_streams=len(video)))

        for s in video[:1]:
            w, h = s.get("width"), s.get("height")
            pix = s.get("pix_fmt")
            if pix and pix not in CAPTURE_PIX_FMTS:
                obs.append(self.observation(
                    "notice",
                    f"Pixel format {pix} is not what capture devices record (they write 4:2:0 "
                    "chroma); the stream was encoded by software, typically from image frames or "
                    "an intermediate.",
                    pix_fmt=pix, profile=s.get("profile")))
            if w and h:
                if w % 2 or h % 2:
                    obs.append(self.observation(
                        "notice", f"Odd frame dimensions {w}x{h}; encoders and cameras use even sizes, "
                        "so the frame was cropped or scaled in software.", width=w, height=h))
                aspect = w / h
                if not any(abs(aspect - a) / a < 0.01 for a in COMMON_ASPECTS):
                    obs.append(self.observation(
                        "info", f"Frame size {w}x{h} has an uncommon aspect ratio ({aspect:.3f}), "
                        "which suggests cropping.", width=w, height=h, aspect=r(aspect, 3)))
            r_rate, a_rate = _fraction(s.get("r_frame_rate")), _fraction(s.get("avg_frame_rate"))
            if r_rate and a_rate and abs(r_rate - a_rate) / r_rate > 0.01:
                obs.append(self.observation(
                    "notice",
                    f"Variable frame rate: the nominal rate is {r_rate:.3f} fps but the average is "
                    f"{a_rate:.3f} fps, so frames were dropped, duplicated or the clip was assembled "
                    "from sources with different timing.",
                    r_frame_rate=s.get("r_frame_rate"), avg_frame_rate=s.get("avg_frame_rate")))
            nb = s.get("nb_frames")
            dur = s.get("duration") or norm["container"].get("duration")
            if nb and dur and a_rate:
                expected = dur * a_rate
                if expected > 0 and abs(nb - expected) / expected > 0.02:
                    obs.append(self.observation(
                        "notice",
                        f"The stream declares {nb} frames but duration x frame rate gives "
                        f"{expected:.0f}; the frame count or timing was edited.",
                        nb_frames=nb, expected_frames=r(expected, 1)))
            if s.get("field_order") not in (None, "progressive", "unknown"):
                obs.append(self.observation(
                    "info", f"Interlaced video (field order {s.get('field_order')}).",
                    field_order=s.get("field_order")))
        for s in audio[:1]:
            sr = _float(s.get("sample_rate"))
            if sr and sr < 32000:
                obs.append(self.observation(
                    "info",
                    f"Audio sample rate is {int(sr)} Hz; speech synthesis and telephony pipelines "
                    "commonly produce 16-24 kHz audio while cameras record 44.1-48 kHz.",
                    sample_rate=int(sr), codec=s.get("codec_name")))
        return obs

    @staticmethod
    def _gop_stats(types: Optional[List[str]]) -> Dict[str, Any]:
        if not types:
            return {"frames_probed": 0}
        key_idx = [i for i, t in enumerate(types) if t == "I"]
        gaps = [b - a for a, b in zip(key_idx, key_idx[1:])]
        counts = {t: types.count(t) for t in sorted(set(types))}
        return {
            "frames_probed": len(types),
            "pict_type_counts": counts,
            "keyframes": len(key_idx),
            "keyframe_positions": key_idx[:64],
            "gop_min": min(gaps) if gaps else None,
            "gop_max": max(gaps) if gaps else None,
            "gop_mean": r(sum(gaps) / len(gaps), 2) if gaps else None,
        }

    def _gop_observations(self, gop: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        if not gop.get("frames_probed"):
            return obs
        if gop.get("gop_min") and gop.get("gop_max"):
            if gop["gop_max"] > 2 * gop["gop_min"] and gop["keyframes"] >= 3:
                obs.append(self.observation(
                    "notice",
                    f"Keyframe spacing is irregular ({gop['gop_min']}-{gop['gop_max']} frames over "
                    f"{gop['keyframes']} keyframes in the first {gop['frames_probed']} frames); "
                    "encoders insert extra keyframes at scene changes, so these mark cuts or splice "
                    "points worth checking against the temporal timeline.",
                    gop_min=gop["gop_min"], gop_max=gop["gop_max"], keyframes=gop["keyframes"]))
            else:
                obs.append(self.observation(
                    "info",
                    f"Regular keyframe spacing of about {gop['gop_mean']} frames "
                    f"({gop['keyframes']} keyframes in the first {gop['frames_probed']} frames).",
                    gop_mean=gop["gop_mean"], keyframes=gop["keyframes"]))
        elif gop["keyframes"] == 1:
            obs.append(self.observation(
                "info", f"A single keyframe in the first {gop['frames_probed']} frames (long GOP).",
                frames_probed=gop["frames_probed"]))
        if gop.get("pict_type_counts", {}).get("B", 0) == 0 and gop["frames_probed"] > 30:
            obs.append(self.observation(
                "info", "No B-frames in the probed range; many software encoders and some phones "
                "disable them, so this is a weak encoder fingerprint.", frames_probed=gop["frames_probed"]))
        return obs

    def _exif_observations(self, path: Path, exif: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".heic", ".tif", ".tiff"):
            if not exif:
                obs.append(self.observation(
                    "notice",
                    "No EXIF metadata at all; camera-originated files carry make, model and capture "
                    "time, so the block was stripped (common on messaging and social platforms) or "
                    "the image was rendered by software.",
                    suffix=suffix))
            else:
                missing = [f for f in CAMERA_EXIF_FIELDS if f not in exif]
                if missing:
                    obs.append(self.observation(
                        "notice",
                        f"EXIF is present but lacks {', '.join(missing)}, which cameras always "
                        "write; an editor rewrote the block or the file was never a camera capture.",
                        missing=missing, present_fields=len(exif)))
                else:
                    obs.append(self.observation(
                        "info",
                        f"Camera EXIF present: {exif.get('Make', '')} {exif.get('Model', '')}, "
                        f"captured {exif.get('DateTimeOriginal')}.",
                        Make=exif.get("Make"), Model=exif.get("Model"),
                        DateTimeOriginal=exif.get("DateTimeOriginal")))
        elif exif:
            obs.append(self.observation(
                "info", f"{len(exif)} EXIF field(s) present in a {suffix} file.", present_fields=len(exif)))
        software = str(exif.get("Software", ""))
        if software and not _software_hits(software):
            obs.append(self.observation("info", f"EXIF Software tag is '{software}'.", Software=software))
        if exif.get("ExifImageWidth") and exif.get("ExifImageHeight"):
            try:
                from PIL import Image
                with Image.open(path) as img:
                    actual = img.size
                claimed = (int(exif["ExifImageWidth"]), int(exif["ExifImageHeight"]))
                if actual != claimed and actual != claimed[::-1]:
                    obs.append(self.observation(
                        "notice",
                        f"EXIF records a size of {claimed[0]}x{claimed[1]} but the image is "
                        f"{actual[0]}x{actual[1]}: it was resized or cropped after the block was written.",
                        exif_size=list(claimed), actual_size=list(actual)))
            except (TypeError, ValueError, OSError):
                pass
        return obs
