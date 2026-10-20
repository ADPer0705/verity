"""Audio track statistics and a log-mel spectrogram.

Rationale: synthetic speech (TTS, voice conversion, vocoders) is usually
generated at 16-24 kHz and band-limited, so the long-term spectrum has
almost no energy above 8 kHz where a microphone recording in a room still
has noise and sibilance. Generated audio also tends to have an unnaturally
low noise floor between phrases (very clean silence), no DC offset and no
clipping, whereas field recordings show the opposite. The spectrogram lets
an examiner see phrase boundaries, missing breaths, repeated identical
segments or a hard spectral cut-off.

The analysis decodes the original audio stream at its native sample rate
(the pipeline's audio extractor resamples to 16 kHz, which would hide the
very band this test looks at) and falls back to the extracted clip if
decoding fails.

Limits: telephone, Bluetooth and heavily compressed audio are band-limited
for innocent reasons; a clean studio recording also has a low noise floor.
Nothing here identifies a speaker or a generator.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verity.forensics.base import ForensicAnalyzer, r, write_json
from verity.forensics.models import ForensicResult, Observation

#: Seconds of audio analysed (bounded so long recordings stay cheap).
MAX_SECONDS = 180.0
FRAME_MS = 25.0
HOP_MS = 10.0
MEL_BANDS = 64
SILENCE_DBFS = -50.0
CLIP_LEVEL = 0.985
HIGH_BAND_HZ = 8000.0
#: Below this high/low band energy ratio the track is called band-limited.
BAND_RATIO_DB = -45.0


def _ffprobe_audio(path: Path) -> Optional[Dict[str, Any]]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels,bit_rate,duration", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    import json
    try:
        streams = json.loads(out.stdout).get("streams") or []
    except json.JSONDecodeError:
        return None
    return streams[0] if streams else None


def _decode(path: Path, out_wav: Path, seconds: float) -> bool:
    cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(path), "-vn", "-t", f"{seconds:.3f}",
           "-ac", "1", "-acodec", "pcm_s16le", "-map_metadata", "-1", "-fflags", "+bitexact",
           "-flags:a", "+bitexact", str(out_wav)]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=600).returncode == 0 and out_wav.exists()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def read_wav(path: Path) -> Tuple[np.ndarray, int, int]:
    """Mono float32 samples in [-1, 1], sample rate, original channel count."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"unsupported WAV sample width {width}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr, ch


def stft_power(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Power spectrogram (frames x bins) with a Hann window, numpy only."""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]
    spec = np.fft.rfft(frames, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float64)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)  # noqa: E731
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, c, hi = bins[m - 1], bins[m], bins[m + 1]
        if c <= lo:
            c = lo + 1
        if hi <= c:
            hi = c + 1
        for k in range(lo, min(c, fb.shape[1])):
            fb[m - 1, k] = (k - lo) / (c - lo)
        for k in range(c, min(hi, fb.shape[1])):
            fb[m - 1, k] = (hi - k) / (hi - c)
    return fb


class AudioAnalyzer(ForensicAnalyzer):
    analyzer_id = "audio"
    label = "Audio track"
    media_types = ("video", "audio")
    method = (
        "The first audio stream is decoded at its native sample rate and summarised: "
        "duration, sample rate, channels, the share of 25 ms frames below -50 dBFS "
        "(silence), the share of samples at full scale (clipping), the DC offset and "
        "the long-term energy above versus below 8 kHz, where microphone recordings "
        "keep room noise and sibilance but many speech synthesisers and vocoders "
        "output nothing. A log-mel spectrogram is rendered for visual inspection. "
        "Band limiting also comes from telephony, Bluetooth and low-bitrate codecs, "
        "and a studio recording is legitimately clean, so these are prompts for "
        "listening, not conclusions."
    )

    def analyze(self, media_id: str, media_type: str) -> ForensicResult:
        path = self.media_path(media_id)
        out_dir = self.output_dir(media_id)
        stream = _ffprobe_audio(path)
        if stream is None:
            obs = [self.observation("info", "No audio stream in the file; audio analyses do not apply.")]
            artifacts = [write_json(out_dir / "audio.json", {"has_audio": False})]
            return self.result(obs, artifacts, data={"has_audio": False})

        wav = out_dir / "audio_native.wav"
        source = "native decode"
        if not _decode(path, wav, MAX_SECONDS):
            fallback = self.artifact_dir / media_id / "audio" / "audio.wav"
            if not fallback.exists():
                raise RuntimeError("ffmpeg could not decode the audio stream and no extracted clip exists")
            wav, source = fallback, "extracted clip (16 kHz)"
        samples, sr, channels = read_wav(wav)
        if len(samples) < sr // 10:
            raise RuntimeError("audio stream shorter than 100 ms")

        data = self._stats(samples, sr)
        data.update({
            "has_audio": True,
            "source": source,
            "codec": stream.get("codec_name"),
            "declared_sample_rate": int(float(stream.get("sample_rate", 0) or 0)) or None,
            "declared_channels": stream.get("channels"),
            "declared_duration": r(float(stream["duration"]), 3) if stream.get("duration") else None,
            "analysed_seconds_limit": MAX_SECONDS,
        })
        spec_path = self._spectrogram(out_dir / "spectrogram.png", samples, sr)
        artifacts = [spec_path, write_json(out_dir / "audio.json", data)]
        if source == "native decode":
            artifacts.append(wav)
        observations = self._observations(data)
        return self.result(observations, artifacts, data={
            "sample_rate": sr,
            "duration": data["duration"],
            "silence_ratio": data["silence_ratio"],
            "clipping_ratio": data["clipping_ratio"],
            "high_band_ratio_db": data["high_band_ratio_db"],
        })

    # ------------------------------------------------------------------ #

    def _stats(self, x: np.ndarray, sr: int) -> Dict[str, Any]:
        n_fft = int(2 ** np.ceil(np.log2(sr * FRAME_MS / 1000.0)))
        hop = max(1, int(sr * HOP_MS / 1000.0))
        power = stft_power(x, n_fft, hop)
        frame_rms = np.sqrt(power.mean(axis=1) / n_fft + 1e-12)
        frame_db = 20.0 * np.log10(frame_rms + 1e-9)
        silence = float((frame_db < SILENCE_DBFS).mean())
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        long_term = power.mean(axis=0)
        high = float(long_term[freqs >= HIGH_BAND_HZ].sum()) if sr / 2 > HIGH_BAND_HZ else None
        low = float(long_term[freqs < HIGH_BAND_HZ].sum()) + 1e-12
        ratio_db = r(10.0 * np.log10(high / low + 1e-15), 2) if high is not None else None
        cumulative = np.cumsum(long_term) / (long_term.sum() + 1e-12)
        rolloff = float(freqs[int(np.searchsorted(cumulative, 0.99))]) if long_term.sum() > 0 else 0.0
        active = frame_db[frame_db >= SILENCE_DBFS]
        quiet = frame_db[frame_db < SILENCE_DBFS]
        return {
            "sample_rate": int(sr),
            "duration": r(len(x) / sr, 3),
            "samples": int(len(x)),
            "frame_ms": FRAME_MS,
            "n_fft": n_fft,
            "silence_threshold_dbfs": SILENCE_DBFS,
            "silence_ratio": r(silence, 4),
            "clipping_level": CLIP_LEVEL,
            "clipping_ratio": r(float((np.abs(x) >= CLIP_LEVEL).mean()), 6),
            "dc_offset": r(float(x.mean()), 6),
            "peak": r(float(np.abs(x).max()), 4),
            "rms_dbfs": r(20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9), 2),
            "noise_floor_dbfs": r(float(np.percentile(frame_db, 5)), 2),
            "active_level_dbfs": r(float(np.median(active)), 2) if len(active) else None,
            "quiet_level_dbfs": r(float(np.median(quiet)), 2) if len(quiet) else None,
            "high_band_hz": HIGH_BAND_HZ,
            "high_band_ratio_db": ratio_db,
            "band_ratio_threshold_db": BAND_RATIO_DB,
            "rolloff_99pct_hz": r(rolloff, 1),
            "nyquist_hz": r(sr / 2.0, 1),
        }

    def _spectrogram(self, path: Path, x: np.ndarray, sr: int) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_fft = int(2 ** np.ceil(np.log2(sr * FRAME_MS / 1000.0)))
        hop = max(1, int(sr * HOP_MS / 1000.0))
        # Cap the number of rendered columns so long clips stay legible.
        max_frames = 3000
        stride = max(1, int(np.ceil((1 + (len(x) - n_fft) // hop) / max_frames)))
        power = stft_power(x, n_fft, hop * stride)
        fb = mel_filterbank(sr, n_fft, MEL_BANDS)
        mel = np.log10(power @ fb.T + 1e-10).T
        duration = len(x) / sr
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
        img = ax.imshow(mel, aspect="auto", origin="lower", cmap="magma",
                        extent=[0, duration, 0, sr / 2000.0],
                        vmin=float(np.percentile(mel, 2)), vmax=float(np.percentile(mel, 99.5)))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mel band -> kHz (approx.)")
        ax.set_title(f"Log-mel spectrogram ({MEL_BANDS} bands, {sr} Hz)", fontsize=9)
        ax.axhline(HIGH_BAND_HZ / 1000.0, color="white", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.tick_params(labelsize=8)
        fig.colorbar(img, ax=ax, pad=0.01).ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(path, metadata={"Software": None})
        plt.close(fig)
        return path

    def _observations(self, d: Dict[str, Any]) -> List[Observation]:
        obs: List[Observation] = []
        obs.append(self.observation(
            "info",
            f"Audio: {d['codec'] or 'unknown codec'}, {d['sample_rate']} Hz, "
            f"{d['declared_channels'] or '?'} channel(s), {d['duration']:.1f} s analysed "
            f"({d['source']}), level {d['rms_dbfs']} dBFS.",
            sample_rate=d["sample_rate"], channels=d["declared_channels"], duration=d["duration"],
            rms_dbfs=d["rms_dbfs"], source=d["source"]))
        if d["sample_rate"] < 32000:
            obs.append(self.observation(
                "notice",
                f"Sample rate {d['sample_rate']} Hz caps the band at {d['nyquist_hz'] / 1000:.0f} kHz; "
                "cameras and phones record at 44.1-48 kHz, while speech synthesis pipelines and "
                "telephony work at 16-24 kHz, so the above-8 kHz test is not applicable.",
                sample_rate=d["sample_rate"]))
        elif d["high_band_ratio_db"] is not None:
            if d["high_band_ratio_db"] < BAND_RATIO_DB:
                obs.append(self.observation(
                    "notice",
                    f"Almost no energy above 8 kHz ({d['high_band_ratio_db']} dB relative to the band "
                    f"below, 99% roll-off at {d['rolloff_99pct_hz'] / 1000:.1f} kHz) despite a "
                    f"{d['sample_rate']} Hz sample rate: the audio was upsampled from a band-limited "
                    "source, such as a vocoder, a phone call or a low-bitrate codec.",
                    high_band_ratio_db=d["high_band_ratio_db"], rolloff_99pct_hz=d["rolloff_99pct_hz"]))
            else:
                obs.append(self.observation(
                    "info",
                    f"Energy above 8 kHz is {d['high_band_ratio_db']} dB relative to the band below "
                    f"(99% roll-off at {d['rolloff_99pct_hz'] / 1000:.1f} kHz), consistent with a "
                    "full-band recording.",
                    high_band_ratio_db=d["high_band_ratio_db"], rolloff_99pct_hz=d["rolloff_99pct_hz"]))
        if d["clipping_ratio"] and d["clipping_ratio"] > 0.001:
            obs.append(self.observation(
                "notice", f"{d['clipping_ratio']:.2%} of samples sit at full scale: the recording or a "
                "later gain stage clipped.", clipping_ratio=d["clipping_ratio"]))
        if d["dc_offset"] is not None and abs(d["dc_offset"]) > 0.01:
            obs.append(self.observation(
                "notice", f"DC offset of {d['dc_offset']:+.4f} full scale; analogue capture chains leave "
                "one, digital synthesis does not.", dc_offset=d["dc_offset"]))
        if d["silence_ratio"] is not None and d["silence_ratio"] > 0.5:
            obs.append(self.observation(
                "info", f"{d['silence_ratio']:.0%} of the track is below {SILENCE_DBFS:.0f} dBFS.",
                silence_ratio=d["silence_ratio"]))
        if d["quiet_level_dbfs"] is not None and d["quiet_level_dbfs"] < -75 and d["silence_ratio"] > 0.05:
            obs.append(self.observation(
                "notice",
                f"Pauses are nearly digital silence (median {d['quiet_level_dbfs']} dBFS): a room "
                "microphone leaves a noise floor, while gating, noise suppression and synthesis do not.",
                quiet_level_dbfs=d["quiet_level_dbfs"], silence_ratio=d["silence_ratio"]))
        return obs
