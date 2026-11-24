"""In-process detector environment manager and hardware/air-gap runtime."""
from verity.runtime.env_manager import (  # noqa: F401
    detect_profile,
    detector_ids,
    ensure_ffmpeg,
    ensure_uv,
    find_uv,
    run_detector_process,
    setup_detectors,
    trained_detector_ids,
    warmup_detectors,
)
