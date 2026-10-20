import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # detectors/ -> common
from common.engine import run_detector  # noqa: E402
from model import build_model  # noqa: E402

if __name__ == "__main__":
    run_detector(build_model, Path(__file__).resolve().parent / "detector.yaml")
