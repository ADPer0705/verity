#!/usr/bin/env python3
"""Run every Verity detector through the CLI in tier order (light -> heavy).

For each detector it executes ``verity run --detector <id>`` (the full
orchestration path: ingest -> analyze -> select -> detect -> compile ->
report) while sampling the process tree's RSS and the GPU's VRAM every second.
When a detector finishes, one result row is appended to docs/matrix_runs.md
and docs/matrix_runs.jsonl.

If the machine runs out of memory and this script itself is killed mid-matrix,
the log shows exactly which detectors finished; re-run with --resume to
continue from the first detector that did not complete.

Usage:
    python scripts/run_detector_matrix.py                  # all detectors
    python scripts/run_detector_matrix.py --tiers L        # light only
    python scripts/run_detector_matrix.py --detectors meso4,xception
    python scripts/run_detector_matrix.py --resume         # continue after a kill
    python scripts/run_detector_matrix.py --list           # print the order
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
DEFAULT_MEDIA = None  # pass --media; there is no bundled sample

TIERS: dict[str, list[str]] = {
    "L": ["meso4", "meso4Inception", "capsule_net", "npr", "mlep"],
    "M": [
        "xception", "core", "ffd", "spsl", "f3net", "fwa", "rfm", "sladd",
        "pcl_xception", "resnet34", "efficientnetb4", "sbi", "sia", "lsda",
        "multi_attention", "lrl", "iid", "ucf", "srm", "recce", "facexray",
        "clip", "uia_vit", "tall", "stil", "timesformer", "lipforensics",
    ],
    "H": [
        "i3d", "altfreezing", "ftcn", "videomae", "xclip", "effort",
        "gend", "univfd", "aeroblade",
    ],
}
TIER_ORDER = ["L", "M", "H"]

TERMINAL = {"ok", "failed", "skipped"}


def log(msg: str) -> None:
    print(f"[matrix] {msg}", flush=True)


# --------------------------------------------------------------------------- #
#  Monitoring                                                                 #
# --------------------------------------------------------------------------- #

def _mem_available_mb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _tree_rss_kb(root_pid: int) -> int:
    """Sum the current RSS of the root process and all its descendants."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0
    children: dict[int, list[tuple[int, int]]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append((pid, rss))
    total, stack, seen = 0, [root_pid], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child_pid, rss in children.get(pid, []):
            total += rss
            stack.append(child_pid)
    return total


def _gpu_mem_mb() -> int | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except (subprocess.SubprocessError, OSError, ValueError, IndexError):
        return None


class Monitor:
    """Samples process-tree RSS and GPU VRAM until the watched process exits."""

    def __init__(self, root_pid: int, interval: float):
        self.root_pid = root_pid
        self.interval = interval
        self.peak_rss_kb = 0
        self.peak_vram_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "Monitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_kb = max(self.peak_rss_kb, _tree_rss_kb(self.root_pid))
            vram = _gpu_mem_mb()
            if vram is not None:
                self.peak_vram_mb = max(self.peak_vram_mb, vram)
            self._stop.wait(self.interval)
        self.peak_rss_kb = max(self.peak_rss_kb, _tree_rss_kb(self.root_pid))
        vram = _gpu_mem_mb()
        if vram is not None:
            self.peak_vram_mb = max(self.peak_vram_mb, vram)


# --------------------------------------------------------------------------- #
#  Result extraction                                                          #
# --------------------------------------------------------------------------- #

def _newest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def _finding_summary(detector_id: str) -> dict:
    """Pull classification / prediction / confidence from the newest finding."""
    summary = {"classification": "", "mean_fake_prob": "", "confidence": "", "faces": ""}
    candidates = sorted(
        (WORKSPACE / "output").glob(f"*_{detector_id}/finding.json"),
        key=lambda p: p.stat().st_mtime,
    )
    finding_path = candidates[-1] if candidates else None
    if finding_path is None:
        return summary
    try:
        data = json.loads(finding_path.read_text())
        raw = data.get("raw_output", {}) or {}
        per_face = raw.get("per_face") or []
        probs = [f.get("fake_prob") for f in per_face if isinstance(f.get("fake_prob"), (int, float))]
        if probs:
            summary["mean_fake_prob"] = f"{sum(probs) / len(probs):.4f}"
        elif isinstance(raw.get("prediction"), (int, float)):
            summary["mean_fake_prob"] = f"{raw['prediction']:.4f}"
        summary["classification"] = str(data.get("classification", ""))
        conf = data.get("confidence")
        if isinstance(conf, (int, float)):
            summary["confidence"] = f"{conf:.4f}"
        stats = raw.get("stats") or {}
        if stats.get("count") is not None:
            summary["faces"] = str(stats["count"])
        elif per_face:
            summary["faces"] = str(len(per_face))
    except (OSError, ValueError, TypeError):
        pass
    return summary


def _result_timing(detector_id: str) -> str:
    candidates = sorted(
        (WORKSPACE / "output").glob(f"*_{detector_id}/result.json"),
        key=lambda p: p.stat().st_mtime,
    )
    result_path = candidates[-1] if candidates else None
    if result_path is None:
        return ""
    try:
        data = json.loads(result_path.read_text())
        timing = data.get("timing_seconds", {})
        device = data.get("device", "")
        total = timing.get("total")
        return f"device={device}" + (f"; engine_total={total}s" if total else "")
    except (OSError, ValueError, TypeError):
        return ""


# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #

def _ensure_logs(log_md: Path, log_jsonl: Path) -> None:
    log_md.parent.mkdir(parents=True, exist_ok=True)
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not log_md.exists():
        log_md.write_text(
            "# Verity detector matrix runs\n\n"
            "Appended by scripts/run_detector_matrix.py. One row per completed "
            "detector; if the machine OOMs mid-run, rows already written were "
            "completed. Peak RSS is the sum over the verity process tree, peak "
            "VRAM is the whole-GPU usage (nothing else runs concurrently).\n\n"
            "| time (UTC) | detector | tier | status | peak RSS MB | peak VRAM MB | "
            "wall s | mean fake prob | confidence | faces | notes |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n"
        )


def _write_row(log_md: Path, log_jsonl: Path, row: dict) -> None:
    row["time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    log_jsonl.open("a").write(json.dumps(row) + "\n")
    notes = (row.get("notes") or "").replace("|", "\\|")
    log_md.open("a").write(
        f"| {row['time']} | {row['detector']} | {row['tier']} | {row['status']} "
        f"| {row.get('peak_rss_mb') or ''} | {row.get('peak_vram_mb') or ''} "
        f"| {row.get('wall_s') or ''} | {row.get('mean_fake_prob') or ''} "
        f"| {row.get('confidence') or ''} | {row.get('faces') or ''} | {notes} |\n"
    )


def _completed_detectors(log_jsonl: Path) -> set[str]:
    if not log_jsonl.exists():
        return set()
    done: set[str] = set()
    for line in log_jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("status") in TERMINAL and row.get("detector"):
            done.add(row["detector"])
    return done


# --------------------------------------------------------------------------- #
#  Runner                                                                     #
# --------------------------------------------------------------------------- #

def run_detector(
    detector_id: str, tier: str, media: Path, fmt: str,
    timeout: int, log_md: Path, log_jsonl: Path,
) -> None:
    verity = shutil.which("verity") or str(ROOT / ".venv" / "bin" / "verity")
    cmd = [
        verity, "run",
        "--media", str(media),
        "--detector", detector_id,
        "--format", fmt,
    ]
    log(f"=== {detector_id} (tier {tier}) ===")
    log("  " + " ".join(cmd))

    avail = _mem_available_mb()
    if avail is not None and avail < 3072:
        log(f"  WARN: only {avail} MB available; heavy detectors may be killed")

    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    monitor = Monitor(proc.pid, interval=1.0).start()
    start = time.time()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        monitor.stop()
        row = {
            "detector": detector_id, "tier": tier, "status": "failed",
            "peak_rss_mb": round(monitor.peak_rss_kb / 1024),
            "peak_vram_mb": monitor.peak_vram_mb,
            "wall_s": round(time.time() - start),
            "notes": f"timed out after {timeout}s",
        }
        _write_row(log_md, log_jsonl, row)
        log(f"  -> timed out after {timeout}s")
        return
    monitor.stop()
    wall_s = round(time.time() - start)
    stderr_tail = (stderr or "").strip().splitlines()[-3:]

    if proc.returncode == -9:
        status = "killed"
        notes = "SIGKILL (likely OOM)"
    elif proc.returncode == 0:
        result_candidates = list((WORKSPACE / "output").glob(f"*_{detector_id}/result.json"))
        if not result_candidates:
            status = "failed"
            notes = "exit 0 but no result.json (engine did not finish)"
        else:
            status = "ok"
            notes = ""
    else:
        status = "failed"
        notes = f"exit {proc.returncode}; stderr: {' | '.join(stderr_tail)}"[-300:]

    summary = _finding_summary(detector_id)
    timing = _result_timing(detector_id)
    if timing:
        notes = (notes + "; " + timing).strip("; ")

    row = {
        "detector": detector_id, "tier": tier, "status": status,
        "peak_rss_mb": round(monitor.peak_rss_kb / 1024),
        "peak_vram_mb": monitor.peak_vram_mb,
        "wall_s": wall_s,
        **summary,
        "notes": notes,
    }
    _write_row(log_md, log_jsonl, row)
    log(f"  -> {status} ({wall_s}s, peak RSS {row['peak_rss_mb']} MB, "
        f"peak VRAM {row['peak_vram_mb'] or 'n/a'} MB)")

    if status == "failed":
        (log_md.parent / "runs" / f"{detector_id}.log").parent.mkdir(parents=True, exist_ok=True)
        (log_md.parent / "runs" / f"{detector_id}.log").write_text(
            f"# {detector_id} run log\nstdout:\n{stdout}\nstderr:\n{stderr}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media", type=Path, required=True,
                        help="media file to test (required)")
    parser.add_argument("--tiers", default="L,M,H",
                        help="comma-separated tiers to run (L, M, H)")
    parser.add_argument("--detectors", default=None,
                        help="explicit comma-separated detector ids (overrides --tiers)")
    parser.add_argument("--format", default="html", choices=["html", "pdf"])
    parser.add_argument("--log-md", type=Path, default=ROOT / "docs" / "matrix_runs.md")
    parser.add_argument("--log-jsonl", type=Path, default=ROOT / "docs" / "matrix_runs.jsonl")
    parser.add_argument("--timeout", type=int, default=3900,
                        help="per-detector hard timeout in seconds (default 3900)")
    parser.add_argument("--resume", action="store_true",
                        help="skip detectors already logged with a terminal status")
    parser.add_argument("--list", action="store_true",
                        help="print the run order and exit")
    args = parser.parse_args()

    if args.detectors:
        order = [(det, "?") for det in args.detectors.split(",")]
    else:
        order = [
            (det, tier)
            for tier in TIER_ORDER if tier in [t.strip() for t in args.tiers.split(",")]
            for det in TIERS[tier]
        ]
    if args.list:
        for det, tier in order:
            print(f"{tier}\t{det}")
        return 0

    if not args.media.exists():
        print(f"[matrix] error: media not found: {args.media}", file=sys.stderr)
        return 2

    _ensure_logs(args.log_md, args.log_jsonl)
    completed = _completed_detectors(args.log_jsonl) if args.resume else set()

    log(f"media: {args.media}")
    log(f"detectors: {len(order)}  resume-skip: {len(completed)}")
    try:
        for detector_id, tier in order:
            if detector_id in completed:
                log(f"skip {detector_id}: already logged (--resume)")
                continue
            run_detector(detector_id, tier, args.media, args.format,
                         args.timeout, args.log_md, args.log_jsonl)
    except KeyboardInterrupt:
        log("interrupted by user")
        return 130
    log("matrix complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
