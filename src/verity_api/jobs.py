"""In-process job manager.

Pipeline stages are long-running and blocking (ffprobe, face scans, detector
subprocesses), so every UI-triggered operation becomes a *job*: a background
thread that reports named steps and streams log lines. The UI polls or
subscribes over SSE.

Jobs live in memory only. That is the right trade for a single-workstation
forensics tool — a restart deliberately loses the queue, never the workspace,
because every stage writes its result to disk as it goes.
"""

from __future__ import annotations

import os
import threading
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

MAX_RETAINED_JOBS = 250
MAX_LOG_LINES = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    key: str
    label: str
    status: StepStatus = StepStatus.PENDING
    detail: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "status": self.status.value,
            "detail": self.detail,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class LogLine:
    seq: int
    ts: str
    level: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "level": self.level, "message": self.message}


class JobCancelled(Exception):
    """Raised inside a worker when the job has been asked to stop."""


@dataclass
class Job:
    id: str
    kind: str
    label: str
    media_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    steps: List[Step] = field(default_factory=list)
    logs: deque = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    result: Any = None
    error: Optional[str] = None
    traceback: Optional[str] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _seq: int = 0

    def to_dict(self, include_logs: bool = True, since: int = 0) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "media_ids": list(self.media_ids),
            "params": self.params,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log_seq": self._seq,
        }
        if include_logs:
            data["logs"] = [line.to_dict() for line in self.logs if line.seq > since]
        return data


class JobContext:
    """Handle passed to a worker function for reporting progress."""

    def __init__(self, job: Job, manager: "JobManager"):
        self._job = job
        self._manager = manager

    # -- identity ------------------------------------------------------ #

    @property
    def job(self) -> Job:
        return self._job

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled()

    def add_media_id(self, media_id: str) -> None:
        with self._manager.lock:
            if media_id not in self._job.media_ids:
                self._job.media_ids.append(media_id)

    def set_result(self, result: Any) -> None:
        with self._manager.lock:
            self._job.result = result

    # -- logging ------------------------------------------------------- #

    def log(self, message: str, level: str = "info") -> None:
        with self._manager.lock:
            self._job._seq += 1
            self._job.logs.append(
                LogLine(seq=self._job._seq, ts=_now(), level=level, message=str(message))
            )

    def warn(self, message: str) -> None:
        self.log(message, level="warn")

    def error(self, message: str) -> None:
        self.log(message, level="error")

    # -- steps --------------------------------------------------------- #

    def declare_steps(self, steps: List[tuple[str, str]]) -> None:
        with self._manager.lock:
            self._job.steps = [Step(key=k, label=l) for k, l in steps]

    def _find(self, key: str) -> Optional[Step]:
        for step in self._job.steps:
            if step.key == key:
                return step
        return None

    def start_step(self, key: str, label: Optional[str] = None) -> None:
        self.raise_if_cancelled()
        with self._manager.lock:
            step = self._find(key)
            if step is None:
                step = Step(key=key, label=label or key)
                self._job.steps.append(step)
            if label:
                step.label = label
            step.status = StepStatus.RUNNING
            step.started_at = _now()
        self.log(f"▸ {(label or (step.label if step else key))}")

    def finish_step(self, key: str, detail: str = "") -> None:
        with self._manager.lock:
            step = self._find(key)
            if step is not None:
                step.status = StepStatus.SUCCEEDED
                step.detail = detail
                step.finished_at = _now()
        if detail:
            self.log(f"  {detail}")

    def fail_step(self, key: str, detail: str = "") -> None:
        with self._manager.lock:
            step = self._find(key)
            if step is not None:
                step.status = StepStatus.FAILED
                step.detail = detail
                step.finished_at = _now()
        if detail:
            self.error(f"  {detail}")

    def skip_step(self, key: str, detail: str = "") -> None:
        with self._manager.lock:
            step = self._find(key)
            if step is not None:
                step.status = StepStatus.SKIPPED
                step.detail = detail
                step.finished_at = _now()


class JobConflict(RuntimeError):
    """A job is already pending or running for one of the requested cases."""


class JobManager:
    """Fixed-size thread pool plus a bounded registry of recent jobs.

    At most one unfinished job per case: two concurrent jobs against the same
    workspace entries would race on the same files. Pool size defaults to one
    worker (``VERITY_MAX_JOBS``) so detector runs queue instead of competing
    for GPU memory.
    """

    def __init__(self, max_workers: int | None = None):
        if max_workers is None:
            max_workers = max(1, int(os.environ.get("VERITY_MAX_JOBS", "1") or 1))
        self.lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._order: deque = deque(maxlen=MAX_RETAINED_JOBS)
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="verity-job")

    # -- lifecycle ----------------------------------------------------- #

    def submit(
        self,
        kind: str,
        label: str,
        worker: Callable[[JobContext], Any],
        params: Optional[Dict[str, Any]] = None,
        media_ids: Optional[List[str]] = None,
        steps: Optional[List[tuple[str, str]]] = None,
    ) -> Job:
        with self.lock:
            for media_id in media_ids or []:
                for existing in self._jobs.values():
                    if not existing.status.terminal and media_id in existing.media_ids:
                        raise JobConflict(
                            f"'{existing.label}' is still {existing.status.value} for this case "
                            f"(job {existing.id}). Wait for it to finish or cancel it first."
                        )
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            label=label,
            params=params or {},
            media_ids=list(media_ids or []),
        )
        if steps:
            job.steps = [Step(key=k, label=l) for k, l in steps]

        with self.lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()

        self._pool.submit(self._run, job, worker)
        return job

    def _run(self, job: Job, worker: Callable[[JobContext], Any]) -> None:
        ctx = JobContext(job, self)
        with self.lock:
            job.status = JobStatus.RUNNING
            job.started_at = _now()

        try:
            result = worker(ctx)
            with self.lock:
                if job.result is None:
                    job.result = result
                job.status = JobStatus.CANCELLED if ctx.cancelled else JobStatus.SUCCEEDED
        except JobCancelled:
            with self.lock:
                job.status = JobStatus.CANCELLED
            ctx.warn("Job cancelled.")
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the UI
            with self.lock:
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
            ctx.error(f"{type(exc).__name__}: {exc}")
            for step in job.steps:
                if step.status is StepStatus.RUNNING:
                    step.status = StepStatus.FAILED
                    step.finished_at = _now()
        finally:
            with self.lock:
                job.finished_at = _now()

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status.terminal:
            return False
        job.cancel_event.set()
        return True

    # -- queries ------------------------------------------------------- #

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50, media_id: Optional[str] = None) -> List[Job]:
        with self.lock:
            jobs = [self._jobs[i] for i in reversed(self._order) if i in self._jobs]
        if media_id:
            jobs = [j for j in jobs if media_id in j.media_ids]
        return jobs[:limit]

    def active_count(self) -> int:
        with self.lock:
            return sum(1 for j in self._jobs.values() if not j.status.terminal)

    def _evict_locked(self) -> None:
        live = set(self._order)
        for job_id in [i for i in self._jobs if i not in live]:
            del self._jobs[job_id]


#: Process-wide manager. Two workers: detector runs saturate the GPU/CPU, and
#: a second slot keeps light jobs (analyze, report) responsive behind a run.
manager = JobManager(max_workers=2)
