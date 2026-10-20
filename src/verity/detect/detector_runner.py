from __future__ import annotations
import json
import os
import shutil
import shlex
import subprocess
import tempfile
import importlib.util
from pathlib import Path
from typing import Optional

from verity.detect.detector_registry import DetectorRegistry
from verity.detect.base_detector import DetectorDefinition
from verity.models.findings import Finding


class DetectorRunner:
    """Runs detectors in their isolated uv-managed environments.

    Each detector is its own uv project (``detectors/<id>/pyproject.toml``).
    Execution delegates to ``scripts/verity.py`` (or the ``verity-run``
    executable in the container), which provisions the environment on demand
    (``uv sync``) and launches ``inference.py`` with ``INPUT_DIR`` /
    ``OUTPUT_DIR`` set. Environments are cached under ``$VERITY_DETECTOR_ENVS`` and
    weights under ``$VERITY_WEIGHTS_ROOT``.
    """

    def __init__(self, workspace: Path, registry: DetectorRegistry):
        self.workspace = workspace
        self.registry = registry
        self.output_dir = workspace / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = int(os.environ.get("VERITY_DETECTOR_TIMEOUT", "3600"))

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, media_id: str, detector_id: str) -> Finding:
        definition = self.registry.get(detector_id)
        if definition is None:
            raise ValueError(f"Unknown detector: {detector_id}")

        media_path = self._resolve_media(media_id)
        artifact_dir = self.workspace / "artifacts" / media_id
        output_path = self.output_dir / f"{media_id}_{detector_id}"
        output_path.mkdir(parents=True, exist_ok=True)

        input_dir = self._prepare_input_dir(media_path, artifact_dir)
        try:
            raw = self._run_detector(definition, input_dir, output_path)
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

        finding = self._parse_output(definition, raw, output_path)
        finding.tier = definition.weights.tier
        finding_path = output_path / "finding.json"
        finding_path.write_text(finding.model_dump_json(indent=2))
        return finding

    def setup(self, detector_ids: Optional[list[str]] = None, force: bool = False,
              concurrency: Optional[int] = None) -> int:
        """Provision detector environments (shells out to ``verity.py setup``)."""
        args = ["setup"]
        if detector_ids:
            args.append(f"--detector={','.join(detector_ids)}")
        else:
            args.append("--all")
        if force:
            args.append("--force")
        if concurrency:
            args.append(f"--concurrency={int(concurrency)}")
        return self._invoke_verity(args)

    # ------------------------------------------------------------------ #
    #  Execution                                                          #
    # ------------------------------------------------------------------ #

    def _run_detector(
        self, definition: DetectorDefinition,
        input_dir: Path, output_path: Path,
    ) -> dict:
        args = [
            "run", definition.detector_id,
            str(input_dir.resolve()), str(output_path.resolve()),
        ]
        result = self._invoke_verity(args, capture=True, cwd=str(self.registry.detectors_dir))

        (output_path / "stdout.log").write_text(result.stdout[-20000:] if result.stdout else "")
        (output_path / "stderr.log").write_text(result.stderr[-20000:] if result.stderr else "")

        result_json = output_path / "result.json"
        if result_json.exists():
            return json.loads(result_json.read_text())

        return {
            "prediction": 0.5,
            "confidence": 0.0,
            "reasoning": (
                f"Detector did not produce result.json "
                f"(exit code {result.returncode}). "
                f"stderr: {(result.stderr or '')[-500:]}"
            ),
        }

    def _invoke_verity(
        self, args: list[str],
        capture: bool = False, cwd: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        cmd = [self._verity_runner(), *args]
        env = dict(os.environ)
        env.setdefault("VERITY_HW_PROFILE", "auto")
        try:
            return subprocess.run(
                cmd, env=env, cwd=cwd,
                capture_output=capture, text=True, timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Detector environment command timed out after {self._timeout}s: "
                + " ".join(shlex.quote(c) for c in cmd)
            )

    def _verity_runner(self) -> str:
        override = os.environ.get("VERITY_RUNNER")
        if override:
            return override
        repo_script = Path(__file__).resolve().parents[3] / "scripts" / "verity.py"
        if repo_script.exists():
            return str(repo_script)
        on_path = shutil.which("verity-run")
        if on_path:
            return on_path
        raise RuntimeError(
            "Cannot locate the detector runtime (scripts/verity.py). Set "
            "VERITY_RUNNER to the script path, or use the container image."
        )

    # ------------------------------------------------------------------ #
    #  Input staging / output parsing                                     #
    # ------------------------------------------------------------------ #

    def _resolve_media(self, media_id: str) -> Path:
        media_dir = self.workspace / "media"
        for p in sorted(media_dir.iterdir()):
            if p.stem == media_id and p.is_file():
                return p
        raise FileNotFoundError(f"No media for id: {media_id}")

    def _prepare_input_dir(self, media_path: Path, artifact_dir: Path) -> Path:
        tmp = Path(tempfile.mkdtemp(prefix="detector_input_"))
        input_media = tmp / "media"
        input_artifacts = tmp / "artifacts"
        input_media.mkdir(parents=True, exist_ok=True)
        input_artifacts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(media_path, input_media / media_path.name)
        if artifact_dir.exists():
            for f in sorted(artifact_dir.rglob("*")):
                if f.is_file():
                    rel = f.relative_to(artifact_dir)
                    dest = input_artifacts / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
        return tmp

    def _parse_output(
        self, definition: DetectorDefinition,
        raw: dict, output_path: Path,
    ) -> Finding:
        parser_path = self.registry.detectors_dir / definition.detector_id / "output_parser.py"
        if parser_path.exists():
            spec = importlib.util.spec_from_file_location(
                f"{definition.detector_id}_parser", parser_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "parse"):
                return module.parse(raw, output_path, definition.version)

        from verity.detect.output_parser import parse as default_parse
        return default_parse(raw, output_path, definition.version, definition.detector_id)
