"""Checkpoint management: manifest, download, verification, hub bundle.

See ``detectors/weights_manifest.yaml`` for the schema. Everything here is
deterministic and offline-safe: nothing is accepted without a SHA-256 match,
and a file that already verifies is never re-downloaded.
"""
from verity.weights.manifest import (  # noqa: F401
    FileSpec, Manifest, Target, load_manifest,
)
from verity.weights.fetch import (  # noqa: F401
    PullResult, pull, sha256_of, status, verify,
)
from verity.weights.bundle import build_bundle  # noqa: F401
