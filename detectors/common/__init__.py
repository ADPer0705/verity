"""Shared inference core for verity detectors.

Every detector under ``detectors/<id>/`` is a self-contained package with its
own ``requirements.txt``.  The heavy lifting (face detection, frame/clip
sampling, per-detector preprocessing, gradient-based explainability,
aggregation and reasoning) lives here so each detector's ``model.py`` only
needs to implement the architecture and a ``features()`` / ``classifier()``
contract.

Interface contract for a ported ``model.py``:

* ``build_model(weights_path, device)`` -> ``nn.Module`` (eval mode)
* the module exposes ``features(x) -> 4D/5D spatial tensor``
* the module exposes ``classifier(feats) -> logits``
* the module exposes ``forward_detailed(x) -> dict`` with at least
  ``{"logits", "probs"}`` and optionally ``{"attention_map"}`` (a normalized
  2D spatial map that overrides Grad-CAM when present)
"""

from .spec import DetectorSpec, load_spec  # noqa: F401
from .engine import run_detector  # noqa: F401
