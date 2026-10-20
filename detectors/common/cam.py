from __future__ import annotations

import numpy as np
import torch


def grad_cam(model: torch.nn.Module, x: torch.Tensor,
             target_class: int = 1) -> np.ndarray | None:
    """Gradient-weighted class activation map for a detector.

    Contract: ``model.forward_detailed(x)`` returns a dict containing at least
    ``logits`` and a spatial ``feat`` tensor (4D ``(B, C, H, W)`` for image
    detectors, 5D ``(B, C, T, H, W)`` for video detectors).  The CAM is built
    by gradient-weighting ``feat`` w.r.t. the logit of ``target_class`` and is
    returned normalized to ``[0, 1]`` as ``(H, W)`` (video clips are pooled
    over time).

    Returns ``None`` when gradients cannot be obtained (e.g. non-differentiable
    preprocessing inside the model) so callers can degrade gracefully.
    """
    if x.grad is not None:
        x.grad = None
    x.requires_grad_(True)
    try:
        with torch.enable_grad():
            out = model.forward_detailed(x)
            logits = out["logits"]
            feat = out["feat"]
            if not isinstance(feat, torch.Tensor) or feat.ndim not in (4, 5):
                return None
            # target class index; fall back to argmax if out of range
            if logits.ndim == 2 and logits.size(1) > target_class:
                cls = target_class
            else:
                cls = int(logits.reshape(-1).argmax().item())
            logit = logits[0, cls]
            grad = torch.autograd.grad(logit, feat, retain_graph=True)[0]
        weights = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1))
        cam = cam[0]
        if cam.ndim == 3:  # (T, H, W) -> mean over time
            cam = cam.mean(dim=0)
        cam = cam.detach().cpu().numpy()
        mx, mn = cam.max(), cam.min()
        if not np.isfinite(mx) or mx - mn < 1e-8:
            return None
        return ((cam - mn) / (mx - mn + 1e-8)).astype("float32")
    except Exception:
        return None


def upscale_to_pil(cam: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear-upscale a (H, W) CAM to (height, width) in [0, 1]."""
    from PIL import Image

    if cam.ndim != 2:
        return cam
    img = Image.fromarray((cam * 255).astype("uint8"))
    return np.asarray(img.resize((width, height), Image.BILINEAR)).astype("float32") / 255.0
