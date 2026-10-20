from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(pil: Image.Image, resolution: int,
                  mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Resize a PIL image to ``resolution`` and apply the DeepfakeBench
    ``(x - mean) / std`` normalization used by every detector config."""
    pil = pil.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    arr = np.array(pil).transpose(2, 0, 1).astype("float32") / 255.0
    arr = (arr - np.asarray(mean, dtype="float32")[:, None, None]) / \
        np.asarray(std, dtype="float32")[:, None, None]
    return torch.from_numpy(arr).unsqueeze(0)
