"""TimeSformer detector (Yan et al. 2023), TimeSformer-Base + CLS token.

Ported from DeepfakeBench ``training/detectors/timesformer_detector.py``; the
detector is ``facebook/timesformer-base-finetuned-k400`` (768-d) with the CLS
token fed to a 2-way linear head. No detector checkpoint is released, so
without a ``weights_path`` the head is randomly initialized.

Input is a (B, 3, T, H, W) clip; it is permuted to the (B, T, 3, H, W) layout
the HF TimeSformer model expects (the DeepfakeBench loader already provides
(B, T, C, H, W)). T must equal the checkpoint's ``num_frames`` (8) at 224x224.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TimeSformerDetector(nn.Module):
    def __init__(self, num_classes=2, model_name="facebook/timesformer-base-finetuned-k400"):
        super().__init__()
        from transformers import TimesformerModel

        self.backbone = TimesformerModel.from_pretrained(model_name)
        self.head = nn.Linear(768, num_classes)

    def features(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B,3,T,H,W) -> (B,T,3,H,W)
        outputs = self.backbone(x, output_hidden_states=True)
        return outputs[0][:, 0]

    def classifier(self, feats):
        return self.head(feats)

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = TimeSformerDetector(num_classes=2, model_name="facebook/timesformer-base-finetuned-k400")
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
