"""VLAForge detector (Zhu, Miao, Zhang, Deng, Pang, CVPR 2026, "Unleashing
Vision-Language Semantics for Deepfake Video Detection").

Ported from ``model/VLAForge.py`` + ``model/ForgePerceiver.py`` + ``model/attn.py``
+ ``model/layer.py`` + the modified ``model/clip`` of github.com/mala-lab/VLAForge
(MIT); shared modules live in ``detectors/common/backbones`` (``clip_openai``,
``forensics_adapter``).  The released checkpoint (Google Drive) is the whole
``VLAForge`` state dict including the frozen CLIP ViT-L/14 (fp16), so no
CLIP download is needed; it is loaded strictly.

Inference path (``VLAForge.forward``):

1. CLIP ViT-L/14 features of the 224-resized face at layers 0/1/8/15 and the
   768-d CLIP image embedding (the *identity prior*).
2. ``ForgePerceiver`` (timm ViT-tiny trunk with 128 query tokens over the
   256x256 face) -> blending-boundary prediction ``xray_pred`` (Q x 16 x 16)
   and a per-head attention bias.
3. ``RecAttnClip``: the frozen CLIP blocks are re-run on
   ``[queries, CLS, patches]`` with the bias as attention mask (intra block
   after layer 10) -> projected, normalised query tokens and raw patch tokens.
4. **Identity-aware VLA score**: the prompts "This is a real/fake photo of id
   person." are encoded with the image embedding injected at the "id" slot;
   the (768-projected) patch tokens are matched to both prompts (``100 * cos``),
   the fake-channel map (16x16) gates ``xray_pred``; 1x1 conv, average pool,
   ``Linear(128, 2)``, softmax -> ``text_output``.
5. ``PostClipProcess`` on the query tokens -> softmax -> ``clip_cls_output``.
6. P(fake) = 0.5 * clip_cls[:, 1] + 0.5 * text_output[:, 1].

The boundary map ``masked_xray_post_process(xray_pred)`` (1 x 256 x 256) is
returned as the native attention map.  Input: 256x256 face crop with CLIP
mean/std (``config/test.yaml``: resolution 256, CLIP mean/std); trained on
FaceForensics++ c23 (DeepfakeBench dlib-aligned crops, no margin).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common.backbones.clip_openai import build_clip, tokenize_openai  # noqa: E402
from common.backbones.forensics_adapter import (Adapter, MaskPostXrayProcess, PostClipProcess,  # noqa: E402
                                                RecAttnClip)

NUM_QUERIES = 128
FUSION_MAP = {0: 0, 1: 1, 2: 8, 3: 15}
PROMPT_REAL = "This is a real photo of id person."
PROMPT_FAKE = "This is a fake photo of id person."
TOKENIZER_ID = "openai/clip-vit-large-patch14"


class VLAForgeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_model = build_clip()
        self.adapter = Adapter("vit_tiny_patch16_224", NUM_QUERIES, FUSION_MAP, 256, 128, 16, fuse=False)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, NUM_QUERIES, intra_layer=10, return_patches=True)
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=NUM_QUERIES)
        self.clip_post_process = PostClipProcess(NUM_QUERIES, 768)
        self.trainable_layer = nn.Linear(1024, 768)
        self.conv = nn.Conv2d(NUM_QUERIES, NUM_QUERIES, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(NUM_QUERIES, 2, bias=False)
        self.register_buffer("tokens_real", torch.zeros(1, 77, dtype=torch.long), persistent=False)
        self.register_buffer("tokens_fake", torch.zeros(1, 77, dtype=torch.long), persistent=False)
        self._tokens_ready = False

    def _prepare_tokens(self):
        if self._tokens_ready:
            return
        from transformers import CLIPTokenizer
        tok = CLIPTokenizer.from_pretrained(TOKENIZER_ID)
        dev = self.tokens_real.device
        self.tokens_real = tokenize_openai(tok, [PROMPT_REAL]).to(dev)
        self.tokens_fake = tokenize_openai(tok, [PROMPT_FAKE]).to(dev)
        self._tokens_ready = True

    def forward_detailed(self, x):
        self._prepare_tokens()
        images = x
        clip_images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        clip_features = self.clip_model.extract_features(clip_images, FUSION_MAP.values())
        img_features = self.clip_model.encode_image(clip_images)

        attn_biases, xray_preds = self.adapter(images, clip_features)
        patch_tokens, clip_output = self.rec_attn_clip(clip_features, attn_biases[-1], normalize=True)

        text_feature = []
        for i in range(images.shape[0]):
            id_emb = img_features[i].unsqueeze(0)
            id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
            tf_real = self.clip_model.encode_text(self.tokens_real, id_emb)
            tf_fake = self.clip_model.encode_text(self.tokens_fake, id_emb)
            tf_real = tf_real / tf_real.norm(dim=-1, keepdim=True)
            tf_fake = tf_fake / tf_fake.norm(dim=-1, keepdim=True)
            text_feature.append(torch.stack([tf_real, tf_fake]))
        text_feature = torch.stack(text_feature)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        text_feature = text_feature.squeeze(2)                               # (N, 2, 768)

        patch_tokens = self.trainable_layer(patch_tokens.float())            # (N, 256, 768)
        anomaly_map = 100.0 * torch.matmul(patch_tokens, text_feature.float().transpose(1, 2))  # (N, 256, 2)
        anomaly_fake = anomaly_map[:, :, 1].reshape(-1, 16, 16).unsqueeze(1)
        att_map = xray_preds[-1] * anomaly_fake
        pooled = self.avg_pool(self.conv(att_map)).reshape(-1, NUM_QUERIES)
        text_output = torch.softmax(self.classifier(pooled), dim=1)

        clip_cls_output = torch.softmax(self.clip_post_process(clip_output.float()), dim=1)
        prob = 0.5 * clip_cls_output[:, 1] + 0.5 * text_output[:, 1]
        prob = prob.clamp(1e-6, 1 - 1e-6)
        logits = torch.log(torch.stack([1 - prob, prob], dim=1))

        boundary = self.masked_xray_post_process(xray_preds[-1])             # (N, 1, 256, 256)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": clip_output,
            "attention_map": torch.sigmoid(boundary[:, 0]),
            "aux": {"clip_branch": clip_cls_output[:, 1], "vla_branch": text_output[:, 1]},
        }

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = VLAForgeDetector()
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
        own = model.state_dict()
        missing = sorted(set(own) - set(state))
        unexpected = sorted(set(state) - set(own))
        if missing or unexpected:
            raise RuntimeError(f"VLAForge checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
        model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
