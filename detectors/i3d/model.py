"""I3D detector (DeepfakeBench port): 3D ResNet-50 (SlowFast ``i3d`` arch).

Ported from DeepfakeBench ``training/detectors/i3d_detector.py``; the
checkpoint (``I3D_8x8_R50.pth``) is the Kinetics-pretrained backbone. The
raw network output is a single sigmoid value, converted to 2-class logits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from common.slowfast.config.defaults import get_cfg  # noqa: E402
from common.slowfast.models.video_model_builder import ResNet as ResNetOri  # noqa: E402

config_text = """
TRAIN:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 16
  SAMPLING_RATE: 8
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3]
RESNET:
  ZERO_INIT_FINAL_BN: True
  WIDTH_PER_GROUP: 64
  NUM_GROUPS: 1
  DEPTH: 50
  TRANS_FUNC: bottleneck_transform
  STRIDE_1X1: False
  NUM_BLOCK_TEMP_KERNEL: [[3], [4], [6], [3]]
NONLOCAL:
  LOCATION: [[[]], [[]], [[]], [[]]]
  GROUP: [[1], [1], [1], [1]]
  INSTANTIATION: softmax
BN:
  USE_PRECISE_STATS: True
  NUM_BATCHES_PRECISE: 200
SOLVER:
  BASE_LR: 0.1
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
MODEL:
  NUM_CLASSES: 1
  ARCH: i3d
  MODEL_NAME: ResNet
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
TEST:
  ENABLE: True
  DATASET: kinetics
  BATCH_SIZE: 64
DATA_LOADER:
  NUM_WORKERS: 8
  PIN_MEMORY: True
NUM_GPUS: 8
NUM_SHARDS: 1
RNG_SEED: 0
OUTPUT_DIR: .
"""


def _prob_to_logits(p):
    p_flat = p.reshape(-1).clamp(1e-7, 1 - 1e-7)
    logits = torch.zeros(p_flat.shape[0], 2, device=p.device, dtype=p.dtype)
    logits[:, 1] = torch.log(p_flat / (1.0 - p_flat))
    return logits


class I3DDetector(nn.Module):
    def __init__(self, clip_size=16):
        super().__init__()
        cfg = get_cfg()
        cfg.merge_from_str(config_text)
        cfg.NUM_GPUS = 1
        cfg.TEST.BATCH_SIZE = 1
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.DATA.NUM_FRAMES = clip_size
        self.resnet = ResNetOri(cfg)

    def forward_detailed(self, x):
        inputs = [x]
        prob = self.resnet(inputs)
        logits = _prob_to_logits(prob)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": prob}


def build_model(weights_path, device, clip_size=16):
    model = I3DDetector(clip_size=clip_size)
    if weights_path is not None:
        pretrained_weights = torch.load(weights_path, map_location="cpu", encoding="latin1")
        modified_weights = {k.replace("resnet.", ""): v for k, v in pretrained_weights.items()}
        modified_weights["head.projection.weight"] = modified_weights["head.projection.weight"][:1, :]
        modified_weights["head.projection.bias"] = modified_weights["head.projection.bias"][:1]
        model.resnet.load_state_dict(modified_weights, strict=True)
    model.to(device)
    model.eval()
    return model
