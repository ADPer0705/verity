"""MesoNet (MesoInception4) detector (Afchar et al. 2018).

Ported from DeepfakeBench ``training/detectors/meso4Inception_detector.py``
(backbone ``networks/mesonet.py``).  The checkpoint (``meso4Incep_best.pth``)
is the full detector state dict (keys ``backbone.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


class MesoInception4(nn.Module):
    def __init__(self, num_classes=2, inc=3):
        super().__init__()
        self.num_classes = num_classes
        self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
        self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
        self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
        self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption1_bn = nn.BatchNorm2d(11)
        self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
        self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
        self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
        self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
        self.Incption2_bn = nn.BatchNorm2d(12)
        self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, self.num_classes)

    def InceptionLayer1(self, x):
        x1 = self.Incption1_conv1(x)
        x2 = self.Incption1_conv2_2(self.Incption1_conv2_1(x))
        x3 = self.Incption1_conv3_2(self.Incption1_conv3_1(x))
        x4 = self.Incption1_conv4_2(self.Incption1_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption1_bn(y)
        return self.maxpooling1(y)

    def InceptionLayer2(self, x):
        x1 = self.Incption2_conv1(x)
        x2 = self.Incption2_conv2_2(self.Incption2_conv2_1(x))
        x3 = self.Incption2_conv3_2(self.Incption2_conv3_1(x))
        x4 = self.Incption2_conv4_2(self.Incption2_conv4_1(x))
        y = torch.cat((x1, x2, x3, x4), 1)
        y = self.Incption2_bn(y)
        return self.maxpooling1(y)

    def features(self, x):
        x = self.InceptionLayer1(x)
        x = self.InceptionLayer2(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling2(x)
        return x.view(x.size(0), -1)

    def classifier(self, feats):
        out = self.dropout(feats)
        out = self.fc1(out)
        out = self.leakyrelu(out)
        out = self.dropout(out)
        return self.fc2(out)

    def forward(self, x):
        return self.classifier(self.features(x))

    def forward_detailed(self, x):
        feats = self.features(x)
        logits = self.classifier(feats)
        return {"logits": logits, "probs": torch.softmax(logits, dim=1), "feat": feats}


def build_model(weights_path, device):
    model = MesoInception4(num_classes=2, inc=3)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
