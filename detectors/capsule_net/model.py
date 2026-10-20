"""Capsule-forensics detector (Nguyen et al. 2019).

Ported from DeepfakeBench ``training/detectors/capsule_net_detector.py``.
A VGG19 stem (first 19 feature layers) feeds 10 parallel capsule feature
extractors whose squashed outputs are routed to 2 output capsules.  The
checkpoint (``capsule_best.pth``) is the full detector state dict (keys
``vgg_ext.*``, ``fea_ext.*``, ``routing_stats.*``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torchvision.models import vgg19  # noqa: E402


class StatsNet(nn.Module):
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        mean = torch.mean(x, 2)
        std = torch.std(x, 2)
        return torch.stack((mean, std), dim=1)


class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class VggExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=None)
        self.vgg_1 = nn.Sequential(*list(vgg.features.children())[0:19])

    def forward(self, x):
        return self.vgg_1(x)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.NO_CAPS = 10
        self.capsules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                StatsNet(),
                nn.Conv1d(2, 8, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(8),
                nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(1),
                View(-1, 8),
            )
            for _ in range(self.NO_CAPS)
        ])

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        output = torch.stack(outputs, dim=-1)
        return self.squash(output, dim=-1)


class RoutingLayer(nn.Module):
    def __init__(self, num_input_capsules, num_output_capsules, data_in,
                 data_out, num_iterations):
        super().__init__()
        self.num_iterations = num_iterations
        self.route_weights = nn.Parameter(
            torch.randn(num_output_capsules, num_input_capsules, data_out, data_in))

    def squash(self, tensor, dim):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x, random=False, dropout=0.0):
        x = x.transpose(2, 1)
        route_weights = self.route_weights
        priors = route_weights[:, None, :, :, :] @ x[None, :, :, :, None]
        priors = priors.transpose(1, 0)
        if dropout > 0.0:
            drop = torch.FloatTensor(*priors.size()).bernoulli(1.0 - dropout).to(priors.device)
            priors = priors * drop
        logits = torch.zeros(*priors.size()).to(priors.device)
        for i in range(self.num_iterations):
            probs = F.softmax(logits, dim=2)
            outputs = self.squash((probs * priors).sum(dim=2, keepdim=True), dim=3)
            if i != self.num_iterations - 1:
                logits = logits + priors * outputs
        outputs = outputs.squeeze()
        if len(outputs.shape) == 3:
            outputs = outputs.transpose(2, 1).contiguous()
        else:
            outputs = outputs.unsqueeze_(dim=0).transpose(2, 1).contiguous()
        return outputs


class CapsuleNetDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.vgg_ext = VggExtractor()
        self.fea_ext = FeatureExtractor()
        self.NO_CAPS = 10
        self.routing_stats = RoutingLayer(num_input_capsules=self.NO_CAPS,
                                          num_output_capsules=self.num_classes,
                                          data_in=8, data_out=4, num_iterations=2)

    def features(self, x):
        x = self.vgg_ext(x)
        return self.fea_ext(x)

    def classifier(self, feats):
        z = self.routing_stats(feats, random=False, dropout=0.0)
        classes = F.softmax(z, dim=-1)
        pred = classes.mean(dim=1)
        return classes, pred

    def forward_detailed(self, x):
        feats = self.features(x)
        classes, pred = self.classifier(feats)
        return {"logits": pred, "probs": torch.softmax(pred, dim=1), "feat": feats}

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = CapsuleNetDetector(num_classes=2)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
