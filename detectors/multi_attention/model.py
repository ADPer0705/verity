"""Multi-Attention detector (Zhao et al. 2021): EfficientNet-B4 backbone with
multi-attentional heads (attention maps, texture enhancement, bilinear pooling,
ensemble classifier).

Ported from DeepfakeBench ``training/detectors/multi_attention_detector.py``.
The AGDA module (kornia-based adversarial deepfaking) only runs inside the
training branch (``if not inference``) and is dropped here.  The feature loop
replicates the original's ``named_children`` traversal of the EfficientNet,
which captures ``final`` before the trailing swish/global pooling.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from efficientnet_pytorch import EfficientNet  # noqa: E402


class EfficientNetBackbone(nn.Module):
    def __init__(self, num_classes=2, inc=3):
        super().__init__()
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3,
                                                 stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, num_classes)


class AttentionMap(nn.Module):
    def __init__(self, in_channels, num_attention):
        super().__init__()
        self.register_buffer('mask', torch.zeros([1, 1, 24, 24]))
        self.mask[0, 0, 2:-2, 2:-2] = 1
        self.num_attentions = num_attention
        self.conv_extract = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, num_attention, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_attention)

    def forward(self, x):
        if self.num_attentions == 0:
            return torch.ones([x.shape[0], 1, 1, 1], device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x) + 1
        mask = F.interpolate(self.mask, (x.shape[2], x.shape[3]), mode='nearest')
        return x * mask


class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, attentions, norm=2):
        feature_size = features.size()[-2:]
        attention_size = attentions.size()[-2:]
        if feature_size != attention_size:
            attentions = F.interpolate(attentions, size=feature_size,
                                       mode='bilinear', align_corners=True)
        if len(features.shape) == 4:
            feature_matrix = torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix = torch.einsum('imjk,imnjk->imn', attentions, features)
        if norm == 1:
            w = torch.sum(attentions + 1e-8, dim=(2, 3)).unsqueeze(-1)
            feature_matrix /= w
        elif norm == 2:
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        return feature_matrix


class TextureEnhanceV1(nn.Module):
    def __init__(self, num_features, num_attentions):
        super().__init__()
        self.output_features = num_features * 4
        self.output_features_d = num_features
        self.conv0 = nn.Conv2d(num_features, num_features, 1)
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features * 2, num_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * num_features)
        self.conv3 = nn.Conv2d(num_features * 3, num_features, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(3 * num_features)
        self.conv_last = nn.Conv2d(num_features * 4, num_features * 4, 1)
        self.bn4 = nn.BatchNorm2d(4 * num_features)
        self.bn_last = nn.BatchNorm2d(num_features * 4)

    def forward(self, feature_maps, attention_maps=(1, 1)):
        B, N, H, W = feature_maps.shape
        if type(attention_maps) == tuple:
            attention_size = (int(H * attention_maps[0]), int(W * attention_maps[1]))
        else:
            attention_size = (attention_maps.shape[2], attention_maps.shape[3])
        feature_maps_d = F.adaptive_avg_pool2d(feature_maps, attention_size)
        feature_maps = feature_maps - F.interpolate(feature_maps_d,
                                                    (feature_maps.shape[2], feature_maps.shape[3]),
                                                    mode='nearest')
        feature_maps0 = self.conv0(feature_maps)
        feature_maps1 = self.conv1(F.relu(self.bn1(feature_maps0), inplace=True))
        feature_maps1_ = torch.cat([feature_maps0, feature_maps1], dim=1)
        feature_maps2 = self.conv2(F.relu(self.bn2(feature_maps1_), inplace=True))
        feature_maps2_ = torch.cat([feature_maps1_, feature_maps2], dim=1)
        feature_maps3 = self.conv3(F.relu(self.bn3(feature_maps2_), inplace=True))
        feature_maps3_ = torch.cat([feature_maps2_, feature_maps3], dim=1)
        feature_maps = self.bn_last(self.conv_last(F.relu(self.bn4(feature_maps3_), inplace=True)))
        return feature_maps, feature_maps_d


class MultiAttentionDetector(nn.Module):
    def __init__(self, num_classes=2, inc=3, mid_dim=256, num_attentions=4,
                 feature_layer="b2", attention_layer="b5", resolution=256,
                 dropout_rate=0.25):
        super().__init__()
        self.block_layer = {"b1": 1, "b2": 5, "b3": 9, "b4": 15, "b5": 21,
                            "b6": 29, "b7": 31}
        self.backbone = EfficientNetBackbone(num_classes=num_classes, inc=inc)

        with torch.no_grad():
            layer_outputs = self.features({"image": torch.zeros(1, 3, resolution, resolution)})

        self.feature_layer = feature_layer
        self.attention_layer = attention_layer
        self.num_classes = num_classes
        self.num_shallow_features = layer_outputs[self.feature_layer].shape[1]
        self.num_attention_features = layer_outputs[self.attention_layer].shape[1]
        self.num_final_features = layer_outputs["final"].shape[1]
        self.num_attentions = num_attentions

        self.attention_generation = AttentionMap(self.num_attention_features, self.num_attentions)
        self.attention_pooling = AttentionPooling()
        self.texture_enhance = TextureEnhanceV1(self.num_shallow_features, self.num_attentions)
        self.num_enhanced_features = self.texture_enhance.output_features
        self.projection_local = nn.Sequential(
            nn.Linear(self.num_attentions * self.num_enhanced_features, mid_dim),
            nn.Hardswish(),
            nn.Linear(mid_dim, mid_dim),
            nn.Hardswish())
        self.projection_final = nn.Sequential(nn.Linear(self.num_final_features, mid_dim),
                                              nn.Hardswish())
        self.ensemble_classifier_fc = nn.Sequential(nn.Linear(mid_dim * 2, mid_dim),
                                                    nn.Hardswish(),
                                                    nn.Linear(mid_dim, self.num_classes))
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def features(self, data_dict):
        x = data_dict["image"]
        layer_output = {}
        for name, module in self.backbone.efficientnet.named_children():
            if name == "_avg_pooling":
                layer_output["final"] = x
                break
            elif name != "_blocks":
                x = module(x)
            else:
                for i in range(len(module)):
                    x = module[i](x)
                    if i == self.block_layer["b1"]:
                        layer_output["b1"] = x
                    elif i == self.block_layer["b2"]:
                        layer_output["b2"] = x
                    elif i == self.block_layer["b3"]:
                        layer_output["b3"] = x
                    elif i == self.block_layer["b4"]:
                        layer_output["b4"] = x
                    elif i == self.block_layer["b5"]:
                        layer_output["b5"] = x
                    elif i == self.block_layer["b6"]:
                        layer_output["b6"] = x
                    elif i == self.block_layer["b7"]:
                        layer_output["b7"] = x

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        layer_output["logit"] = self.backbone.last_layer(x)
        return layer_output

    def forward_detailed(self, x):
        layer_output = self.features({"image": x})
        raw_attentions = layer_output[self.attention_layer]
        attention_maps = self.attention_generation(raw_attentions)

        shallow_features = layer_output[self.feature_layer]
        enhanced_features, feature_maps_d = self.texture_enhance(shallow_features, attention_maps)
        textural_feature_matrix_p = self.attention_pooling(enhanced_features, attention_maps)
        B, M, N = textural_feature_matrix_p.size()
        feature_matrix = self.dropout(textural_feature_matrix_p).view(B, -1)
        feature_matrix = self.projection_local(feature_matrix)

        final = layer_output["final"]
        attention_maps2 = attention_maps.sum(dim=1, keepdim=True)
        final = self.attention_pooling(final, attention_maps2, norm=1).squeeze(1)
        final = self.projection_final(final)
        final = F.hardswish(final)

        feature_matrix = torch.cat((feature_matrix, final), dim=1)
        logits = self.ensemble_classifier_fc(feature_matrix)
        return {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1),
            "feat": layer_output["final"],
            "attention_map": attention_maps.mean(dim=1, keepdim=True),
        }

    def forward(self, x):
        return self.forward_detailed(x)["logits"]


def build_model(weights_path, device):
    model = MultiAttentionDetector(num_classes=2, inc=3, mid_dim=256, num_attentions=4,
                                   feature_layer="b2", attention_layer="b5",
                                   resolution=256, dropout_rate=0.25)
    if weights_path is not None:
        state = torch.load(weights_path, map_location="cpu")
        if all(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model
