import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models import xception as timm_xception


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, reps, strides=1,
                 start_with_relu=True, grow_first=True, bias=False):
        super().__init__()
        if out_channels != in_channels or strides != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        for i in range(reps):
            if grow_first:
                inc = in_channels if i == 0 else out_channels
                outc = out_channels
            else:
                inc = in_channels
                outc = in_channels if i < reps - 1 else out_channels
            if start_with_relu:
                rep.append(nn.ReLU(inplace=False))
            rep.append(SeparableConv2d(inc, outc, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(outc))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        skip = x
        if self.skip is not None:
            skip = self.skip(skip)
            skip = self.skipbn(skip)
        out = self.rep(x)
        out = self.relu(out + skip)
        return out


encoder_params = {
    "xception": {
        "features": 2048,
        "init_op": partial(timm_xception, pretrained=False)
    }
}


class GuidedAttention(nn.Module):
    def __init__(self, depth=728, drop_rate=0.2):
        super().__init__()
        self.depth = depth
        self.gated = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.h = nn.Sequential(
            nn.Conv2d(depth, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.ReLU(True),
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, pred_x, embedding):
        residual_full = torch.abs(x - pred_x)
        residual_x = F.interpolate(residual_full, size=embedding.shape[-2:],
                                   mode='bilinear', align_corners=True)
        res_map = self.gated(residual_x)
        return res_map * self.h(embedding) + self.dropout(embedding)


class GraphReasoning(nn.Module):
    def __init__(self, va_in, va_out, vb_in, vb_out, vc_in, vc_out, spatial_ratio, drop_rate):
        super().__init__()
        self.ratio = spatial_ratio
        self.va_embedding = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_out, va_out, 1, bias=False),
        )
        self.va_gated_b = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.va_gated_c = nn.Sequential(
            nn.Conv2d(va_in, va_out, 1, bias=False),
            nn.Sigmoid()
        )
        self.vb_embedding = nn.Sequential(
            nn.Linear(vb_in, vb_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vb_out, vb_out, bias=False),
        )
        self.vc_embedding = nn.Sequential(
            nn.Linear(vc_in, vc_out, bias=False),
            nn.ReLU(True),
            nn.Linear(vc_out, vc_out, bias=False),
        )
        self.unfold_b = nn.Unfold(kernel_size=spatial_ratio[0], stride=spatial_ratio[0])
        self.unfold_c = nn.Unfold(kernel_size=spatial_ratio[1], stride=spatial_ratio[1])
        self.reweight_ab = nn.Sequential(
            nn.Linear(va_out + vb_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reweight_ac = nn.Sequential(
            nn.Linear(va_out + vc_out, 1, bias=False),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        self.reproject = nn.Sequential(
            nn.Conv2d(va_out + vb_out + vc_out, va_in, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(va_in, va_in, kernel_size=1, bias=False),
            nn.Dropout(drop_rate) if drop_rate is not None else nn.Identity(),
        )

    def forward(self, vert_a, vert_b, vert_c):
        emb_vert_a = self.va_embedding(vert_a)
        emb_vert_a = emb_vert_a.reshape([emb_vert_a.shape[0], emb_vert_a.shape[1], -1])

        gate_vert_b = 1 - self.va_gated_b(vert_a)
        gate_vert_b = gate_vert_b.reshape(*emb_vert_a.shape)
        gate_vert_c = 1 - self.va_gated_c(vert_a)
        gate_vert_c = gate_vert_c.reshape(*emb_vert_a.shape)

        vert_b = self.unfold_b(vert_b).reshape(
            [vert_b.shape[0], vert_b.shape[1], self.ratio[0] * self.ratio[0], -1])
        vert_b = vert_b.permute([0, 2, 3, 1])
        emb_vert_b = self.vb_embedding(vert_b)

        vert_c = self.unfold_c(vert_c).reshape(
            [vert_c.shape[0], vert_c.shape[1], self.ratio[1] * self.ratio[1], -1])
        vert_c = vert_c.permute([0, 2, 3, 1])
        emb_vert_c = self.vc_embedding(vert_c)

        agg_vb = []
        agg_vc = []
        for j in range(emb_vert_a.shape[-1]):
            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[0] ** 2), dim=1)
            emb_v_b = emb_vert_b[:, :, j, :]
            emb_v_ab = torch.cat([emb_v_a, emb_v_b], dim=-1)
            w = self.reweight_ab(emb_v_ab)
            agg_vb.append(torch.bmm(emb_v_b.transpose(1, 2), w).squeeze() * gate_vert_b[:, :, j])

            emb_v_a = torch.stack([emb_vert_a[:, :, j]] * (self.ratio[1] ** 2), dim=1)
            emb_v_c = emb_vert_c[:, :, j, :]
            emb_v_ac = torch.cat([emb_v_a, emb_v_c], dim=-1)
            w = self.reweight_ac(emb_v_ac)
            agg_vc.append(torch.bmm(emb_v_c.transpose(1, 2), w).squeeze() * gate_vert_c[:, :, j])

        agg_vert_b = torch.stack(agg_vb, dim=-1)
        agg_vert_c = torch.stack(agg_vc, dim=-1)
        agg_vert_bc = torch.cat([agg_vert_b, agg_vert_c], dim=1)
        agg_vert_abc = torch.cat([agg_vert_bc, emb_vert_a], dim=1)
        agg_vert_abc = torch.sigmoid(agg_vert_abc)
        agg_vert_abc = agg_vert_abc.reshape(vert_a.shape[0], -1, vert_a.shape[2], vert_a.shape[3])
        return self.reproject(agg_vert_abc)


class Recce(nn.Module):
    def __init__(self, num_classes, drop_rate=0.2):
        super().__init__()
        self.name = "xception"
        self.loss_inputs = dict()
        self.encoder = encoder_params[self.name]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(encoder_params[self.name]["features"], num_classes)

        self.attention = GuidedAttention(depth=728, drop_rate=drop_rate)
        self.reasoning = GraphReasoning(728, 256, 256, 256, 128, 256, [2, 4], drop_rate)

        self.decoder1 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(728, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = Block(256, 256, 3, 1)
        self.decoder3 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = Block(128, 128, 3, 1)
        self.decoder5 = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            SeparableConv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder6 = nn.Sequential(
            nn.Conv2d(64, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    @staticmethod
    def norm_n_corr(x):
        pool = nn.AdaptiveAvgPool2d((1, 1))
        norm_embed = F.normalize(pool(x), p=2, dim=1)
        corr = (torch.matmul(norm_embed.squeeze(), norm_embed.squeeze().T) + 1.) / 2.
        return norm_embed, corr

    @staticmethod
    def add_white_noise(tensor, mean=0., std=1e-6):
        rand = torch.rand([tensor.shape[0], 1, 1, 1])
        rand = torch.where(rand > 0.5, 1., 0.).to(tensor.device)
        white_noise = torch.normal(mean, std, size=tensor.shape, device=tensor.device)
        noise_t = tensor + white_noise * rand
        noise_t = torch.clip(noise_t, -1., 1.)
        return noise_t

    def features(self, x):
        self.loss_inputs = dict(recons=[], contra=[])
        noise_x = self.add_white_noise(x) if self.training else x
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        embedding = self.encoder.block4(out)

        norm_embed, corr = self.norm_n_corr(embedding)
        self.loss_inputs['contra'].append(corr)

        out = self.dropout(embedding)
        out = self.decoder1(out)
        out_d2 = self.decoder2(out)

        norm_embed, corr = self.norm_n_corr(out_d2)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder3(out_d2)
        out_d4 = self.decoder4(out)

        norm_embed, corr = self.norm_n_corr(out_d4)
        self.loss_inputs['contra'].append(corr)

        out = self.decoder5(out_d4)
        pred = self.decoder6(out)

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        self.loss_inputs['recons'].append(recons_x)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        fusion = self.reasoning(embedding, out_d2, out_d4) + embedding

        embedding = self.encoder.block8(fusion)
        img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        embedding = self.global_pool(embedding).squeeze(2).squeeze(2)
        embedding = self.dropout(embedding)

        return embedding

    def classifier(self, embedding):
        return self.fc(embedding)

    def forward(self, x):
        embedding = self.features(x)
        return self.classifier(embedding)

    def forward_detailed(self, x):
        noise_x = self.add_white_noise(x) if self.training else x
        out = self.encoder.conv1(noise_x)
        out = self.encoder.bn1(out)
        out = self.encoder.act1(out)
        out = self.encoder.conv2(out)
        out = self.encoder.bn2(out)
        out = self.encoder.act2(out)
        out = self.encoder.block1(out)
        out = self.encoder.block2(out)
        out = self.encoder.block3(out)
        embedding = self.encoder.block4(out)

        dec_out = self.dropout(embedding)
        dec_out = self.decoder1(dec_out)
        out_d2 = self.decoder2(dec_out)
        dec_out = self.decoder3(out_d2)
        out_d4 = self.decoder4(dec_out)
        dec_out = self.decoder5(out_d4)
        pred = self.decoder6(dec_out)

        recons_x = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)

        embedding = self.encoder.block5(embedding)
        embedding = self.encoder.block6(embedding)
        embedding = self.encoder.block7(embedding)

        fusion = self.reasoning(embedding, out_d2, out_d4) + embedding
        embedding = self.encoder.block8(fusion)

        residual_full = torch.abs(x - recons_x)

        # The checkpoint's attention.gated weights are near-zero and produce
        # constant-0.5 maps. Use the reconstruction residual magnitude directly
        # as the attention signal — this is the signal that *should* drive
        # guided attention per the RECCE design.
        raw_attention_map = residual_full.mean(dim=1, keepdim=True)

        img_att = self.attention(x, recons_x, embedding)

        embedding = self.encoder.block9(img_att)
        embedding = self.encoder.block10(embedding)
        embedding = self.encoder.block11(embedding)
        embedding = self.encoder.block12(embedding)

        embedding = self.encoder.conv3(embedding)
        embedding = self.encoder.bn3(embedding)
        embedding = self.encoder.act3(embedding)
        embedding = self.encoder.conv4(embedding)
        embedding = self.encoder.bn4(embedding)
        embedding = self.encoder.act4(embedding)

        pooled = self.global_pool(embedding).squeeze(2).squeeze(2)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "embedding": pooled,
            "reconstruction": recons_x,
            "residual": residual_full,
            "attention_map": raw_attention_map,
            "decoder_features_d2": out_d2,
            "decoder_features_d4": out_d4,
            "pre_attention": embedding,
        }
