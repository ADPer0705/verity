# LAA-Net weights

| Property | Value |
|---|---|
| File | `laa_net_sbi.pth` (default) |
| Size | 435,494,267 bytes |
| sha256 | `fde726610e778045f65c3fbe839bc463c4d3a9970f72ca109d5911917305c991` |
| Upstream name | `PoseEfficientNet_EFN_hm10_EFPN_NoBasedCLS_Focal_C3_256Cst100_8SBI_SAM(Adam)_ADV_Era1_OutSigmoid_1e7_boost500_UnFZ_model_best.pth` |
| Alternate | `laa_net_bi.pth` = `PoseEfficientNet_EFN_hm100_EFPN_NoBasedCLS_Focal_C3_256Cstency100_32BI_SAM(Adam)_ADV_Erasing1_OutSigmoid_model_best.pth` (435,494,203 bytes, sha256 `b1279335e0c5ad87e1bbf87de20deb9310621ae99bb88a9e974527b0d786d26b`) |
| Origin | Dropbox folder linked from the README of 10Ring/LAA-Net ("LAA-Net Pre-trained Models") |
| Training | FF++ c23 real faces + self-blended images (SBI) / blended images (BI) |
| License | SnT academic license (University of Luxembourg): academic research use only; no redistribution outside that scope |

The files are **not tracked in git** and are never mirrored by Verity.
Download the folder as a zip from

```
https://www.dropbox.com/scl/fo/dzmldaytujdeuky69d5x1/AIJrH2mit1hxnl1qzavM3vk?rlkey=nzzliincrfwejw2yr0ovldru1&dl=1
```

(`verity weights pull` does this and extracts the member), unzip, and
rename the SBI checkpoint to `laa_net_sbi.pth` in this directory.

Checkpoint layout: `{"epoch", "state_dict", "optimizer"}`; `state_dict`
has 803 tensors with EfficientNet keys at top level (`_conv_stem`,
`_blocks.*`, ...), the decoder (`deconv_1..4`, `inception_block`) and the
`hm` / `cls` / `cstency` heads. The port prefixes the trunk keys with
`backbone.` and loads strictly.

```bibtex
@InProceedings{Nguyen_2024_CVPR,
  author    = {Nguyen, Dat and Mejri, Nesryne and Singh, Inder Pal and Kuleshova, Polina and Astrid, Marcella and Kacem, Anis and Ghorbel, Enjie and Aouada, Djamila},
  title     = {LAA-Net: Localized Artifact Attention Network for Quality-Agnostic and Generalizable Deepfake Detection},
  booktitle = {CVPR},
  year      = {2024}
}
```
