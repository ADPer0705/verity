# RECCE Weights

| Property | Value |
|---|---|
| File | `recce_best.pth` |
| Size | 191,466,941 bytes |
| MD5 | `7610c0e485af5962e3c9d78020d95e65` |
| Origin | Official DeepfakeBench v1.0.1 release (byte-identical to the checkpoint shipped in the `SCLBD/DeepfakeBench` GitHub release assets) |
| Training | FaceForensics++ (c23), DeepfakeBench default config (`cross_entropy` loss on the classification head only) |
| License | Follows the RECCE / DeepfakeBench project terms; research use only |

## What this file is for

This checkpoint is **not distributed and not tracked in git**. It exists on
this machine for a single purpose: rebuilding the detector container:

```bash
python scripts/build_sif.py recce --push-sif
```

The Dockerfile does `COPY weights/ /detector/weights/`, baking the model into
`containers/recce.sif`. **The SIF is the deployment artifact** — a deployed
system only needs the SIF, never this file. To rebuild a SIF, re-obtain this
file from the official DeepfakeBench release if it is missing.

## Important: the reconstruction decoder is untrained in this checkpoint

The published checkpoint is trained with classification loss only, so the
reconstruction decoder (and the graph-reasoning branch) received no gradients
during training. In practice this means:

- `decoder1`–`decoder5` pointwise convolutions hold values on the order of
  `1e-11`, `decoder6` around `1e-7`, and batch-norm `running_var` is ~0
  (`1e-19`…`1e-45`).
- The decoder output is a constant tensor (~0 after `tanh`), so the
  reconstruction and residual maps carry no information.
- This is **not** a wiring bug in this pipeline: we verified the encoder
  features reaching the decoder are alive (block4 embedding std ≈ 2.08) and
  the classifier head, guided attention, and encoder are fully trained and
  produce the published results (e.g. 99.3 AUC on FF++ c23).

The detector therefore audits decoder liveness at load time (`tensor_audit` in
`result.json`, `reconstruction.available: false`) and, when the decoder is
dead, produces **no** reconstruction visualizations, region-energy scores, or
reconstruction-based reasoning text. Classification still uses the trained
classifier logits + attention maps.

If a properly trained RECCE checkpoint (one trained with reconstruction loss)
becomes available, replace this file and the liveness guards will
automatically re-enable the full reconstruction analysis.
