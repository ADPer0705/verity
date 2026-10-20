# AEROBLADE Detector

Training-free detection of latent diffusion images using autoencoder
reconstruction error (Ricker et al., CVPR 2024), ported from
[jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) using the
`scripts/run_aeroblade.py` defaults (3 AEs, `lpips_vgg_2` distance).

## Setup (uv-managed)

```bash
cd detectors/aeroblade
uv sync
```


## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, `result.json`)
are written to `OUTPUT_DIR`.

## Method

1. **Reconstruction.** The input face crop is reconstructed by the latent
   autoencoders of three LDMs, loaded (and auto-downloaded) from the
   HuggingFace hub via diffusers exactly like the official repo's
   `image.py`/`run_aeroblade.py`:
   - `CompVis/stable-diffusion-v1-1` (SD1.1, `pipe.vae`)
   - `stabilityai/stable-diffusion-2-1-base` (SD2.1, `pipe.vae`)
     (the official default is `stabilityai/stable-diffusion-2-base`; SD2.1 is
     used here per the porting spec — both ship the same `sd-vae-ft-ema` VAE)
   - `kandinsky-community/kandinsky-2-1` (Kandinsky 2.1, `pipe.movq` VQModel)

   Encode is `retrieve_latents(ae.encode(x), generator)` (KL-VAE posterior
   sampling / VQModel quantization), decode follows the repo's dtype handling
   (`post_quant_conv` dtype, `force_not_quantize=True` for the VQModel), and
   the output is de-normalized back to [0, 1].

2. **Distance.** The second VGG LPIPS layer (`lpips_vgg_2` = VGG `relu2_2`,
   the best layer in the paper) measures the reconstruction error between the
   original and the reconstruction.

3. **Aggregation.** Generated images reconstruct *better* than real ones, so
   the **minimum** error across the three autoencoders (`d_min`, the repo's
   ΔMax of negative distances / "best AE" row) is the detection score.

4. **Score → probability.** Heuristic, **uncalibrated** mapping:
   `logit_fake = 10 * (0.2 - d_min)`, `logits = [-logit_fake, logit_fake]`,
   `p_fake = sigmoid(10 * (0.2 - d_min))`. `d_min = 0.2` gives `p = 0.5`.
   These constants were picked so typical generated-image distances (≈0.05)
   land above 0.5 and typical real-image distances (≈0.3–0.6) below it. The
   official AEROBLADE evaluation is AP-based on the raw distances; this
   calibration is a convenience for the Verity 2-class contract and should not
   be treated as a trained confidence.

## Input preprocessing

`detector.yaml` uses `resolution: 512` (SD training size, divisible by the AEs'
8× downsampling) and `mean/std = 0.5`. The engine's `(img01 - 0.5) / 0.5`
normalization maps the image to [-1, 1], which is exactly the autoencoder input
range; the [0, 1] image needed by LPIPS is recovered in `forward_detailed`.

## Weights

No checkpoint file. On the first run `build_model` downloads:

| Component | Source | Approx. size |
|---|---|---|
| SD1.1 pipeline (VAE used) | `CompVis/stable-diffusion-v1-1` (HF hub) | ~4 GB |
| SD2.1 pipeline (VAE used) | `stabilityai/stable-diffusion-2-1-base` (HF hub) | ~4 GB |
| Kandinsky 2.1 pipeline (VAE used) | `kandinsky-community/kandinsky-2-1` (HF hub) | ~3 GB |
| LPIPS VGG | `lpips` package → torch hub cache | ~0.6 GB |

Cached by HuggingFace hub / torch hub on the host (see `weights/README.md`).

## Computational cost

This detector is **expensive**: three autoencoders reconstruct every face at
512×512, so per-image inference is orders of magnitude slower than a CNN
classifier, and the model load downloads several GB on first use. A GPU is
strongly recommended (CPU fp16/fp32 VAE inference is very slow). The
autoencoders are loaded once in `build_model` and cached in the model instance
— never per image. Videos face the same per-frame cost for every sampled face.

## Explainability

No attention maps / CAM (training-free; `cam: false`). The `feat` output of
`forward_detailed` carries the three per-AE LPIPS distances per face.
