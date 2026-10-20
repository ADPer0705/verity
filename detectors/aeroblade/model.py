"""AEROBLADE detector (Ricker et al., CVPR 2024).

Training-free detection of latent-diffusion images via autoencoder
reconstruction error.  Ported from jonasricker/aeroblade using the
``scripts/run_aeroblade.py`` defaults: the input image is reconstructed by the
autoencoders of three LDMs and the reconstruction error is measured with the
second VGG LPIPS layer; the minimum error across the three autoencoders is the
detection score (generated images reconstruct BETTER than real ones, so a low
error indicates a generated image).

Each autoencoder matches the one the official ``image.py``
(``compute_reconstructions``) extracts from
``AutoPipelineForImage2Image.from_pretrained(repo_id, torch_dtype=float16,
use_safetensors=True, variant="fp16" | None)`` as ``pipe.vae`` (SD) or
``pipe.movq`` (Kandinsky VQModel); here the autoencoder subfolder is loaded
directly (see ``AUTOENCODERS``), which is the same module without the unused
UNet/text-encoder download.  Encode/decode use
``retrieve_latents(ae.encode(x), generator)`` with the decode dtype taken from
``ae.post_quant_conv`` parameters (``force_not_quantize=True`` for VQModels).

Preprocessing (``input_mode: full_frame``): the official ``ImageFolder``
(``data.py``) applies only ``ToImage`` + ``ToDtype(float32, scale=True)`` --
no resize, no crop -- and ``image.py`` feeds ``images * 2 - 1`` to the
autoencoder; the paper's images are 512x512 (real LAION images were
center-cropped to 512 by ``img2dataset --resize_mode center_crop
--image_size 512``, the generated ones were sampled at 512).  Resampling
would change the reconstruction error itself, so ``preprocess`` keeps native
pixels: the whole image when both sides are <= 512, otherwise a 512x512
centre crop (the same preparation as the paper's real data, and a memory
bound for the three VAEs); sides are floored to a multiple of 8 for the 8x
latent downsampling.  Output is ``2 * img01 - 1`` (the [-1, 1] range the
autoencoders expect; the yaml mean/std 0.5 describe the same mapping); the
[0, 1] image needed by LPIPS (``normalize=True``) is recovered inside
``forward_detailed`` with ``img01 = (x + 1) / 2``.

Score -> probability is a documented heuristic calibration (see README.md):
``logit_fake = SCALE * (THRESHOLD - d_min)`` with ``SCALE = 10``,
``THRESHOLD = 0.2``.  All heavy imports are lazy so this module imports
cleanly without torch/diffusers/lpips installed.
"""

from __future__ import annotations

# (name, [(repo_id, subfolder, variant), ...]) -- candidates are tried in
# order.  Only the autoencoder subfolder is fetched (``pipe.vae`` /
# ``pipe.movq`` of the reference pipelines are exactly these modules), which
# avoids downloading the multi-GB UNets the detector never uses.
# ``stabilityai/stable-diffusion-2-*`` were removed from the public hub
# (HTTP 401/404 as of 2026-10); ``LanguageMachines/stable-diffusion-2-1-base``
# is a verbatim mirror (its VAE encoder is bit-identical to SD1.1's, only the
# SD2 fine-tuned decoder differs).  It ships fp32 safetensors only, so it is
# loaded with ``variant=None`` and cast to fp16, which yields the same weights
# as the official fp16 variant.
AUTOENCODERS = (
    ("sd1", [("CompVis/stable-diffusion-v1-1", "vae", "fp16")]),
    ("sd2", [("stabilityai/stable-diffusion-2-1-base", "vae", "fp16"),
             ("LanguageMachines/stable-diffusion-2-1-base", "vae", None)]),
    ("kd2", [("kandinsky-community/kandinsky-2-1", "movq", None)]),
)
LPIPS_NET = "vgg"
LPIPS_LAYER = 1  # repo default "lpips_vgg_2": 2nd VGG layer (relu2_2)
SCALE = 10.0
THRESHOLD = 0.2
SEED = 1
CROP = 512     # paper resolution; larger inputs are centre-cropped, never resized
MULTIPLE = 8   # VAE latent downsampling factor


class AEROBLADEModel:
    def __init__(self, device):
        self.device = device
        self.autoencoders = {}
        self.decode_dtype = {}
        self.lpips = None
        self._torch = None
        self._retrieve_latents = None
        self._vqmodel = None
        self._generator = None

    def _load_autoencoder(self, name, candidates):
        """Load one LDM autoencoder per the reference ``image.py``.

        The reference loads the full img2img pipeline and keeps ``pipe.vae``
        (SD, ``AutoencoderKL``) or ``pipe.movq`` (Kandinsky, ``VQModel``);
        ``pipe.upcast_vae`` does not exist on those pipelines in the pinned
        diffusers 0.25.1, so the autoencoder stays in fp16.  Loading the
        subfolder directly is equivalent and only fetches the autoencoder.
        """
        import sys

        from diffusers import AutoencoderKL
        from diffusers.models import VQModel
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )

        errors = []
        ae = None
        for repo_id, subfolder, variant in candidates:
            cls = VQModel if subfolder == "movq" else AutoencoderKL
            try:
                ae = cls.from_pretrained(
                    repo_id,
                    subfolder=subfolder,
                    torch_dtype=self._torch.float16,
                    use_safetensors=True,
                    variant=variant,
                )
                break
            except Exception as exc:  # repo gone/gated, missing variant, ...
                errors.append(f"{repo_id}/{subfolder}: {type(exc).__name__}: {exc}")
                print(f"[aeroblade] {name}: could not load {repo_id}/{subfolder} "
                      f"({type(exc).__name__}); trying next candidate",
                      file=sys.stderr)
        if ae is None:
            raise RuntimeError(
                f"[aeroblade] no candidate repo could be loaded for {name}:\n"
                + "\n".join(errors))
        ae.to(self.device)
        ae.eval()
        decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype
        self._retrieve_latents = retrieve_latents
        self._vqmodel = VQModel
        return ae, decode_dtype

    def preprocess(self, pil):
        """Native-resolution image (whole if <= 512 px, else 512 centre crop),
        sides floored to a multiple of 8, scaled to [-1, 1] -> (1, 3, H, W)."""
        import numpy as np

        torch = self._torch
        pil = pil.convert("RGB")
        w, h = pil.size
        cw = max(MULTIPLE, (min(w, CROP) // MULTIPLE) * MULTIPLE)
        ch = max(MULTIPLE, (min(h, CROP) // MULTIPLE) * MULTIPLE)
        left, top = (w - cw) // 2, (h - ch) // 2
        pil = pil.crop((left, top, left + cw, top + ch))
        arr = np.asarray(pil, dtype="float32").transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr * 2.0 - 1.0).unsqueeze(0)

    def forward_detailed(self, x):
        torch = self._torch
        with torch.no_grad():
            img01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0).to(dtype=torch.float32)
            distances = []
            for name, ae in self.autoencoders.items():
                enc = x.to(device=self.device, dtype=ae.dtype)
                latents = self._retrieve_latents(
                    ae.encode(enc), generator=self._generator
                )
                if isinstance(ae, self._vqmodel):
                    rec = ae.decode(
                        latents.to(self.decode_dtype[name]),
                        force_not_quantize=True,
                        return_dict=False,
                    )[0]
                else:
                    rec = ae.decode(
                        latents.to(self.decode_dtype[name]), return_dict=False
                    )[0]
                rec01 = ((rec / 2.0) + 0.5).clamp(0.0, 1.0).to(dtype=torch.float32)
                _, per_layer = self.lpips(
                    img01, rec01, retPerLayer=True, normalize=True
                )
                distances.append(per_layer[LPIPS_LAYER].flatten(1).mean(dim=1))

            d = torch.stack(distances, dim=1)
            d_min = d.min(dim=1).values
            logit_fake = SCALE * (THRESHOLD - d_min)
            logits = torch.stack([-logit_fake, logit_fake], dim=1)
            probs = torch.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs, "feat": d}


def build_model(weights_path, device):
    """Load the three LDM autoencoders and the LPIPS net (auto-downloaded)."""
    import lpips
    import torch

    model = AEROBLADEModel(device)
    model._torch = torch
    model._generator = torch.Generator().manual_seed(SEED)
    for name, candidates in AUTOENCODERS:
        ae, decode_dtype = model._load_autoencoder(name, candidates)
        model.autoencoders[name] = ae
        model.decode_dtype[name] = decode_dtype
    model.lpips = lpips.LPIPS(net=LPIPS_NET, version="0.1", spatial=False)
    model.lpips.to(device)  # inputs live on `device`; LPIPS buffers must too
    model.lpips.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model
