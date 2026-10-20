# AEROBLADE Weights

No checkpoint file is stored here — AEROBLADE is training-free. All weights are
auto-downloaded at runtime by `build_model` on the first run and cached by the
HuggingFace hub (`~/.cache/huggingface`) and torch hub
(`~/.cache/torch/hub`). Nothing needs to be fetched with `curl`.

| Component | HuggingFace repo id | How it is loaded | Approx. size |
|---|---|---|---|
| SD1.1 autoencoder | `CompVis/stable-diffusion-v1-1` | diffusers `AutoPipelineForImage2Image` (fp16 variant, safetensors) → `pipe.vae` | ~4 GB (full pipeline; VAE alone ~0.4 GB) |
| SD2.1 autoencoder | `stabilityai/stable-diffusion-2-1-base` | diffusers `AutoPipelineForImage2Image` (fp16 variant, safetensors) → `pipe.vae` | ~4 GB (full pipeline; VAE alone ~0.4 GB) |
| Kandinsky 2.1 autoencoder | `kandinsky-community/kandinsky-2-1` | diffusers `AutoPipelineForImage2Image` (safetensors) → `pipe.movq` (VQModel) | ~3 GB (full pipeline) |
| LPIPS VGG backbone + linear weights | — | `lpips.LPIPS(net="vgg", version="0.1")` via torch hub | ~0.6 GB (`lpips_weights_v0.1/vgg.pth` + torchvision VGG16) |

Notes:

- Only the VAE/VQModel is kept after loading; the UNet / text encoder / prior
  of each pipeline are discarded, but the download and the transient RAM cost
  cover the full pipeline (this mirrors the official `run_aeroblade.py`).
- The SD autoencoders are upcast to fp32 (`pipe.upcast_vae()`) before use,
  exactly like the reference repo; Kandinsky's VQModel stays fp16.
- Offline/frozen environments: pre-populate the hub caches listed above or set
  `HF_HOME` / `TORCH_HOME` to a shared location.
