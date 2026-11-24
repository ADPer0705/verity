# M2F2-Det Detector (stage-1 binary detector)

"Rethinking Vision-Language Model in Face Forensics: Multi-Modal
Interpretable Forged Face Detector" (Guo, Song, Zhang, Liu, Liu, CVPR 2025
oral), ported from github.com/CHELSEA234/M2F2_Det (MIT). This is the
**stage-1 binary detector**: a frozen CLIP ViT-L/14-336 (the LLaVA-1.5
vision tower) and an EfficientNet-B4 forensic branch, fused through a
bridge adapter over intermediate features of both. The textual explanation
stages (LLaVA-7B fine-tuned on DD-VQA) are not ported.

## Caveat (reported in every result)

The authors' released `stage_1` weights do not contain the three
bridge-adapter Transformer blocks: upstream keeps them in a plain Python
list, so `state_dict()` never saved them. The port re-initialises them
deterministically (seed 100, as the upstream inference script does
implicitly), which means 128 of the 3,712 features entering the final linear
layer are produced by untrained layers. The score is therefore a *degraded*
version of the published detector; the CLIP and EfficientNet branches
(3,584 features) are intact.

## Setup (uv-managed)

```bash
cd detectors/m2f2_det
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

336x336 MTCNN face crops with CLIP normalisation (what `CLIPImageProcessor`
produces from a square crop); Grad-CAM over the EfficientNet stage-6 map.

## Weights

`weights/m2f2_det_stage1.pth` (official, MIT, 2.0 GB incl. both CLIP
towers); see `weights/README.md`. The CLIP tower config is fetched from
HuggingFace (`openai/clip-vit-large-patch14-336`) to instantiate the
architecture; its weights are overwritten by the checkpoint.
