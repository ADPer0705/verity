# UnivFD Detector

Universal fake image detector (UnivFD, Ojha et al., CVPR 2023) ported from
[WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect).

Architecture: frozen CLIP ViT-L/14 (HuggingFace `openai/clip-vit-large-patch14`)
followed by the official 1-dim linear probe (`fc_weights.pth`). The probe is fed
the 768-d CLIP-projected image embedding (equivalent to the official repo's
`CLIP.encode_image`), and its single output logit is the fake score:
P(fake) = sigmoid(s). The Verity adapter stacks `[0, s]` as 2-class logits so
that `softmax[:, 1] == sigmoid(s)` exactly.

## Setup (uv-managed)

```bash
cd detectors/univfd
uv sync
```


## Weights

The linear probe is committed upstream and downloaded locally:

```bash
mkdir -p weights
curl -L -o weights/fc_weights.pth 'https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth'
```

The CLIP ViT-L/14 backbone is fetched automatically from HuggingFace by
transformers at runtime (no local file needed).

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

`INPUT_DIR/media/` is scanned for media files; artifacts (faces, `result.json`)
are written to `OUTPUT_DIR`.
