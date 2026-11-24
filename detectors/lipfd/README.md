# LipFD Detector

"Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual
in Lip-syncing DeepFakes" (Liu, She, Liu, Li, Yao, Liang, Wang, NeurIPS
2024), ported from github.com/AaronComo/LipFD. LipFD is the first detector
dedicated to lip-sync forgeries, which keep the identity and show no
blending artifacts: it looks at the *temporal* agreement between the audio
spectrogram and the lip/head motion. Trained on the authors' AVLips dataset
(Wav2Lip, TalkLip, MakeItTalk, SadTalker generators); 95.3 % accuracy on
AVLips and up to 90.2 % on real-world video-call recordings in the paper.

## How it scores

For each window (5 consecutive frames; ten evenly spaced windows per video):

1. the frames are resized to 500x500 and tiled; the viridis-rendered mel
   spectrogram of the soundtrack (22.05 kHz) for the same span is stacked on
   top (1000 x 2500 composite, exactly `preprocess.py`);
2. the composite (resized to 1120, BGR 0-255 as in `datasets.py`) goes
   through a 5x5/5 conv and the CLIP ViT-L/14 image encoder for a global
   feature;
3. five region views of the frame row, at three scales, are encoded by a
   ResNet-50, concatenated with the global feature and gated by learned
   region weights; the averaged vector gives one logit (sigmoid = P(fake)).

Upstream slides the five region views by one pixel rather than one frame
(`img[:, 500:, i:i+500]` for i in 0..4); the port reproduces this verbatim
because the released weights were trained that way.

## Setup (uv-managed)

```bash
cd detectors/lipfd
uv sync
```

## Run

```bash
INPUT_DIR=/path/to/dir/containing/media OUTPUT_DIR=/path/to/output \
python inference.py
```

Videos with a soundtrack (`input_mode: av_clip`); `timeline.png` shows the
per-window fake probability over the spectrogram.

## Weights

`weights/ckpt.pth` (official, 1.8 GB incl. the CLIP tower; no license
stated, fetched from the authors' Google Drive only). See
`weights/README.md`.
