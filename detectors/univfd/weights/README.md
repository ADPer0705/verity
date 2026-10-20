# UnivFD Weights

Two weight sources combine at runtime:

1. **Linear probe** (`fc_weights.pth`, 4083 bytes, md5
   `392b2d8b637e932a2534ded56d9185bd`) — official checkpoint committed in the
   UnivFD repo, shape `{"weight": (1, 768), "bias": (1,)}`. Download:

   ```bash
   curl -L -o fc_weights.pth 'https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth'
   ```

2. **CLIP ViT-L/14 backbone** — not a local file; fetched automatically from
   HuggingFace by transformers at runtime (`openai/clip-vit-large-patch14`).
   The backbone is frozen; only the probe carries learned deepfake-detection
   weights.
