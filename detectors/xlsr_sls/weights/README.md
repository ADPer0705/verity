# XLS-R + SLS Weights

| Property | Value |
|---|---|
| File | `xlsr-sls.onnx` |
| Size | 1,356,274,368 bytes |
| sha256 | `0aa36298a1a893bfb58a8d721dd30d046157f559da1165f95d2a0b69988c1bbd` |
| Origin | HF `SpeechAntiSpoofingBenchmarks/XLSR-SLS` (ONNX export, opset 17, of the official `MMpaper_model.pth` = `model_15` from github.com/QiShanZhang/SLSforASVspoof-2021-DF) |
| Training | ASVspoof2019 LA train (dev EER 1.45 %); ASVspoof2021 DF EER 1.92 %, In-the-Wild 7.46 % (paper) |
| License | MIT per the HF model card; the upstream GitHub repository ships **no LICENSE file** (none stated), so treat redistribution as uncertain |

This file is **not tracked in git**. Download it with:

```bash
curl -L -o detectors/xlsr_sls/weights/xlsr-sls.onnx \
  'https://huggingface.co/SpeechAntiSpoofingBenchmarks/XLSR-SLS/resolve/main/xlsr-sls.onnx'
```

ONNX I/O: input `wav` float32 (batch, 64600) raw 16 kHz waveform (first
64,600 samples, tile-repeat if shorter, no normalisation); output `logits`
(batch, 2) = upstream log-softmax over [spoof, bonafide]. The port swaps the
columns so Verity's index 1 = fake.

Citation:

```bibtex
@inproceedings{zhang2024audio,
  title={Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier},
  author={Zhang, Qishan and Wen, Shuangbing and Hu, Tao},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6765--6773},
  year={2024},
  doi={10.1145/3664647.3681345}
}
```
