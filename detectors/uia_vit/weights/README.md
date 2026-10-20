#  weights

No checkpoint is loaded today: the network is randomly initialised and the
detector is tier `random-init` (research use only). Upstream initialises from
an ImageNet backbone; wiring that download in, or dropping a trained
checkpoint here and setting `weights.trained: true` in `detector.yaml`, would
promote it.
