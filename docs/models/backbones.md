# Backbones

Feature extraction backbones for model architectures.

## ResNet

```python
from ununennium.models.backbones import ResNetBackbone

backbone = ResNetBackbone(variant="resnet50", in_channels=12)
```

## EfficientNet

```python
from ununennium.models.backbones import EfficientNetBackbone

backbone = EfficientNetBackbone(variant="efficientnet_b0")
```
