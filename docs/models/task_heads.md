# Task Heads

Output heads for different ML tasks.

## Classification

```python
from ununennium.models.heads import ClassificationHead

head = ClassificationHead(in_channels=2048, num_classes=10)
```

## Segmentation

```python
from ununennium.models.heads import SegmentationHead

head = SegmentationHead(encoder_channels=[256, 512, 1024, 2048], num_classes=10)
```
