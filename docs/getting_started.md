# Getting Started

Welcome to Ununennium! This guide will help you get started with satellite imagery machine learning.

## Installation

```bash
pip install ununennium
```

For full functionality including geospatial I/O:

```bash
pip install "ununennium[all]"
```

## Quick Start

### Loading Satellite Imagery

```python
import ununennium as uu

# Read a GeoTIFF file
tensor = uu.io.read_geotiff("path/to/image.tif")
print(f"Shape: {tensor.shape}")
print(f"CRS: {tensor.crs}")
print(f"Bounds: {tensor.bounds}")

# Select specific bands
rgb = tensor.select_bands([2, 1, 0])  # B4, B3, B2 for RGB

# Crop to an area of interest
from ununennium.core import BoundingBox
bbox = BoundingBox(minx=500000, miny=4500000, maxx=510000, maxy=4510000)
cropped = tensor.crop(bbox)
```

### Creating a Segmentation Model

```python
from ununennium.models import create_model

# Create a U-Net with ResNet-50 backbone for 12-band input
model = create_model(
    "unet_resnet50",
    in_channels=12,
    num_classes=10,
)

# Forward pass
import torch
x = torch.randn(4, 12, 256, 256)
output = model(x)
print(f"Output shape: {output.shape}")  # (4, 10, 256, 256)
```

### Training a Model

```python
from ununennium.training import Trainer, CheckpointCallback
from ununennium.datasets import SyntheticDataset
from torch.utils.data import DataLoader

# Create datasets
train_dataset = SyntheticDataset(num_samples=1000, num_channels=12)
val_dataset = SyntheticDataset(num_samples=200, num_channels=12)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Setup training
model = create_model("unet_resnet50", in_channels=12, num_classes=10)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

callbacks = [
    CheckpointCallback("checkpoints/", monitor="val_loss"),
]

# Train
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=callbacks,
)

history = trainer.fit(epochs=10)
```

## Next Steps

- [API Reference](api/core.md)
- [Tutorials](tutorials/segmentation.md)
- [Model Zoo](models/zoo.md)
