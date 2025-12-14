# Quickstart Tutorial

Get started with Ununennium in 5 minutes.

## Installation

```bash
pip install "ununennium[all]"
```

## Load Satellite Imagery

```python
import ununennium as uu

tensor = uu.io.read_geotiff("path/to/image.tif")
print(f"Shape: {tensor.shape}, CRS: {tensor.crs}")
```

## Create a Model

```python
model = uu.models.create("unet_resnet50", in_channels=12, num_classes=10)
```

## Make Predictions

```python
import torch

with torch.no_grad():
    output = model(tensor.data.unsqueeze(0))
    prediction = output.argmax(dim=1)
```
