# Ununennium

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/ununennium?color=blue)](https://pypi.org/project/ununennium/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/ununennium)](https://pypi.org/project/ununennium/)
[![PyPI Status](https://img.shields.io/pypi/status/ununennium.svg)](https://pypi.org/project/ununennium/)
[![Wheel](https://img.shields.io/pypi/wheel/ununennium.svg)](https://pypi.org/project/ununennium/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Repo stars](https://img.shields.io/github/stars/olaflaitinen/ununennium?style=social)](https://github.com/olaflaitinen/ununennium)

[![Build Status](https://img.shields.io/github/actions/workflow/status/olaflaitinen/ununennium/ci.yml?branch=main&label=CI)](https://github.com/olaflaitinen/ununennium/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/ununennium/badge/?version=latest)](https://ununennium.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://img.shields.io/codecov/c/github/olaflaitinen/ununennium)](https://codecov.io/gh/olaflaitinen/ununennium)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20win%20%7C%20macos-lightgrey)](https://pypi.org/project/ununennium/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Issues](https://img.shields.io/github/issues/olaflaitinen/ununennium)](https://github.com/olaflaitinen/ununennium/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/olaflaitinen/ununennium)](https://github.com/olaflaitinen/ununennium/pulls)
[![Contributors](https://img.shields.io/github/contributors/olaflaitinen/ununennium)](https://github.com/olaflaitinen/ununennium/graphs/contributors)
[![Commit Activity](https://img.shields.io/github/commit-activity/m/olaflaitinen/ununennium)](https://github.com/olaflaitinen/ununennium/commits/main)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

**Production-grade Python library for satellite and geospatial imagery machine learning.**

[Documentation](https://ununennium.readthedocs.io) •
[PyPI](https://pypi.org/project/ununennium/) •
[GitHub](https://github.com/olaflaitinen/ununennium) •
[Examples](https://github.com/olaflaitinen/ununennium/tree/main/examples)

</div>

---

## Overview

Ununennium (Element 119, the next alkali metal) represents the cutting edge of satellite imagery machine learning. This library provides a unified, GPU-first framework for end-to-end Earth observation workflows, from cloud-native data access through model training to deployment.

### Why Ununennium?

| Challenge | Traditional Approach | Ununennium Solution |
|-----------|---------------------|---------------------|
| **CRS Handling** | Manual, error-prone | Automatic CRS tracking with `GeoTensor` |
| **Large Rasters** | Memory overflow | Streaming I/O with COG/Zarr |
| **Multi-spectral** | Custom band handling | First-class n-band support |
| **Reproducibility** | Ad-hoc seeds | Deterministic training pipeline |
| **Physics** | Purely data-driven | Physics-informed constraints (PINN) |

---

## Key Features

| Module | Capability | Highlight |
|--------|------------|-----------|
| **`core`** | GeoTensor, GeoBatch | CRS-aware tensors with automatic coordinate tracking |
| **`io`** | COG, STAC, Zarr | Cloud-native streaming with lazy loading |
| **`models`** | CNN, ViT, GAN, PINN | 15+ architectures with registry pattern |
| **`training`** | Trainer, Callbacks | Mixed precision, gradient accumulation, DDP |
| **`preprocessing`** | Indices, Normalization | NDVI, EVI, SAVI with sensor-aware math |
| **`augmentation`** | Geometric, Radiometric | CRS-preserving transforms |
| **`tiling`** | Sampler, Tiler | Overlap-aware patch extraction |
| **`metrics`** | IoU, Dice, ECE | Calibrated uncertainty quantification |
| **`export`** | ONNX, TorchScript | Production deployment ready |

---

## Performance Benchmarks

Benchmarks on NVIDIA A100 80GB, PyTorch 2.1, CUDA 12.1:

| Model | Input Size | Batch | Throughput | Memory | mIoU |
|-------|------------|-------|------------|--------|------|
| U-Net ResNet-50 | 512×512×12 | 16 | 142 img/s | 12.4 GB | 0.78 |
| U-Net EfficientNet-B4 | 512×512×12 | 16 | 98 img/s | 14.2 GB | 0.81 |
| ViT-L/16 | 224×224×12 | 32 | 256 img/s | 18.1 GB | 0.83 |
| Pix2Pix | 256×256×12 | 8 | 67 img/s | 8.6 GB | N/A |

---

## Installation

```bash
# Core installation
pip install ununennium

# With geospatial dependencies (rasterio, pyproj, shapely)
pip install "ununennium[geo]"

# Full installation with all features
pip install "ununennium[all]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+
- GDAL 3.4+ (optional, for geospatial I/O)

---

## Quick Start

### Load Satellite Imagery

```python
import ununennium as uu

# Read with automatic CRS detection
tensor = uu.io.read_geotiff("sentinel2_l2a.tif")
print(f"Shape: {tensor.shape}")     # (12, 10980, 10980)
print(f"CRS: {tensor.crs}")         # EPSG:32632
print(f"Resolution: {tensor.resolution}")  # (10.0, 10.0)
```

### Train a Segmentation Model

```python
from ununennium.models import create_model
from ununennium.training import Trainer, CheckpointCallback
from ununennium.losses import DiceLoss
import torch

# Create U-Net with ResNet-50 backbone
model = create_model(
    "unet_resnet50",
    in_channels=12,      # Sentinel-2 bands
    num_classes=10,      # Land cover classes
)

# Configure training
trainer = Trainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    loss_fn=DiceLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    callbacks=[CheckpointCallback("checkpoints/")],
    mixed_precision=True,
)

# Train with progress tracking
history = trainer.fit(epochs=100)
```

### Physics-Informed Learning

```python
from ununennium.models.pinn import PINN, DiffusionEquation, MLP

# Define PDE constraint
equation = DiffusionEquation(diffusivity=0.1)

# Create PINN
pinn = PINN(
    network=MLP([2, 128, 128, 1]),
    equation=equation,
    lambda_pde=10.0,  # Weight for physics loss
)

# Train with collocation points
losses = pinn.compute_loss(x_data, u_data, x_collocation)
```

---

## Architecture

```
ununennium/
├── core/           # GeoTensor, GeoBatch, types, CRS handling
├── io/             # COG, STAC, Zarr readers/writers
├── preprocessing/  # Normalization, spectral indices
├── augmentation/   # Geometric and radiometric transforms
├── tiling/         # Spatial sampling and tiling
├── datasets/       # Dataset abstractions
├── models/         # Backbones, heads, GAN, PINN
│   ├── backbones/  # ResNet, EfficientNet, ViT
│   ├── heads/      # Classification, Segmentation, Detection
│   ├── gan/        # Pix2Pix, CycleGAN, ESRGAN
│   └── pinn/       # Physics-informed networks
├── losses/         # Dice, Focal, Perceptual, Physics
├── metrics/        # IoU, Dice, Calibration
├── training/       # Trainer, Callbacks, Distributed
├── export/         # ONNX, TorchScript
└── sensors/        # Sentinel-2, Landsat, MODIS specs
```

---

## Supported Tasks

| Task | Models | Metrics |
|------|--------|---------|
| Scene Classification | ResNet, EfficientNet, ViT | Accuracy, F1, AUC |
| Semantic Segmentation | U-Net, DeepLabV3, FPN | mIoU, Dice, PA |
| Object Detection | RetinaNet, FasterRCNN, FCOS | mAP, AP50 |
| Change Detection | Siamese + Diff | F1, κ |
| Super-Resolution | ESRGAN, Real-ESRGAN | PSNR, SSIM, LPIPS |
| Image Translation | Pix2Pix, CycleGAN | FID, SAM |
| Physics-Informed | PINN | L2 Error, PDE Residual |

---

## Documentation

- [Getting Started](https://ununennium.readthedocs.io/getting-started/)
- [API Reference](https://ununennium.readthedocs.io/api/)
- [Tutorials](https://ununennium.readthedocs.io/tutorials/)
- [Model Zoo](https://ununennium.readthedocs.io/models/)
- [Theory](https://ununennium.readthedocs.io/theory/)

---

## Authors

- **Olaf Yunus Laitinen Imanov** - Lead Architect & Creator
- **Hafiz Rzazade** - Contributor
- **Laman Mamedova** - Contributor
- **Farid Mirzaliyev** - Contributor
- **Aian Ajili** - Contributor

---

## Citation

If you use Ununennium in your research, please cite:

```bibtex
@software{ununennium2025,
  title = {Ununennium: Production-grade Satellite Imagery Machine Learning},
  author = {Laitinen Imanov, Olaf Yunus and Rzazade, Hafiz and Mamedova, Laman and Mirzaliyev, Farid and Ajili, Ayan},
  year = {2025},
  version = {1.0.5},
  url = {https://github.com/olaflaitinen/ununennium}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/olaflaitinen/ununennium.git
cd ununennium
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

---

<div align="center">

**Built with passion for Earth observation and machine learning.**

*Ununennium: Where geospatial meets deep learning.*

</div>
