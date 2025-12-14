# Ununennium

**Production-grade Python library for satellite and geospatial imagery machine learning.**

## Features

- **Cloud-Native I/O**: COG, STAC, Zarr support
- **CRS-Aware**: Geographic coordinate tracking
- **Modern Architectures**: CNNs, ViTs, Foundation Models
- **GAN Support**: Image translation and super-resolution
- **Physics-Informed**: PDE-constrained learning
- **Reproducible**: Deterministic training

## Architecture

```mermaid
graph TD
    A[Raw Satellite Data] --> B[I/O Layer];
    B -->|GeoTensor| C[Preprocessing];
    C -->|Augmentation| D[Dataset];
    D -->|GeoBatch| E[Training Loop];
    E -->|Backprop| F[Model Registry];
    F -->|Update| E;
    F -->|Export| G[ONNX / TorchScript];
    
    subgraph "Core Modules"
    B[I/O: COG/STAC]
    C[Preprocessing: Norm/Indices]
    F[Models: CNN/ViT/GAN/PINN]
    end
```

## Quick Start


```python
import ununennium as uu

# Load imagery
tensor = uu.io.read_geotiff("image.tif")

# Create model
model = uu.models.create("unet", in_channels=12, num_classes=10)

# Train
trainer = uu.training.Trainer(model=model, ...)
trainer.fit(epochs=100)
```

## Installation

```bash
pip install "ununennium[all]"
```

## Documentation

- [Getting Started](getting_started.md)
- [API Reference](api/core.md)
- [Tutorials](tutorials/segmentation.md)
