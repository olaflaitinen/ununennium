# Architecture Overview

This document provides a comprehensive overview of the Ununennium library architecture, design principles, and system components.

## Design Philosophy

Ununennium is built on five core principles:

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **CRS Awareness** | Every tensor carries its coordinate reference system | `GeoTensor` wrapper with CRS metadata |
| **Type Safety** | Full static typing for early error detection | pyright strict mode, comprehensive annotations |
| **Modularity** | Compose complex systems from simple parts | Registry pattern, backbone/head separation |
| **Cloud-Native** | First-class support for cloud storage | COG, STAC, Zarr, fsspec integration |
| **Reproducibility** | Deterministic training for science | Seed management, checkpoint versioning |

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Application Layer                          │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │   CLI   │  │ Training │  │ Inference│  │   Experiment Mgmt    │ │
│  └────┬────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────┘ │
└───────┼────────────┼─────────────┼───────────────────┼─────────────┘
        │            │             │                   │
┌───────┼────────────┼─────────────┼───────────────────┼─────────────┐
│       │      Model Composition Layer                 │             │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌───────────▼──────────┐  │
│  │Backbones│  │  Heads  │  │  Losses │  │        Metrics       │  │
│  │ ResNet  │  │  Seg.   │  │  Dice   │  │  IoU, Dice, Pixel    │  │
│  │ EffNet  │  │  Class. │  │  Focal  │  │  Calibration         │  │
│  │   ViT   │  │  Det.   │  │  Percep │  │  Uncertainty         │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────────────────────┘  │
└───────┼────────────┼─────────────┼─────────────────────────────────┘
        │            │             │
┌───────┼────────────┼─────────────┼─────────────────────────────────┐
│       │       Data Pipeline Layer                                  │
│  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐  ┌──────────────────────┐  │
│  │ Dataset │  │ Tiling  │  │  Aug.   │  │    Preprocessing     │  │
│  │GeoDataset│ │ Sampler │  │ Compose │  │  Norm, Indices       │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  └──────────┬───────────┘  │
└───────┼────────────┼─────────────┼──────────────────┼──────────────┘
        │            │             │                  │
┌───────┼────────────┼─────────────┼──────────────────┼──────────────┐
│       │        Core Data Layer                      │              │
│  ┌────▼─────────────▼─────────────▼─────────────────▼──────────┐  │
│  │                     GeoTensor / GeoBatch                    │  │
│  │   CRS  │  Transform  │  Bounds  │  Band Specs  │  NoData   │  │
│  └────────────────────────────────┬────────────────────────────┘  │
└───────────────────────────────────┼────────────────────────────────┘
                                    │
┌───────────────────────────────────┼────────────────────────────────┐
│                          I/O Layer                                 │
│  ┌─────────┐  ┌─────────┐  ┌──────▼──┐  ┌──────────────────────┐  │
│  │ GeoTIFF │  │   COG   │  │  STAC   │  │        Zarr          │  │
│  └─────────┘  └─────────┘  └─────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

| Module | Dependencies | Dependents |
|--------|-------------|------------|
| `core` | numpy, torch | All modules |
| `io` | core, rasterio | datasets, preprocessing |
| `preprocessing` | core, torch | datasets, augmentation |
| `augmentation` | core, torch | datasets |
| `tiling` | core | datasets |
| `datasets` | core, io, preprocessing | training |
| `models` | core, torch | training, export |
| `losses` | torch | training |
| `metrics` | torch | training, evaluation |
| `training` | models, datasets, losses | cli |
| `export` | models, torch | cli |

## Data Flow

**Training Pipeline:**
```
Raw Data → I/O (COG/STAC) → GeoTensor → Preprocessing → Augmentation
    → Tiling → Dataset → DataLoader → Model → Loss → Optimizer
```

**Inference Pipeline:**
```
Raw Data → I/O → GeoTensor → Tiling → Model → Merge → GeoTIFF
```

## Memory Management

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| Lazy Loading | Large COG files | `rasterio.open()` with windowed read |
| Tile Processing | Very large rasters | `Tiler` with overlap |
| Mixed Precision | GPU memory reduction | `torch.cuda.amp.autocast()` |
| Gradient Checkpointing | Large models | `torch.utils.checkpoint` |

## Extensibility Points

1. **Model Registry**: Add new architectures via `@register_model`
2. **Loss Registry**: Custom losses with `nn.Module`
3. **Callback System**: Training hooks for logging, checkpointing
4. **Dataset Adapters**: Wrap external formats
5. **Sensor Specs**: Add new satellite specifications
