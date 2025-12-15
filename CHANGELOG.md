# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] - 2025-12-15

### Changed
- Updated documentation formulas for proper MathJax rendering
- Improved metadata and author attributions

### Fixed
- Formula rendering issues in Markdown documentation files

---

## [1.0.0] - 2025-12-15

### Added
- Core `GeoTensor` and `GeoBatch` abstractions for CRS-aware tensors
- I/O support for GeoTIFF, COG (Cloud-Optimized GeoTIFF)
- Preprocessing utilities: normalization, spectral indices (NDVI, EVI, NDWI)
- Tiling module with spatial samplers
- Training system with `Trainer`, callbacks (Checkpoint, EarlyStopping)
- Model architectures: U-Net with configurable backbones
- ResNet and EfficientNet backbones
- Classification, Segmentation, and FPN heads
- GAN module: Pix2Pix, CycleGAN, generators, discriminators
- GAN losses: Adversarial, Perceptual, Spectral Angle
- PINN module: Base PINN class, PDE equations, collocation samplers
- Benchmarking utilities: Profiler, throughput measurement
- CLI with train, evaluate, export commands
- Synthetic dataset for testing
- Comprehensive test suite (29 tests)
- Full documentation with MkDocs
- CI/CD workflows for GitHub Actions

### Contributors
- Olaf Yunus Laitinen Imanov (Lead Architect)
- Hafiz Rzazade (Contributor)
- Laman Mamedova (Contributor)
- Farid Mirzaliyev (Contributor)
- Ayan Ajili (Contributor)
