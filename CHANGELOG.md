# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- GitHub-compatible math formula rendering in documentation
- Conda package recipe and submission guide
- Comprehensive badge set in README

### Changed
- Updated bibliography with valid data

---

## [1.0.7] - 2025-12-16

### Fixed
- Resolved CI/CD pipeline issues caused by history rewrite for security.
- Fixed Conda build dependency `einops` by adding `conda-forge` channel.
- Updated repository workflow configuration to use secure GitHub Secrets.

## [1.0.6] - 2025-12-16

### Added
- Documentation rebuild with comprehensive API reference
- Link checker script for CI validation
- Synthetic image generation for documentation assets

### Changed
- Restructured documentation with consistent style
- Updated all tutorials with runnable examples

---

## [1.0.5] - 2025-12-16

### Added
- ESRGAN super-resolution model implementation
- Siamese change detection network
- PINN advection-diffusion example
- Comprehensive documentation overhaul

### Changed
- Improved GAN training stability
- Enhanced PINN collocation sampling

### Fixed
- CRS preservation in tiling operations
- Memory leak in streaming data loader

---

## [1.0.4] - 2025-12-15

### Added
- CycleGAN unpaired image translation
- Uncertainty calibration module
- Bootstrap confidence intervals for metrics

### Changed
- Optimized COG streaming performance
- Improved mixed precision training stability

### Fixed
- STAC temporal query edge cases
- Gradient accumulation with DDP

---

## [1.0.3] - 2025-12-14

### Added
- Pix2Pix paired image translation
- Physics-informed neural network module
- Collocation point samplers (uniform, Latin hypercube)

### Changed
- Refactored model registry for extensibility
- Updated PyTorch requirement to 2.0+

### Fixed
- GeoTensor coordinate transform propagation
- Loss function NaN handling

---

## [1.0.2] - 2025-12-13

### Added
- ONNX export functionality
- TorchScript compilation support
- Model quantization utilities

### Changed
- Improved checkpoint serialization format
- Enhanced callback system flexibility

### Fixed
- Multi-GPU synchronization in metrics
- Resolution mismatch in super-resolution

---

## [1.0.1] - 2025-12-13

### Added
- Early stopping callback
- Learning rate scheduler integration
- Gradient clipping options

### Changed
- Improved training progress display
- Enhanced error messages

### Fixed
- Validation metric aggregation
- Checkpoint loading from different devices

---

## [1.0.0] - 2025-12-12

### Added
- Core GeoTensor and GeoBatch abstractions
- COG, STAC, and Zarr I/O modules
- U-Net, DeepLabV3+, FPN architectures
- ResNet, EfficientNet, ViT backbones
- Trainer with mixed precision and DDP
- Comprehensive metric suite (IoU, Dice, ECE)
- Spectral index computation (NDVI, NDWI, NBR)
- Tiling and sampling utilities
- Augmentation pipeline

### Changed
- Initial stable release

---

## [0.1.0] - 2025-12-12

### Added
- Initial project structure
- Basic GeoTensor implementation
- Prototype training loop
- Core dependencies

---

[Unreleased]: https://github.com/olaflaitinen/ununennium/compare/v1.0.6...HEAD
[1.0.6]: https://github.com/olaflaitinen/ununennium/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/olaflaitinen/ununennium/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/olaflaitinen/ununennium/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/olaflaitinen/ununennium/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/olaflaitinen/ununennium/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/olaflaitinen/ununennium/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/olaflaitinen/ununennium/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/olaflaitinen/ununennium/releases/tag/v0.1.0
