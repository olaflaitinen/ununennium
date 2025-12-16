# Ununennium Notebooks

Comprehensive tutorial and example notebooks covering the entire Ununennium library.

---

## Quick Start

```bash
# Run smoke tests (5 notebooks, for CI)
make notebooks-smoke

# Run all notebooks locally
make notebooks-all

# Or using Python directly
python scripts/run_notebooks_smoke.py
python scripts/run_notebooks_all.py
```

---

## Notebook Index

### IO and Geospatial (00-09)

| ID | Notebook | Description |
|----|----------|-------------|
| 00 | [quickstart](00_quickstart.ipynb) | Library overview, installation verification, first GeoTensor |
| 01 | [geotensor_basics](01_geotensor_basics.ipynb) | GeoTensor creation, CRS handling, metadata preservation |
| 02 | [reading_geotiffs](02_reading_geotiffs.ipynb) | COG and GeoTIFF I/O with synthetic and optional real data |
| 03 | [stac_catalog_access](03_stac_catalog_access.ipynb) | STAC API queries and asset loading |
| 04 | [zarr_io](04_zarr_io.ipynb) | Zarr array storage for large datasets |
| 05 | [crs_transformations](05_crs_transformations.ipynb) | Coordinate reference system operations |
| 06 | [sensor_specifications](06_sensor_specifications.ipynb) | Sentinel-2, Landsat, MODIS band configurations |
| 07 | [geobatch_operations](07_geobatch_operations.ipynb) | Batch processing multiple GeoTensors |
| 08 | [streaming_large_rasters](08_streaming_large_rasters.ipynb) | Memory-efficient processing of large imagery |
| 09 | [data_validation](09_data_validation.ipynb) | Data quality checks, NaN detection, CRS validation |

---

### Preprocessing (10-14)

| ID | Notebook | Description |
|----|----------|-------------|
| 10 | [spectral_indices](10_spectral_indices.ipynb) | NDVI, EVI, SAVI, NDWI computation |
| 11 | [normalization_strategies](11_normalization_strategies.ipynb) | Min-max, z-score, percentile normalization |
| 12 | [augmentation_transforms](12_augmentation_transforms.ipynb) | Geometric and radiometric augmentations |
| 13 | [cloud_masking](13_cloud_masking.ipynb) | Cloud detection and mask application |
| 14 | [temporal_compositing](14_temporal_compositing.ipynb) | Multi-temporal image stacking |

---

### Tiling and Sampling (15-19)

| ID | Notebook | Description |
|----|----------|-------------|
| 15 | [tiling_strategies](15_tiling_strategies.ipynb) | Fixed-size and adaptive patch extraction |
| 16 | [overlap_handling](16_overlap_handling.ipynb) | Overlap-aware tiling and reconstruction |
| 17 | [balanced_sampling](17_balanced_sampling.ipynb) | Class-balanced sampling strategies |
| 18 | [spatial_sampling](18_spatial_sampling.ipynb) | Geographic stratification techniques |
| 19 | [dataset_creation](19_dataset_creation.ipynb) | End-to-end dataset building workflow |

---

### Training and Evaluation (20-29)

| ID | Notebook | Description |
|----|----------|-------------|
| 20 | [training_basics](20_training_basics.ipynb) | Training loop fundamentals with Trainer API |
| 21 | [segmentation_training](21_segmentation_training.ipynb) | U-Net semantic segmentation training |
| 22 | [classification_training](22_classification_training.ipynb) | Scene classification with backbones |
| 23 | [detection_training](23_detection_training.ipynb) | Object detection with RetinaNet/FCOS |
| 24 | [loss_functions](24_loss_functions.ipynb) | Dice, Focal, GIoU loss comparison |
| 25 | [evaluation_metrics](25_evaluation_metrics.ipynb) | IoU, mAP, calibration metrics |
| 26 | [train_val_test_splits](26_train_val_test_splits.ipynb) | Spatial and random data splitting |
| 27 | [mixed_precision](27_mixed_precision.ipynb) | AMP training for efficiency |
| 28 | [distributed_training](28_distributed_training.ipynb) | Multi-GPU and DDP setup |
| 29 | [callbacks_checkpoints](29_callbacks_checkpoints.ipynb) | Training callbacks and model checkpointing |

---

### GAN (30-34)

| ID | Notebook | Description |
|----|----------|-------------|
| 30 | [pix2pix_training](30_pix2pix_training.ipynb) | Paired image-to-image translation |
| 31 | [cyclegan_training](31_cyclegan_training.ipynb) | Unpaired domain translation |
| 32 | [esrgan_superres](32_esrgan_superres.ipynb) | Super-resolution with ESRGAN |
| 33 | [gan_evaluation](33_gan_evaluation.ipynb) | FID, IS, LPIPS metrics |
| 34 | [gan_disclosure_stamping](34_gan_disclosure_stamping.ipynb) | AI-generated content watermarking |

---

### PINN (35-39)

| ID | Notebook | Description |
|----|----------|-------------|
| 35 | [pinn_fundamentals](35_pinn_fundamentals.ipynb) | Physics-informed neural network basics |
| 36 | [pde_residuals](36_pde_residuals.ipynb) | PDE residual loss computation |
| 37 | [collocation_sampling](37_collocation_sampling.ipynb) | Collocation point sampling strategies |
| 38 | [boundary_conditions](38_boundary_conditions.ipynb) | Boundary condition enforcement |
| 39 | [pinn_pipeline_integration](39_pinn_pipeline_integration.ipynb) | End-to-end PINN workflow |

---

### Deployment and Export (40-44)

| ID | Notebook | Description |
|----|----------|-------------|
| 40 | [onnx_export](40_onnx_export.ipynb) | Export models to ONNX format |
| 41 | [torchscript_export](41_torchscript_export.ipynb) | TorchScript compilation |
| 42 | [inference_pipeline](42_inference_pipeline.ipynb) | Production inference patterns |
| 43 | [batch_prediction](43_batch_prediction.ipynb) | Large-scale batch prediction |
| 44 | [model_serving](44_model_serving.ipynb) | Model deployment patterns |

---

### Advanced and End-to-End (45-49)

| ID | Notebook | Description |
|----|----------|-------------|
| 45 | [uncertainty_quantification](45_uncertainty_quantification.ipynb) | Calibration and uncertainty estimation |
| 46 | [benchmarking](46_benchmarking.ipynb) | Performance measurement and JSON reporting |
| 47 | [end_to_end_segmentation](47_end_to_end_segmentation.ipynb) | Complete segmentation workflow |
| 48 | [end_to_end_detection](48_end_to_end_detection.ipynb) | Complete detection workflow |
| 49 | [custom_model_integration](49_custom_model_integration.ipynb) | Extending the library with custom models |

---

## Artifacts

Notebook outputs are saved under `artifacts/notebooks/<notebook_id>/`.

## Notes

- All notebooks run on CPU by default using synthetic data.
- Real data examples are optional and require explicit opt-in.
- No credentials or large binaries required.
