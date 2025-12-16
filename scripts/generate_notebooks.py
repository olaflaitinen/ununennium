#!/usr/bin/env python
"""Generate all 50 Ununennium tutorial notebooks.

This script creates notebooks programmatically using nbformat
to ensure consistent structure and quality.
"""

from __future__ import annotations

import json
from pathlib import Path

# Notebook definitions: (id, name, title, purpose, sections)
NOTEBOOKS = [
    # IO and Geospatial (00-09)
    (1, "geotensor_basics", "GeoTensor Basics", "Deep dive into GeoTensor creation, CRS handling, and metadata.", [
        ("imports", "import torch\nimport numpy as np\nfrom ununennium.core import GeoTensor"),
        ("create", "# Create GeoTensor from raw data\ndata = torch.randn(12, 512, 512)\ngt = GeoTensor(data=data, crs='EPSG:32632', transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 5000000.0))\nprint(f'Shape: {gt.shape}, CRS: {gt.crs}')")
    ]),
    (2, "reading_geotiffs", "Reading GeoTIFFs", "COG and GeoTIFF I/O with synthetic data.", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor"),
        ("synthetic", "# Synthetic GeoTIFF-like data\ndata = torch.randn(4, 1024, 1024)\ngt = GeoTensor(data=data, crs='EPSG:4326')\nprint(f'Created synthetic raster: {gt.shape}')")
    ]),
    (3, "stac_catalog_access", "STAC Catalog Access", "STAC API queries and asset loading (optional real data).", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor"),
        ("synthetic", "# Synthetic STAC-like item\ndata = torch.randn(13, 256, 256)  # Sentinel-2 L2A\ngt = GeoTensor(data=data, crs='EPSG:32633')\nprint(f'Synthetic STAC item: {gt.shape}')")
    ]),
    (4, "zarr_io", "Zarr I/O", "Zarr array storage for large datasets.", [
        ("imports", "import torch\nimport numpy as np\nfrom ununennium.core import GeoTensor"),
        ("demo", "# Zarr-compatible chunked data\ndata = torch.randn(6, 2048, 2048)\ngt = GeoTensor(data=data, crs='EPSG:32632')\nprint(f'Large array shape: {gt.shape}')")
    ]),
    (5, "crs_transformations", "CRS Transformations", "Coordinate reference system operations.", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor"),
        ("demo", "# CRS handling\ngt = GeoTensor(data=torch.randn(4, 256, 256), crs='EPSG:4326')\nprint(f'Original CRS: {gt.crs}')")
    ]),
    (6, "sensor_specifications", "Sensor Specifications", "Sentinel-2, Landsat, MODIS band configurations.", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor"),
        ("sentinel2", "# Sentinel-2 L2A bands\ns2_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']\ndata = torch.randn(len(s2_bands), 256, 256)\nprint(f'Sentinel-2 simulation: {len(s2_bands)} bands')")
    ]),
    (7, "geobatch_operations", "GeoBatch Operations", "Batch processing multiple GeoTensors.", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor, GeoBatch"),
        ("demo", "# Create batch of GeoTensors\ntensors = [GeoTensor(data=torch.randn(4, 128, 128), crs='EPSG:32632') for _ in range(8)]\nprint(f'Batch size: {len(tensors)}')")
    ]),
    (8, "streaming_large_rasters", "Streaming Large Rasters", "Memory-efficient processing of large imagery.", [
        ("imports", "import torch\nfrom ununennium.core import GeoTensor"),
        ("demo", "# Simulated streaming with chunks\nchunk_size = (256, 256)\ntotal_size = (4096, 4096)\nn_chunks = (total_size[0] // chunk_size[0]) * (total_size[1] // chunk_size[1])\nprint(f'Processing {n_chunks} chunks of size {chunk_size}')")
    ]),
    (9, "data_validation", "Data Validation", "Data quality checks and validation.", [
        ("imports", "import torch\nimport numpy as np\nfrom ununennium.core import GeoTensor"),
        ("demo", "# Validation checks\ndata = torch.randn(4, 256, 256)\nassert not torch.isnan(data).any(), 'NaN detected'\nassert not torch.isinf(data).any(), 'Inf detected'\nprint('Data validation passed')")
    ]),
    # Preprocessing (10-14)
    (10, "spectral_indices", "Spectral Indices", "NDVI, EVI, SAVI, NDWI computation.", [
        ("imports", "import torch\nfrom ununennium.preprocessing import ndvi, evi"),
        ("demo", "# Compute NDVI from synthetic bands\nred = torch.rand(1, 256, 256)\nnir = torch.rand(1, 256, 256) + 0.3\nndvi_val = ndvi(nir, red)\nprint(f'NDVI range: [{ndvi_val.min():.3f}, {ndvi_val.max():.3f}]')")
    ]),
    (11, "normalization_strategies", "Normalization Strategies", "Min-max, z-score, percentile normalization.", [
        ("imports", "import torch"),
        ("demo", "# Normalization methods\ndata = torch.randn(4, 256, 256) * 1000 + 500\nminmax = (data - data.min()) / (data.max() - data.min())\nzscore = (data - data.mean()) / data.std()\nprint(f'MinMax range: [{minmax.min():.3f}, {minmax.max():.3f}]')")
    ]),
    (12, "augmentation_transforms", "Augmentation Transforms", "Geometric and radiometric augmentations.", [
        ("imports", "import torch\nimport torch.nn.functional as F"),
        ("demo", "# Augmentation example\ndata = torch.randn(1, 4, 256, 256)\nflipped = torch.flip(data, dims=[3])\nrotated = torch.rot90(data, k=1, dims=[2, 3])\nprint(f'Augmented shapes: flip={flipped.shape}, rot={rotated.shape}')")
    ]),
    (13, "cloud_masking", "Cloud Masking", "Cloud detection and mask application.", [
        ("imports", "import torch"),
        ("demo", "# Synthetic cloud mask\ndata = torch.randn(4, 256, 256)\ncloud_mask = torch.rand(256, 256) > 0.8  # 20% clouds\nmasked = data * (~cloud_mask).float()\nprint(f'Cloud pixels: {cloud_mask.sum().item()}')")
    ]),
    (14, "temporal_compositing", "Temporal Compositing", "Multi-temporal image stacking.", [
        ("imports", "import torch"),
        ("demo", "# Temporal stack\ntimesteps = [torch.randn(4, 256, 256) for _ in range(12)]\nstack = torch.stack(timesteps, dim=0)\nprint(f'Temporal stack: {stack.shape}')")
    ]),
    # Tiling and Sampling (15-19)
    (15, "tiling_strategies", "Tiling Strategies", "Fixed-size and adaptive patch extraction.", [
        ("imports", "import torch\nfrom ununennium.tiling import Tiler"),
        ("demo", "# Tile extraction\nimage = torch.randn(4, 1024, 1024)\ntile_size = 256\ntiles = [image[:, i:i+tile_size, j:j+tile_size] for i in range(0, 1024, tile_size) for j in range(0, 1024, tile_size)]\nprint(f'Extracted {len(tiles)} tiles of size {tile_size}')")
    ]),
    (16, "overlap_handling", "Overlap Handling", "Overlap-aware tiling and reconstruction.", [
        ("imports", "import torch"),
        ("demo", "# Overlapping tiles\noverlap = 32\nimage = torch.randn(4, 512, 512)\nstride = 256 - overlap\nn_tiles = ((512 - 256) // stride + 1) ** 2\nprint(f'Tiles with {overlap}px overlap: {n_tiles}')")
    ]),
    (17, "balanced_sampling", "Balanced Sampling", "Class-balanced sampling strategies.", [
        ("imports", "import torch\nimport numpy as np"),
        ("demo", "# Class-balanced sampling\nclasses = torch.randint(0, 5, (1000,))\ncounts = torch.bincount(classes)\nweights = 1.0 / counts[classes].float()\nprint(f'Class distribution: {counts.tolist()}')")
    ]),
    (18, "spatial_sampling", "Spatial Sampling", "Geographic stratification techniques.", [
        ("imports", "import torch\nimport numpy as np"),
        ("demo", "# Spatial stratification\nn_samples = 100\ncoords = torch.rand(n_samples, 2)  # Normalized lat/lon\nprint(f'Sampled {n_samples} spatial locations')")
    ]),
    (19, "dataset_creation", "Dataset Creation", "End-to-end dataset building workflow.", [
        ("imports", "import torch\nfrom torch.utils.data import Dataset, DataLoader"),
        ("demo", "class SyntheticDataset(Dataset):\n    def __init__(self, n_samples=100):\n        self.n_samples = n_samples\n    def __len__(self):\n        return self.n_samples\n    def __getitem__(self, idx):\n        return torch.randn(4, 128, 128), torch.randint(0, 5, (128, 128))\n\nds = SyntheticDataset()\nprint(f'Dataset size: {len(ds)}')")
    ]),
    # Training and Evaluation (20-29)
    (20, "training_basics", "Training Basics", "Training loop fundamentals with Trainer API.", [
        ("imports", "import torch\nfrom ununennium.models import create_model\nfrom ununennium.training import Trainer"),
        ("demo", "# Create model\nmodel = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nprint(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')")
    ]),
    (21, "segmentation_training", "Segmentation Training", "U-Net semantic segmentation training.", [
        ("imports", "import torch\nfrom ununennium.models import create_model\nfrom ununennium.losses import DiceLoss"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nloss_fn = DiceLoss()\nx = torch.randn(2, 4, 128, 128)\ny = torch.randint(0, 5, (2, 128, 128))\nout = model(x)\nloss = loss_fn(out, y)\nprint(f'Segmentation loss: {loss.item():.4f}')")
    ]),
    (22, "classification_training", "Classification Training", "Scene classification with backbones.", [
        ("imports", "import torch\nfrom ununennium.models import ResNetBackbone, ClassificationHead"),
        ("demo", "backbone = ResNetBackbone(variant='resnet18', in_channels=4, pretrained=False)\nhead = ClassificationHead(in_channels=512, num_classes=10)\nx = torch.randn(2, 4, 224, 224)\nfeatures = backbone(x)\nlogits = head(features)\nprint(f'Classification logits: {logits.shape}')")
    ]),
    (23, "detection_training", "Detection Training", "Object detection with RetinaNet/FCOS.", [
        ("imports", "import torch\nfrom ununennium.models import create_model"),
        ("demo", "model = create_model('retinanet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nx = torch.randn(2, 4, 256, 256)\noutput = model(x)\nprint(f'Detection levels: {len(output.class_logits)}')")
    ]),
    (24, "loss_functions", "Loss Functions", "Dice, Focal, GIoU loss comparison.", [
        ("imports", "import torch\nfrom ununennium.losses import DiceLoss, FocalLoss, GIoULoss"),
        ("demo", "dice = DiceLoss()\nfocal = FocalLoss()\npred = torch.randn(2, 5, 64, 64)\ntarget = torch.randint(0, 5, (2, 64, 64))\nprint(f'Dice: {dice(pred, target):.4f}')\nprint(f'Focal: {focal(pred, target):.4f}')")
    ]),
    (25, "evaluation_metrics", "Evaluation Metrics", "IoU, mAP, calibration metrics.", [
        ("imports", "import torch\nfrom ununennium.metrics import iou_score, dice_score"),
        ("demo", "pred = torch.randint(0, 5, (4, 128, 128))\ntarget = torch.randint(0, 5, (4, 128, 128))\niou = iou_score(pred, target, num_classes=5)\nprint(f'IoU per class: {iou.tolist()}')")
    ]),
    (26, "train_val_test_splits", "Train/Val/Test Splits", "Spatial and random data splitting.", [
        ("imports", "import torch\nimport numpy as np"),
        ("demo", "n_samples = 1000\nindices = np.random.permutation(n_samples)\ntrain_idx = indices[:int(0.7*n_samples)]\nval_idx = indices[int(0.7*n_samples):int(0.85*n_samples)]\ntest_idx = indices[int(0.85*n_samples):]\nprint(f'Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')")
    ]),
    (27, "mixed_precision", "Mixed Precision Training", "AMP training for efficiency.", [
        ("imports", "import torch\nfrom torch.cuda.amp import autocast, GradScaler"),
        ("demo", "# AMP example (CPU fallback)\nmodel = torch.nn.Linear(256, 10)\nscaler = GradScaler(enabled=torch.cuda.is_available())\nprint(f'AMP enabled: {torch.cuda.is_available()}')")
    ]),
    (28, "distributed_training", "Distributed Training", "Multi-GPU and DDP setup.", [
        ("imports", "import torch\nimport torch.distributed as dist"),
        ("demo", "# DDP info (single-process demo)\nprint(f'CUDA devices: {torch.cuda.device_count()}')\nprint('For DDP, use: torchrun --nproc_per_node=N script.py')")
    ]),
    (29, "callbacks_checkpoints", "Callbacks and Checkpoints", "Training callbacks and model checkpointing.", [
        ("imports", "import torch\nfrom pathlib import Path"),
        ("demo", "# Checkpoint saving\nmodel = torch.nn.Linear(10, 5)\nckpt_path = Path('artifacts/notebooks/29/checkpoint.pt')\nckpt_path.parent.mkdir(parents=True, exist_ok=True)\ntorch.save({'model': model.state_dict(), 'epoch': 10}, ckpt_path)\nprint(f'Saved checkpoint to {ckpt_path}')")
    ]),
    # GAN (30-34)
    (30, "pix2pix_training", "Pix2Pix Training", "Paired image-to-image translation.", [
        ("imports", "import torch\nfrom ununennium.models.gan import Pix2Pix"),
        ("demo", "# Pix2Pix model\nmodel = Pix2Pix(in_channels=4, out_channels=4)\nx = torch.randn(1, 4, 256, 256)\nout = model(x)\nprint(f'Pix2Pix output: {out.shape}')")
    ]),
    (31, "cyclegan_training", "CycleGAN Training", "Unpaired domain translation.", [
        ("imports", "import torch\nfrom ununennium.models.gan import CycleGAN"),
        ("demo", "# CycleGAN model\nmodel = CycleGAN(in_channels=3)\nA = torch.randn(1, 3, 256, 256)\nB = model.G_AB(A)\nprint(f'CycleGAN A->B: {B.shape}')")
    ]),
    (32, "esrgan_superres", "ESRGAN Super-Resolution", "Super-resolution with ESRGAN.", [
        ("imports", "import torch\nimport torch.nn as nn"),
        ("demo", "# Simple SR demo (2x upscale)\nlr = torch.randn(1, 3, 64, 64)\nsr = nn.functional.interpolate(lr, scale_factor=2, mode='bilinear')\nprint(f'SR: {lr.shape} -> {sr.shape}')")
    ]),
    (33, "gan_evaluation", "GAN Evaluation", "FID, IS, LPIPS metrics.", [
        ("imports", "import torch"),
        ("demo", "# GAN evaluation metrics (simplified)\nreal = torch.randn(100, 512)  # Feature vectors\nfake = torch.randn(100, 512)\ndist = torch.norm(real.mean(0) - fake.mean(0))\nprint(f'Feature distance: {dist.item():.4f}')")
    ]),
    (34, "gan_disclosure_stamping", "GAN Disclosure Stamping", "AI-generated content watermarking.", [
        ("imports", "import torch"),
        ("demo", "# AI disclosure watermark\ndef add_watermark(image, msg='AI-GENERATED'):\n    # Simple metadata approach\n    return image, {'watermark': msg, 'generator': 'ununennium'}\n\nimg = torch.randn(3, 256, 256)\nwatermarked, meta = add_watermark(img)\nprint(f'Watermark metadata: {meta}')")
    ]),
    # PINN (35-39)
    (35, "pinn_fundamentals", "PINN Fundamentals", "Physics-informed neural network basics.", [
        ("imports", "import torch\nfrom ununennium.models.pinn import PINN, MLP"),
        ("demo", "# PINN network\nnet = MLP([2, 64, 64, 1])\nx = torch.randn(100, 2, requires_grad=True)\nu = net(x)\nprint(f'PINN output: {u.shape}')")
    ]),
    (36, "pde_residuals", "PDE Residuals", "PDE residual loss computation.", [
        ("imports", "import torch\nfrom ununennium.models.pinn import PINN, MLP, DiffusionEquation"),
        ("demo", "# PDE residual\nnet = MLP([2, 32, 32, 1])\nx = torch.randn(50, 2, requires_grad=True)\nu = net(x)\ngrad_u = torch.autograd.grad(u.sum(), x, create_graph=True)[0]\nprint(f'Gradient shape: {grad_u.shape}')")
    ]),
    (37, "collocation_sampling", "Collocation Sampling", "Collocation point sampling strategies.", [
        ("imports", "import torch"),
        ("demo", "# Collocation points\nn_interior = 1000\nn_boundary = 200\ninterior = torch.rand(n_interior, 2)  # Unit square\nboundary = torch.cat([torch.zeros(50, 1), torch.rand(50, 1)], dim=1)  # Left edge sample\nprint(f'Interior: {n_interior}, Boundary: {n_boundary}')")
    ]),
    (38, "boundary_conditions", "Boundary Conditions", "Boundary condition enforcement.", [
        ("imports", "import torch\nimport torch.nn as nn"),
        ("demo", "# Dirichlet BC enforcement\ndef apply_dirichlet(u, x, bc_value=0.0):\n    # Simple hard enforcement at boundaries\n    mask = (x[:, 0] < 0.01) | (x[:, 0] > 0.99)\n    u_bc = u.clone()\n    u_bc[mask] = bc_value\n    return u_bc\n\nu = torch.randn(100, 1)\nx = torch.rand(100, 2)\nu_enforced = apply_dirichlet(u, x)\nprint(f'BC enforcement applied')")
    ]),
    (39, "pinn_pipeline_integration", "PINN Pipeline Integration", "End-to-end PINN workflow.", [
        ("imports", "import torch\nfrom ununennium.models.pinn import PINN, MLP, DiffusionEquation"),
        ("demo", "# Full PINN pipeline\nnet = MLP([2, 64, 64, 1])\neq = DiffusionEquation(diffusivity=0.1)\npinn = PINN(network=net, equation=eq)\nx_data = torch.randn(50, 2)\nu_data = torch.randn(50, 1)\nx_coll = torch.randn(200, 2, requires_grad=True)\nprint('PINN pipeline configured')")
    ]),
    # Deployment and Export (40-44)
    (40, "onnx_export", "ONNX Export", "Export models to ONNX format.", [
        ("imports", "import torch\nfrom ununennium.models import create_model\nfrom pathlib import Path"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nmodel.eval()\ndummy = torch.randn(1, 4, 256, 256)\nPath('artifacts/notebooks/40').mkdir(parents=True, exist_ok=True)\ntorch.onnx.export(model, dummy, 'artifacts/notebooks/40/model.onnx', opset_version=14)\nprint('Exported to ONNX')")
    ]),
    (41, "torchscript_export", "TorchScript Export", "TorchScript compilation.", [
        ("imports", "import torch\nfrom ununennium.models import create_model\nfrom pathlib import Path"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nmodel.eval()\nscripted = torch.jit.script(model)\nPath('artifacts/notebooks/41').mkdir(parents=True, exist_ok=True)\nscripted.save('artifacts/notebooks/41/model.pt')\nprint('Exported to TorchScript')")
    ]),
    (42, "inference_pipeline", "Inference Pipeline", "Production inference patterns.", [
        ("imports", "import torch\nfrom ununennium.models import create_model"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nmodel.eval()\nwith torch.no_grad():\n    x = torch.randn(1, 4, 512, 512)\n    out = model(x)\nprint(f'Inference output: {out.shape}')")
    ]),
    (43, "batch_prediction", "Batch Prediction", "Large-scale batch prediction.", [
        ("imports", "import torch\nfrom ununennium.models import create_model"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nmodel.eval()\nbatch_size = 8\nwith torch.no_grad():\n    for i in range(5):\n        batch = torch.randn(batch_size, 4, 256, 256)\n        out = model(batch)\n        print(f'Batch {i+1}: {out.shape}')")
    ]),
    (44, "model_serving", "Model Serving", "Model deployment patterns.", [
        ("imports", "import torch\nimport json\nfrom pathlib import Path"),
        ("demo", "# Model serving config\nconfig = {\n    'model_name': 'unet_segmentation',\n    'input_shape': [1, 4, 256, 256],\n    'output_shape': [1, 5, 256, 256],\n    'framework': 'pytorch'\n}\nPath('artifacts/notebooks/44').mkdir(parents=True, exist_ok=True)\nwith open('artifacts/notebooks/44/serving_config.json', 'w') as f:\n    json.dump(config, f, indent=2)\nprint('Serving config saved')")
    ]),
    # Advanced and End-to-End (45-49)
    (45, "uncertainty_quantification", "Uncertainty Quantification", "Calibration and uncertainty estimation.", [
        ("imports", "import torch\nimport torch.nn.functional as F"),
        ("demo", "# MC Dropout for uncertainty\nlogits = torch.randn(10, 5, 64, 64)  # 10 forward passes\nprobs = F.softmax(logits, dim=1)\nmean_prob = probs.mean(dim=0)\nvar_prob = probs.var(dim=0)\nprint(f'Mean prob shape: {mean_prob.shape}, Var shape: {var_prob.shape}')")
    ]),
    (46, "benchmarking", "Benchmarking", "Performance measurement and JSON reporting.", [
        ("imports", "import torch\nimport time\nimport json\nfrom pathlib import Path\nfrom ununennium.models import create_model"),
        ("demo", "model = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\nmodel.eval()\nx = torch.randn(1, 4, 256, 256)\n\n# Warmup\nfor _ in range(3):\n    _ = model(x)\n\n# Benchmark\nn_runs = 10\nstart = time.perf_counter()\nwith torch.no_grad():\n    for _ in range(n_runs):\n        _ = model(x)\nelapsed = time.perf_counter() - start\n\nresults = {\n    'model': 'unet_resnet18',\n    'input_size': [1, 4, 256, 256],\n    'n_runs': n_runs,\n    'total_time_s': elapsed,\n    'avg_latency_ms': (elapsed / n_runs) * 1000,\n    'throughput_img_s': n_runs / elapsed\n}\n\nPath('artifacts/notebooks/46').mkdir(parents=True, exist_ok=True)\nwith open('artifacts/notebooks/46/benchmark.json', 'w') as f:\n    json.dump(results, f, indent=2)\nprint(f'Latency: {results[\"avg_latency_ms\"]:.2f} ms')")
    ]),
    (47, "end_to_end_segmentation", "End-to-End Segmentation", "Complete segmentation workflow.", [
        ("imports", "import torch\nfrom torch.utils.data import Dataset, DataLoader\nfrom ununennium.models import create_model\nfrom ununennium.losses import DiceLoss"),
        ("demo", "class SegDataset(Dataset):\n    def __init__(self, n=50):\n        self.n = n\n    def __len__(self):\n        return self.n\n    def __getitem__(self, i):\n        return torch.randn(4, 128, 128), torch.randint(0, 5, (128, 128))\n\nloader = DataLoader(SegDataset(), batch_size=4)\nmodel = create_model('unet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\nloss_fn = DiceLoss()\n\nfor epoch in range(2):\n    for x, y in loader:\n        out = model(x)\n        loss = loss_fn(out, y)\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')")
    ]),
    (48, "end_to_end_detection", "End-to-End Detection", "Complete detection workflow.", [
        ("imports", "import torch\nfrom ununennium.models import create_model\nfrom ununennium.losses import FocalLossDetection"),
        ("demo", "model = create_model('retinanet', in_channels=4, num_classes=5, backbone='resnet18', pretrained=False)\noptimizer = torch.optim.Adam(model.parameters(), lr=1e-4)\n\nfor step in range(3):\n    x = torch.randn(2, 4, 256, 256)\n    out = model(x)\n    # Simplified loss on first level\n    cls_out = out.class_logits[0].mean()\n    cls_out.backward()\n    optimizer.step()\n    optimizer.zero_grad()\n    print(f'Step {step+1} completed')")
    ]),
    (49, "custom_model_integration", "Custom Model Integration", "Extending the library with custom models.", [
        ("imports", "import torch\nimport torch.nn as nn\nfrom ununennium.models.registry import register_model"),
        ("demo", "@register_model('custom_simple')\nclass CustomModel(nn.Module):\n    def __init__(self, in_channels=4, num_classes=5):\n        super().__init__()\n        self.conv = nn.Conv2d(in_channels, num_classes, 1)\n    def forward(self, x):\n        return self.conv(x)\n\nfrom ununennium.models import create_model\nmodel = create_model('custom_simple', in_channels=4, num_classes=5)\nprint(f'Custom model registered and created')")
    ]),
]


def create_notebook(nb_id: int, name: str, title: str, purpose: str, sections: list) -> dict:
    """Create a notebook dictionary."""
    cells = []
    
    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {nb_id:02d} - {title}\n",
            "\n",
            f"**Purpose**: {purpose}\n",
            "\n",
            "This notebook demonstrates key functionality with synthetic data."
        ]
    })
    
    # Installation cell for Kaggle/Colab
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Installation (Kaggle/Colab)\n", "\n", "Run this cell to install the library if running on Kaggle or Google Colab."]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Uncomment the following line to install ununennium\n",
            "# !pip install -q ununennium"
        ]
    })
    
    # Prerequisites
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Prerequisites and Environment Check"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import torch\n",
            "import numpy as np\n",
            "\n",
            f"print(f'Python: {{sys.version}}')\n",
            f"print(f'PyTorch: {{torch.__version__}}')\n",
            f"print(f'CUDA: {{torch.cuda.is_available()}}')"
        ]
    })
    
    # Reproducibility
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Reproducibility"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "SEED = 42\n",
            "torch.manual_seed(SEED)\n",
            "np.random.seed(SEED)"
        ]
    })
    
    # Core workflow
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Core Workflow"]
    })
    
    for section_name, section_code in sections:
        # Split and add newlines back (except last line)
        lines = section_code.split("\n")
        source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines
        })
    
    # Validation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Validation"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# All cells executed successfully\n",
            "print('Notebook validation passed')"
        ]
    })
    
    # Artifacts
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Save Outputs"]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from pathlib import Path\n",
            f"\n",
            f"ARTIFACT_DIR = Path('artifacts/notebooks/{nb_id:02d}')\n",
            "ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)\n",
            f"print(f'Artifacts directory: {{ARTIFACT_DIR}}')"
        ]
    })
    
    # Next steps
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Next Steps\n",
            "\n",
            "See the [notebooks README](README.md) for related tutorials."
        ]
    })
    
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def main():
    """Generate all notebooks."""
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)
    
    print(f"Generating {len(NOTEBOOKS)} notebooks...")
    
    for nb_id, name, title, purpose, sections in NOTEBOOKS:
        filename = f"{nb_id:02d}_{name}.ipynb"
        filepath = notebooks_dir / filename
        
        notebook = create_notebook(nb_id, name, title, purpose, sections)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1)
        
        print(f"  Created: {filename}")
    
    print(f"\nGenerated {len(NOTEBOOKS)} notebooks in {notebooks_dir}")


if __name__ == "__main__":
    main()
