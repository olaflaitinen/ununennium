# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy 1.24+

## Installation Options

### Basic Installation

```bash
pip install ununennium
```

### With Geospatial Support

```bash
pip install "ununennium[geo]"
```

Includes: rasterio, pyproj, affine, shapely

### With STAC Support

```bash
pip install "ununennium[stac]"
```

Includes: pystac, pystac-client, planetary-computer

### With Zarr/Cloud Storage

```bash
pip install "ununennium[zarr]"
```

Includes: zarr, xarray, fsspec, s3fs, gcsfs

### Full Installation

```bash
pip install "ununennium[all]"
```

All optional dependencies included.

## Development Installation

```bash
git clone https://github.com/olaflaitinen/ununennium.git
cd ununennium
pip install -e ".[dev]"
pre-commit install
```

## GPU Support

PyTorch with CUDA must be installed separately:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install ununennium
```

## Verification

```python
import ununennium as uu
print(uu.__version__)  # Should print 1.0.5
```
