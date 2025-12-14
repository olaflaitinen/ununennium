# GeoTensor and GeoBatch Data Model

This document details the core data abstractions in Ununennium: `GeoTensor` and `GeoBatch`.

## Mathematical Foundation

### GeoTensor Definition

A GeoTensor T is defined as a tuple:

```
T = (D, C, A, B, N, M)
```

Where:
- **D** ∈ ℝ^(C×H×W) - Data tensor with channels, height, width
- **C** - Coordinate Reference System (CRS)
- **A** - Affine transformation matrix (2×3)
- **B** - Bounding box in CRS units
- **N** - NoData value
- **M** - Band metadata dictionary

### Affine Transform

The affine transform maps pixel coordinates to geographic coordinates:

```
┌ x ┐   ┌ a  b  c ┐ ┌ col ┐
│   │ = │         │ │     │
└ y ┘   └ d  e  f ┘ └ row ┘
```

Where:
- `a` = pixel width (x resolution)
- `e` = pixel height (y resolution, typically negative)
- `c` = x coordinate of upper-left corner
- `f` = y coordinate of upper-left corner
- `b, d` = rotation parameters (usually 0)

### Coordinate Transformation

For a pixel at (col, row):

```
x = a × col + b × row + c
y = d × col + e × row + f
```

Inverse transformation for geographic coordinates (x, y):

```
col = (x × e - y × b - c × e + f × b) / (a × e - b × d)
row = (x × d - y × a - c × d + f × a) / (b × d - a × e)
```

## Shape Conventions

| Tensor Type | Shape | Description |
|-------------|-------|-------------|
| Single image | (C, H, W) | Channels, Height, Width |
| Batch | (B, C, H, W) | Batch, Channels, Height, Width |
| Temporal | (B, T, C, H, W) | Batch, Time, Channels, Height, Width |
| Segmentation labels | (H, W) | Integer class labels |
| Instance masks | (N, H, W) | N instance masks |

## GeoTensor API

```python
@dataclass
class GeoTensor:
    data: torch.Tensor          # (C, H, W) or (B, C, H, W)
    crs: CRS | None             # pyproj.CRS object
    transform: Affine | None    # Affine transform
    band_names: list[str]       # Band identifiers
    nodata: float | None        # NoData sentinel value

    # Properties
    @property
    def shape(self) -> tuple[int, ...]
    @property
    def num_bands(self) -> int
    @property
    def height(self) -> int
    @property
    def width(self) -> int
    @property
    def resolution(self) -> tuple[float, float]
    @property
    def bounds(self) -> BoundingBox

    # Methods
    def to(self, device: str) -> GeoTensor
    def crop(self, bbox: BoundingBox) -> GeoTensor
    def select_bands(self, bands: list[int | str]) -> GeoTensor
    def reproject(self, target_crs: CRS) -> GeoTensor
    def resample(self, scale: float) -> GeoTensor
```

## GeoBatch API

```python
@dataclass
class GeoBatch:
    images: torch.Tensor        # (B, C, H, W)
    labels: torch.Tensor        # (B, H, W) or (B, C, H, W)
    crs: list[CRS]              # CRS per sample
    transforms: list[Affine]    # Transform per sample

    @property
    def batch_size(self) -> int
    @classmethod
    def collate(cls, samples: list[tuple]) -> GeoBatch
```

## Memory Layout

| Format | Layout | Use Case |
|--------|--------|----------|
| `torch.contiguous_format` | C, H, W contiguous | Default operations |
| `torch.channels_last` | H, W, C contiguous | CNN inference optimization |

**Memory-efficient operations:**

```python
# Avoid: Creates copy
tensor.permute(1, 2, 0).contiguous()

# Prefer: Memory format change
tensor.to(memory_format=torch.channels_last)
```

## Type Conversions

| Source | Target | Method |
|--------|--------|--------|
| NumPy → GeoTensor | `GeoTensor(data=np_array)` |
| GeoTensor → NumPy | `tensor.numpy()` |
| GeoTensor → PIL | `tensor.to_pil()` |
| GeoTensor → xarray | `tensor.to_xarray()` |

## BoundingBox

```python
@dataclass
class BoundingBox:
    minx: float   # West boundary
    miny: float   # South boundary
    maxx: float   # East boundary
    maxy: float   # North boundary

    @property
    def width(self) -> float
    @property
    def height(self) -> float
    @property
    def area(self) -> float
    @property
    def center(self) -> tuple[float, float]

    def intersection(self, other: BoundingBox) -> BoundingBox | None
    def union(self, other: BoundingBox) -> BoundingBox
    def buffer(self, distance: float) -> BoundingBox
    def contains(self, x: float, y: float) -> bool
```

## Performance Considerations

| Operation | Time Complexity | Memory Complexity |
|-----------|-----------------|-------------------|
| Band selection | O(C) | O(selected_bands × H × W) |
| Cropping | O(1) | O(crop_H × crop_W × C) |
| Reprojection | O(H × W) | O(H × W × C) |
| CRS transform | O(1) | O(1) |
