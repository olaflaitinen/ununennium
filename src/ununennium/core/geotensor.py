"""GeoTensor: CRS-aware tensor for geospatial imagery."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import torch
from affine import Affine

from ununennium.core.bounds import BoundingBox

if TYPE_CHECKING:
    from pyproj import CRS

    from ununennium.core.types import Device


@dataclass
class GeoTensor:
    """A georeferenced tensor with coordinate reference system awareness.

    GeoTensor wraps array data (NumPy or PyTorch) with geospatial metadata
    including CRS, affine transform, and band information. It provides
    operations that maintain geographic consistency.

    Attributes:
        data: The underlying array data, shape (C, H, W) or (B, C, H, W).
        crs: Coordinate reference system.
        transform: Affine transform from pixel to world coordinates.
        band_names: Optional names for each band.
        nodata: Value representing missing data.
        timestamp: Acquisition timestamp.

    Shape Conventions:
        - (C, H, W): Single image with C channels
        - (B, C, H, W): Batch of B images
        - (T, C, H, W): Time series of T timesteps
        - (B, T, C, H, W): Batched time series

    Example:
        >>> tensor = GeoTensor(
        ...     data=torch.randn(12, 256, 256),
        ...     crs="EPSG:32632",
        ...     transform=Affine(10, 0, 500000, 0, -10, 5000000),
        ... )
        >>> print(f"Bounds: {tensor.bounds}")
    """

    data: torch.Tensor | np.ndarray
    crs: CRS | None = None
    transform: Affine | None = None
    band_names: list[str] | None = None
    nodata: float | None = None
    timestamp: str | None = None
    _bounds: BoundingBox | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the GeoTensor."""
        # Convert numpy to torch if needed
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data)

        # Validate dimensions
        if self.data.ndim < 2:
            raise ValueError(f"Data must be at least 2D, got {self.data.ndim}D")
        if self.data.ndim > 5:
            raise ValueError(f"Data must be at most 5D, got {self.data.ndim}D")

        # Validate band names if provided
        if self.band_names is not None:
            expected = self.num_bands
            if len(self.band_names) != expected:
                raise ValueError(
                    f"band_names length ({len(self.band_names)}) must match "
                    f"number of bands ({expected})"
                )

        # Parse CRS if string
        if self.crs is not None and isinstance(self.crs, (str, int)):
            from pyproj import CRS as PyprojCRS

            self.crs = PyprojCRS.from_user_input(self.crs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying data."""
        return tuple(self.data.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the underlying tensor."""
        return self.data.dtype

    @property
    def device(self) -> torch.device:
        """Device where the tensor resides."""
        return self.data.device

    @property
    def height(self) -> int:
        """Height of the image in pixels."""
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        """Width of the image in pixels."""
        return self.data.shape[-1]

    @property
    def num_bands(self) -> int:
        """Number of spectral bands (channels)."""
        if self.data.ndim == 2:
            return 1
        elif self.data.ndim == 3:
            return self.data.shape[0]
        else:
            # For 4D or 5D, assume channel is second-to-last spatial dim
            return self.data.shape[-3]

    @property
    def resolution(self) -> tuple[float, float] | None:
        """Pixel resolution (x_res, y_res) in CRS units."""
        if self.transform is None:
            return None
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> BoundingBox | None:
        """Geographic bounds of the image."""
        if self._bounds is not None:
            return self._bounds

        if self.transform is None:
            return None

        # Calculate bounds from transform and shape
        minx = self.transform.c
        maxy = self.transform.f
        maxx = minx + self.width * self.transform.a
        miny = maxy + self.height * self.transform.e

        self._bounds = BoundingBox(
            minx=min(minx, maxx),
            miny=min(miny, maxy),
            maxx=max(minx, maxx),
            maxy=max(miny, maxy),
        )
        return self._bounds

    def to(self, device: Device) -> Self:
        """Move tensor to specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            New GeoTensor on the target device.
        """
        return GeoTensor(
            data=self.data.to(device),
            crs=self.crs,
            transform=self.transform,
            band_names=self.band_names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    def cuda(self, device: int = 0) -> Self:
        """Move tensor to CUDA device.

        Args:
            device: CUDA device index.

        Returns:
            New GeoTensor on CUDA.
        """
        return self.to(f"cuda:{device}")

    def cpu(self) -> Self:
        """Move tensor to CPU.

        Returns:
            New GeoTensor on CPU.
        """
        return self.to("cpu")

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array.

        Returns:
            NumPy array (moved to CPU if necessary).
        """
        if isinstance(self.data, torch.Tensor):
            return self.data.detach().cpu().numpy()
        return self.data

    def float(self) -> Self:
        """Convert to float32.

        Returns:
            New GeoTensor with float32 dtype.
        """
        return GeoTensor(
            data=self.data.float(),
            crs=self.crs,
            transform=self.transform,
            band_names=self.band_names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    def half(self) -> Self:
        """Convert to float16.

        Returns:
            New GeoTensor with float16 dtype.
        """
        return GeoTensor(
            data=self.data.half(),
            crs=self.crs,
            transform=self.transform,
            band_names=self.band_names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    @overload
    def __getitem__(self, key: int) -> Self: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: tuple[Any, ...]) -> Self: ...

    def __getitem__(self, key: int | slice | tuple[Any, ...]) -> Self:
        """Index into the tensor.

        Note:
            Spatial indexing (last two dimensions) updates the transform.
            Band/batch indexing preserves the transform.
        """
        data = self.data[key]

        # Update transform if spatial dimensions were sliced
        new_transform = self.transform
        if self.transform is not None and isinstance(key, tuple) and len(key) >= 2:
            # Handle spatial slicing
            h_key = key[-2] if len(key) >= 2 else slice(None)
            w_key = key[-1]

            if isinstance(h_key, slice) or isinstance(w_key, slice):
                y_start = h_key.start if isinstance(h_key, slice) and h_key.start else 0
                x_start = w_key.start if isinstance(w_key, slice) and w_key.start else 0

                new_transform = self.transform * Affine.translation(x_start, y_start)

        return GeoTensor(
            data=data,
            crs=self.crs,
            transform=new_transform,
            band_names=self.band_names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    def crop(self, bounds: BoundingBox) -> Self:
        """Crop to geographic bounds.

        Args:
            bounds: Target bounding box in the same CRS.

        Returns:
            Cropped GeoTensor.

        Raises:
            ValueError: If bounds don't intersect with the image.
        """
        if self.bounds is None or self.transform is None:
            raise ValueError("Cannot crop without transform")

        if not self.bounds.intersects(bounds):
            raise ValueError("Crop bounds don't intersect with image")

        # Convert bounds to pixel coordinates
        from affine import Affine

        inv_transform = ~self.transform

        col_start, row_start = inv_transform * (bounds.minx, bounds.maxy)
        col_end, row_end = inv_transform * (bounds.maxx, bounds.miny)

        # Clamp to image bounds
        col_start = max(0, int(col_start))
        row_start = max(0, int(row_start))
        col_end = min(self.width, int(np.ceil(col_end)))
        row_end = min(self.height, int(np.ceil(row_end)))

        # Slice data
        if self.data.ndim == 2:
            cropped_data = self.data[row_start:row_end, col_start:col_end]
        elif self.data.ndim == 3:
            cropped_data = self.data[:, row_start:row_end, col_start:col_end]
        else:
            cropped_data = self.data[..., row_start:row_end, col_start:col_end]

        # Update transform
        new_transform = self.transform * Affine.translation(col_start, row_start)

        return GeoTensor(
            data=cropped_data,
            crs=self.crs,
            transform=new_transform,
            band_names=self.band_names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    def select_bands(self, bands: list[int] | list[str]) -> Self:
        """Select specific bands by index or name.

        Args:
            bands: List of band indices or names.

        Returns:
            GeoTensor with selected bands.
        """
        if isinstance(bands[0], str):
            if self.band_names is None:
                raise ValueError("Cannot select by name without band_names")
            indices = [self.band_names.index(b) for b in bands]
            names = list(bands)
        else:
            indices = list(bands)
            names = [self.band_names[i] for i in indices] if self.band_names else None

        if self.data.ndim == 3:
            selected = self.data[indices]
        elif self.data.ndim >= 4:
            selected = self.data[..., indices, :, :]
        else:
            raise ValueError("Cannot select bands from 2D data")

        return GeoTensor(
            data=selected,
            crs=self.crs,
            transform=self.transform,
            band_names=names,
            nodata=self.nodata,
            timestamp=self.timestamp,
        )

    def mask_nodata(self) -> torch.Tensor:
        """Create a boolean mask where True indicates valid (non-nodata) pixels.

        Returns:
            Boolean tensor of shape (H, W).
        """
        if self.nodata is None:
            return torch.ones(self.height, self.width, dtype=torch.bool, device=self.device)

        if self.data.ndim == 2:
            return self.data != self.nodata
        else:
            # Any band has nodata -> pixel is nodata
            return (self.data != self.nodata).all(dim=-3)

    def __repr__(self) -> str:
        crs_str = self.crs.to_epsg() if self.crs else None
        return (
            f"GeoTensor(shape={self.shape}, dtype={self.dtype}, "
            f"crs=EPSG:{crs_str}, device={self.device})"
        )
