"""Readers for various geospatial raster formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ununennium.core.geotensor import GeoTensor

if TYPE_CHECKING:
    from ununennium.core.bounds import BoundingBox
    from ununennium.core.types import PathLike


def read_geotiff(
    path: PathLike,
    bands: list[int] | None = None,
    window: tuple[int, int, int, int] | None = None,
    bounds: BoundingBox | None = None,
) -> GeoTensor:
    """Read a GeoTIFF file into a GeoTensor.

    Args:
        path: Path to the GeoTIFF file.
        bands: Optional list of band indices (1-indexed) to read.
            If None, reads all bands.
        window: Optional pixel window (row_start, col_start, height, width).
            Mutually exclusive with bounds.
        bounds: Optional geographic bounds to read.
            Mutually exclusive with window.

    Returns:
        GeoTensor with the raster data and metadata.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidFormatError: If the file is not a valid GeoTIFF.

    Example:
        >>> tensor = read_geotiff("image.tif")
        >>> tensor = read_geotiff("image.tif", bands=[1, 2, 3])
        >>> tensor = read_geotiff("image.tif", bounds=bbox)
    """
    import rasterio
    from pyproj import CRS
    from rasterio.windows import Window, from_bounds

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        # Determine which bands to read
        if bands is None:
            bands = list(range(1, src.count + 1))

        # Determine window
        rio_window = None
        if window is not None and bounds is not None:
            raise ValueError("Cannot specify both window and bounds")

        if window is not None:
            row_start, col_start, height, width = window
            rio_window = Window(col_start, row_start, width, height)

        if bounds is not None:
            rio_window = from_bounds(
                bounds.minx,
                bounds.miny,
                bounds.maxx,
                bounds.maxy,
                src.transform,
            )

        # Read data
        data = src.read(bands, window=rio_window)

        # Get transform for the window
        transform = src.window_transform(rio_window) if rio_window is not None else src.transform

        # Convert to torch tensor
        tensor = torch.from_numpy(data.astype(np.float32))

        # Get band names if available
        band_names = None
        if src.descriptions:
            band_names = [src.descriptions[i - 1] for i in bands]
            # Replace None with default names
            band_names = [name if name else f"band_{i}" for i, name in enumerate(band_names, 1)]

        return GeoTensor(
            data=tensor,
            crs=CRS.from_wkt(src.crs.to_wkt()) if src.crs else None,
            transform=transform,
            band_names=band_names,
            nodata=src.nodata,
        )


def read_cog(
    url: str,
    bounds: BoundingBox | None = None,
    resolution: float | None = None,
    bands: list[int] | None = None,
) -> GeoTensor:
    """Read a Cloud-Optimized GeoTIFF (COG) from a URL.

    This function uses HTTP range requests to efficiently read only
    the required tiles from a remote COG.

    Args:
        url: URL to the COG (http://, https://, s3://, gs://).
        bounds: Optional geographic bounds to read.
        resolution: Target resolution in CRS units. If specified,
            reads from the appropriate overview level.
        bands: Optional list of band indices (1-indexed) to read.

    Returns:
        GeoTensor with the raster data.

    Example:
        >>> tensor = read_cog(
        ...     "https://example.com/image.tif",
        ...     bounds=bbox,
        ...     resolution=10.0,
        ... )
    """
    import rasterio
    from affine import Affine
    from pyproj import CRS
    from rasterio.windows import from_bounds

    with rasterio.open(url) as src:
        # Determine overview level based on resolution
        overview_level = None
        if resolution is not None:
            native_res = abs(src.transform.a)
            overviews = src.overviews(1)
            for i, factor in enumerate(overviews):
                if native_res * factor >= resolution:
                    overview_level = i
                    break

        # Determine which bands to read
        if bands is None:
            bands = list(range(1, src.count + 1))

        # Determine window
        rio_window = None
        if bounds is not None:
            rio_window = from_bounds(
                bounds.minx,
                bounds.miny,
                bounds.maxx,
                bounds.maxy,
                src.transform,
            )

        # Calculate output shape based on overview level
        out_shape = None
        if overview_level is not None and rio_window is not None:
            factor = src.overviews(1)[overview_level]
            out_shape = (
                len(bands),
                int(rio_window.height / factor),
                int(rio_window.width / factor),
            )

        # Read data
        data = src.read(
            bands,
            window=rio_window,
            out_shape=out_shape,
        )

        # Get transform for the window
        if rio_window is not None:
            transform = src.window_transform(rio_window)
            if overview_level is not None:
                # Scale transform for overview
                factor = src.overviews(1)[overview_level]
                transform = Affine(
                    transform.a * factor,
                    transform.b,
                    transform.c,
                    transform.d,
                    transform.e * factor,
                    transform.f,
                )
        else:
            transform = src.transform

        tensor = torch.from_numpy(data.astype(np.float32))

        return GeoTensor(
            data=tensor,
            crs=CRS.from_wkt(src.crs.to_wkt()) if src.crs else None,
            transform=transform,
            nodata=src.nodata,
        )
