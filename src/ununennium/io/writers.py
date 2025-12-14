"""Writers for geospatial raster formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ununennium.core.geotensor import GeoTensor
    from ununennium.core.types import PathLike


def write_geotiff(
    tensor: GeoTensor,
    path: PathLike,
    dtype: str | None = None,
    compress: str = "deflate",
    tiled: bool = True,
    tile_size: int = 256,
    nodata: float | None = None,
) -> None:
    """Write a GeoTensor to a GeoTIFF file.

    Args:
        tensor: GeoTensor to write.
        path: Output file path.
        dtype: Output data type (e.g., 'uint8', 'float32').
            If None, uses the tensor's dtype.
        compress: Compression algorithm ('deflate', 'lzw', 'zstd', None).
        tiled: Whether to write as tiled TIFF.
        tile_size: Tile size in pixels (default 256).
        nodata: NoData value to use. If None, uses tensor's nodata.

    Example:
        >>> write_geotiff(tensor, "output.tif", compress="lzw")
    """
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    data = tensor.numpy()

    # Ensure 3D (C, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    # Determine output dtype
    if dtype is None:
        out_dtype = data.dtype
    else:
        out_dtype = np.dtype(dtype)
        data = data.astype(out_dtype)

    # Determine nodata
    out_nodata = nodata if nodata is not None else tensor.nodata

    # Build profile
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
    }

    if tensor.crs is not None:
        profile["crs"] = tensor.crs.to_wkt()

    if tensor.transform is not None:
        profile["transform"] = tensor.transform

    if compress:
        profile["compress"] = compress

    if tiled:
        profile["tiled"] = True
        profile["blockxsize"] = tile_size
        profile["blockysize"] = tile_size

    if out_nodata is not None:
        profile["nodata"] = out_nodata

    # Write
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

        # Write band names if available
        if tensor.band_names is not None:
            for i, name in enumerate(tensor.band_names, 1):
                dst.set_band_description(i, name)


def write_cog(
    tensor: GeoTensor,
    path: PathLike,
    dtype: str | None = None,
    compress: str = "deflate",
    overview_levels: list[int] | None = None,
    overview_resampling: str = "average",
) -> None:
    """Write a GeoTensor as a Cloud-Optimized GeoTIFF (COG).

    A COG is a GeoTIFF with internal tiling and overviews, optimized
    for cloud storage access via HTTP range requests.

    Args:
        tensor: GeoTensor to write.
        path: Output file path.
        dtype: Output data type.
        compress: Compression algorithm.
        overview_levels: Overview factors (e.g., [2, 4, 8, 16]).
            If None, auto-calculates based on image size.
        overview_resampling: Resampling method for overviews.

    Example:
        >>> write_cog(tensor, "output.tif", overview_levels=[2, 4, 8])
    """
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = tensor.numpy()
    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if dtype is None:
        out_dtype = data.dtype
    else:
        out_dtype = np.dtype(dtype)
        data = data.astype(out_dtype)

    # Auto-calculate overview levels
    if overview_levels is None:
        overview_levels = []
        size = max(data.shape[1], data.shape[2])
        factor = 2
        while size // factor > 256:
            overview_levels.append(factor)
            factor *= 2

    # COG profile
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "dtype": out_dtype,
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress,
        "interleave": "pixel",
    }

    if tensor.crs is not None:
        profile["crs"] = tensor.crs.to_wkt()

    if tensor.transform is not None:
        profile["transform"] = tensor.transform

    if tensor.nodata is not None:
        profile["nodata"] = tensor.nodata

    # Write with COG layout
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

        # Build overviews
        if overview_levels:
            resampling = getattr(Resampling, overview_resampling)
            dst.build_overviews(overview_levels, resampling)
            dst.update_tags(ns="rio_overview", resampling=overview_resampling)
