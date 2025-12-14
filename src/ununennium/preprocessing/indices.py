"""Spectral indices for remote sensing analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ununennium.core.geotensor import GeoTensor

if TYPE_CHECKING:
    import torch


def ndvi(
    nir: torch.Tensor,
    red: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-infrared band.
        red: Red band.
        epsilon: Small value to avoid division by zero.

    Returns:
        NDVI values in range [-1, 1].
    """
    return (nir - red) / (nir + red + epsilon)


def ndwi(
    green: torch.Tensor,
    nir: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate Normalized Difference Water Index.

    NDWI = (Green - NIR) / (Green + NIR)

    Args:
        green: Green band.
        nir: Near-infrared band.
        epsilon: Small value to avoid division by zero.

    Returns:
        NDWI values in range [-1, 1].
    """
    return (green - nir) / (green + nir + epsilon)


def evi(
    nir: torch.Tensor,
    red: torch.Tensor,
    blue: torch.Tensor,
    g: float = 2.5,
    c1: float = 6.0,
    c2: float = 7.5,
    L_factor: float = 1.0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate Enhanced Vegetation Index.

    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)

    Args:
        nir: Near-infrared band.
        red: Red band.
        blue: Blue band.
        g: Gain factor.
        c1: Red correction coefficient.
        c2: Blue correction coefficient.
        L_factor: Canopy background adjustment.
        epsilon: Small value to avoid division by zero.

    Returns:
        EVI values.
    """
    denominator = nir + c1 * red - c2 * blue + L_factor + epsilon
    return g * (nir - red) / denominator


def savi(
    nir: torch.Tensor,
    red: torch.Tensor,
    L_factor: float = 0.5,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate Soil-Adjusted Vegetation Index.

    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

    Args:
        nir: Near-infrared band.
        red: Red band.
        L_factor: Soil brightness correction factor.
        epsilon: Small value to avoid division by zero.

    Returns:
        SAVI values.
    """
    return ((nir - red) / (nir + red + L_factor + epsilon)) * (1 + L_factor)


def nbr(
    nir: torch.Tensor,
    swir: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Calculate Normalized Burn Ratio.

    NBR = (NIR - SWIR) / (NIR + SWIR)

    Args:
        nir: Near-infrared band.
        swir: Short-wave infrared band.
        epsilon: Small value to avoid division by zero.

    Returns:
        NBR values in range [-1, 1].
    """
    return (nir - swir) / (nir + swir + epsilon)


def compute_index(
    tensor: GeoTensor,
    index_name: str,
    band_mapping: dict[str, int] | None = None,
) -> GeoTensor:
    """Compute a spectral index from a GeoTensor.

    Args:
        tensor: Input GeoTensor with multi-spectral bands.
        index_name: Name of the index ('ndvi', 'ndwi', 'evi', etc.).
        band_mapping: Mapping from band names to indices.
            If None, uses Sentinel-2 L2A ordering.

    Returns:
        GeoTensor with computed index as single band.
    """
    # Default Sentinel-2 L2A band ordering (10m + 20m bands)
    if band_mapping is None:
        band_mapping = {
            "blue": 0,  # B02
            "green": 1,  # B03
            "red": 2,  # B04
            "rededge1": 3,  # B05
            "rededge2": 4,  # B06
            "rededge3": 5,  # B07
            "nir": 6,  # B08
            "rededge4": 7,  # B8A
            "swir16": 8,  # B11
            "swir22": 9,  # B12
        }

    import numpy as np  # noqa: PLC0415
    import torch as _torch  # noqa: PLC0415

    raw_data = tensor.data
    if isinstance(raw_data, _torch.Tensor):
        data = raw_data
    else:
        data = _torch.from_numpy(np.asarray(raw_data))

    if index_name.lower() == "ndvi":
        result = ndvi(
            data[..., band_mapping["nir"], :, :],
            data[..., band_mapping["red"], :, :],
        )
    elif index_name.lower() == "ndwi":
        result = ndwi(
            data[..., band_mapping["green"], :, :],
            data[..., band_mapping["nir"], :, :],
        )
    elif index_name.lower() == "evi":
        result = evi(
            data[..., band_mapping["nir"], :, :],
            data[..., band_mapping["red"], :, :],
            data[..., band_mapping["blue"], :, :],
        )
    else:
        raise ValueError(f"Unknown index: {index_name}")

    # Add channel dimension
    result = result.unsqueeze(-3)

    return GeoTensor(
        data=result,
        crs=tensor.crs,
        transform=tensor.transform,
        band_names=[index_name.upper()],
        nodata=tensor.nodata,
    )
