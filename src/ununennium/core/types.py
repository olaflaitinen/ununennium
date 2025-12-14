"""Type definitions and aliases for the ununennium package."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from affine import Affine
    from pyproj import CRS

# Path types
PathLike: TypeAlias = str | os.PathLike[str]

# Array types
ArrayLike: TypeAlias = np.ndarray | torch.Tensor

# CRS can be specified as string (EPSG code, WKT, PROJ) or CRS object
CRSType: TypeAlias = Union[str, int, "CRS"]

# Affine transform type
TransformType: TypeAlias = "Affine"

# Shape tuple (can be 2D, 3D, or 4D)
Shape: TypeAlias = tuple[int, ...]

# Device specification
Device: TypeAlias = str | torch.device

# Resampling methods
ResamplingMethod: TypeAlias = Literal[
    "nearest",
    "bilinear",
    "cubic",
    "lanczos",
]

# Data types
DType: TypeAlias = np.dtype | torch.dtype | str

# Normalization methods
NormalizationMethod: TypeAlias = Literal[
    "minmax",
    "zscore",
    "percentile",
]

# Common band names (STAC-aligned)
BandName: TypeAlias = Literal[
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "nir09",
    "cirrus",
    "swir16",
    "swir22",
    "thermal",
    "pan",
    "vv",
    "vh",
]

# Sensor identifiers
Sensor: TypeAlias = Literal[
    "sentinel2",
    "sentinel1",
    "landsat8",
    "landsat9",
    "modis",
    "viirs",
    "planet",
    "maxar",
]

# Task types
TaskType: TypeAlias = Literal[
    "classification",
    "segmentation",
    "detection",
    "change_detection",
    "super_resolution",
    "translation",
    "regression",
]
