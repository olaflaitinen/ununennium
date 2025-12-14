"""Core data structures and abstractions."""

from ununennium.core.bounds import BoundingBox
from ununennium.core.geobatch import GeoBatch
from ununennium.core.geotensor import GeoTensor
from ununennium.core.types import (
    ArrayLike,
    CRSType,
    Device,
    PathLike,
    Shape,
    TransformType,
)

__all__ = [
    "ArrayLike",
    "BoundingBox",
    "CRSType",
    "Device",
    "GeoBatch",
    "GeoTensor",
    "PathLike",
    "Shape",
    "TransformType",
]
