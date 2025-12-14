"""CRS (Coordinate Reference System) utilities."""

from __future__ import annotations

from typing import Any

import pyproj
from pyproj import CRS

CRSType = str | int | CRS | dict[str, Any]


def parse_crs(crs: CRSType) -> CRS:
    """Parse CRS from various input formats.

    Args:
        crs: CRS as EPSG code (int), WKT/PROJ string, or pyproj.CRS.

    Returns:
        pyproj.CRS object.

    Raises:
        ValueError: If CRS cannot be parsed.

    Example:
        >>> crs = parse_crs("EPSG:4326")
        >>> crs = parse_crs(32632)
    """
    if isinstance(crs, CRS):
        return crs
    if isinstance(crs, int):
        return CRS.from_epsg(crs)
    if isinstance(crs, str):
        if crs.upper().startswith("EPSG:"):
            return CRS.from_epsg(int(crs.split(":")[1]))
        return CRS.from_string(crs)
    if isinstance(crs, dict):
        return CRS.from_dict(crs)

    raise ValueError(f"Cannot parse CRS from {type(crs)}: {crs}")


def crs_to_epsg(crs: CRSType) -> int | None:
    """Extract EPSG code from CRS.

    Args:
        crs: CRS object.

    Returns:
        EPSG code or None if not available.
    """
    parsed = parse_crs(crs)
    return parsed.to_epsg()


def crs_to_wkt(crs: CRSType) -> str:
    """Convert CRS to WKT2 string.

    Args:
        crs: CRS object.

    Returns:
        WKT2 representation.
    """
    parsed = parse_crs(crs)
    return parsed.to_wkt()


def are_crs_equal(crs1: CRSType, crs2: CRSType) -> bool:
    """Check if two CRS are equivalent.

    Args:
        crs1: First CRS.
        crs2: Second CRS.

    Returns:
        True if CRS are equivalent.
    """
    return parse_crs(crs1).equals(parse_crs(crs2))


def get_transformer(
    source_crs: CRSType,
    target_crs: CRSType,
    always_xy: bool = True,
) -> pyproj.Transformer:
    """Get a coordinate transformer between CRS.

    Args:
        source_crs: Source coordinate reference system.
        target_crs: Target coordinate reference system.
        always_xy: If True, coordinates are (x, y) order.

    Returns:
        pyproj.Transformer object.
    """
    return pyproj.Transformer.from_crs(
        parse_crs(source_crs),
        parse_crs(target_crs),
        always_xy=always_xy,
    )


def transform_coordinates(
    x: float,
    y: float,
    source_crs: CRSType,
    target_crs: CRSType,
) -> tuple[float, float]:
    """Transform coordinates between CRS.

    Args:
        x: X coordinate (or longitude).
        y: Y coordinate (or latitude).
        source_crs: Source CRS.
        target_crs: Target CRS.

    Returns:
        Tuple of (x, y) in target CRS.
    """
    transformer = get_transformer(source_crs, target_crs)
    return transformer.transform(x, y)
