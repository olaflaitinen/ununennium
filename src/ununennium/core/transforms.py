"""Affine transform utilities for geospatial operations."""

from __future__ import annotations

import math

from affine import Affine


def create_transform(
    x_res: float,
    y_res: float,
    x_origin: float,
    y_origin: float,
    rotation: float = 0.0,
) -> Affine:
    """Create an affine transform from resolution and origin.

    The transform maps pixel (col, row) to geographic (x, y):
        x = a * col + b * row + c
        y = d * col + e * row + f

    Args:
        x_res: X resolution (pixel size in X direction).
        y_res: Y resolution (pixel size in Y direction, typically negative).
        x_origin: X coordinate of top-left corner.
        y_origin: Y coordinate of top-left corner.
        rotation: Rotation angle in degrees (default 0).

    Returns:
        Affine transform object.

    Example:
        >>> transform = create_transform(10.0, -10.0, 500000, 5000000)
    """

    if rotation != 0:
        theta = math.radians(rotation)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        return Affine(
            x_res * cos_t,
            -y_res * sin_t,
            x_origin,
            x_res * sin_t,
            y_res * cos_t,
            y_origin,
        )
    return Affine(x_res, 0, x_origin, 0, y_res, y_origin)


def pixel_to_geo(
    transform: Affine,
    col: float,
    row: float,
) -> tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates.

    Args:
        transform: Affine transform.
        col: Column (x-pixel).
        row: Row (y-pixel).

    Returns:
        Tuple of (x, y) geographic coordinates.
    """
    x, y = transform * (col, row)  # type: ignore
    return (x, y)


def geo_to_pixel(
    transform: Affine,
    x: float,
    y: float,
) -> tuple[float, float]:
    """Convert geographic coordinates to pixel coordinates.

    Args:
        transform: Affine transform.
        x: X geographic coordinate.
        y: Y geographic coordinate.

    Returns:
        Tuple of (col, row) pixel coordinates.
    """
    inv = ~transform
    col, row = inv * (x, y)  # type: ignore
    return (col, row)


def get_resolution(transform: Affine) -> tuple[float, float]:
    """Get pixel resolution from transform.

    Args:
        transform: Affine transform.

    Returns:
        Tuple of (x_res, y_res) in CRS units.
    """
    return (abs(transform.a), abs(transform.e))


def scale_transform(transform: Affine, scale: float) -> Affine:
    """Scale transform to different resolution.

    Args:
        transform: Original transform.
        scale: Scale factor (2.0 = double resolution).

    Returns:
        Scaled affine transform.
    """
    return transform * Affine.scale(1 / scale, 1 / scale)  # type: ignore[return-value]


def translate_transform(transform: Affine, dx: float, dy: float) -> Affine:
    """Translate transform origin.

    Args:
        transform: Original transform.
        dx: Translation in X.
        dy: Translation in Y.

    Returns:
        Translated affine transform.
    """
    return Affine(
        transform.a,
        transform.b,
        transform.c + dx,
        transform.d,
        transform.e,
        transform.f + dy,
    )
