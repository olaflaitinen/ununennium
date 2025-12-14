"""Bounding box utilities for geospatial operations."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    import numpy as np
    from pyproj import Transformer


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in geographic or projected coordinates.

    Attributes:
        minx: Western bound (minimum x coordinate).
        miny: Southern bound (minimum y coordinate).
        maxx: Eastern bound (maximum x coordinate).
        maxy: Northern bound (maximum y coordinate).

    Note:
        Coordinates are in the CRS of the associated GeoTensor.
        For geographic CRS, x = longitude, y = latitude.
        For projected CRS, x = easting, y = northing.
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.minx > self.maxx:
            raise ValueError(f"minx ({self.minx}) must be <= maxx ({self.maxx})")
        if self.miny > self.maxy:
            raise ValueError(f"miny ({self.miny}) must be <= maxy ({self.maxy})")

    @property
    def width(self) -> float:
        """Width of the bounding box (maxx - minx)."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Height of the bounding box (maxy - miny)."""
        return self.maxy - self.miny

    @property
    def area(self) -> float:
        """Area of the bounding box in square units of the CRS."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Center point (x, y) of the bounding box."""
        return (
            (self.minx + self.maxx) / 2,
            (self.miny + self.maxy) / 2,
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (minx, miny, maxx, maxy) tuple."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this bounding box intersects with another.

        Args:
            other: Another bounding box.

        Returns:
            True if the bounding boxes overlap, False otherwise.
        """
        return not (
            self.maxx < other.minx
            or self.minx > other.maxx
            or self.maxy < other.miny
            or self.miny > other.maxy
        )

    def intersection(self, other: BoundingBox) -> BoundingBox | None:
        """Compute the intersection with another bounding box.

        Args:
            other: Another bounding box.

        Returns:
            The intersection bounding box, or None if no intersection.
        """
        if not self.intersects(other):
            return None

        return BoundingBox(
            minx=max(self.minx, other.minx),
            miny=max(self.miny, other.miny),
            maxx=min(self.maxx, other.maxx),
            maxy=min(self.maxy, other.maxy),
        )

    def union(self, other: BoundingBox) -> BoundingBox:
        """Compute the union (minimum bounding box) with another.

        Args:
            other: Another bounding box.

        Returns:
            The smallest bounding box containing both inputs.
        """
        return BoundingBox(
            minx=min(self.minx, other.minx),
            miny=min(self.miny, other.miny),
            maxx=max(self.maxx, other.maxx),
            maxy=max(self.maxy, other.maxy),
        )

    def buffer(self, distance: float) -> BoundingBox:
        """Expand the bounding box by a distance in all directions.

        Args:
            distance: Buffer distance in CRS units. Positive expands,
                negative shrinks.

        Returns:
            New buffered bounding box.
        """
        return BoundingBox(
            minx=self.minx - distance,
            miny=self.miny - distance,
            maxx=self.maxx + distance,
            maxy=self.maxy + distance,
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within the bounding box.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if the point is inside or on the boundary.
        """
        return self.minx <= x <= self.maxx and self.miny <= y <= self.maxy

    def contains_bbox(self, other: BoundingBox) -> bool:
        """Check if this bounding box fully contains another.

        Args:
            other: Another bounding box.

        Returns:
            True if other is completely inside this bounding box.
        """
        return (
            self.minx <= other.minx
            and self.miny <= other.miny
            and self.maxx >= other.maxx
            and self.maxy >= other.maxy
        )

    def transform(self, transformer: Transformer) -> BoundingBox:
        """Transform bounding box to another CRS.

        Args:
            transformer: pyproj Transformer for coordinate conversion.

        Returns:
            Bounding box in the target CRS.

        Note:
            For accuracy, transforms corners and finds the bounding box
            of the transformed corners.
        """
        corners = [
            (self.minx, self.miny),
            (self.minx, self.maxy),
            (self.maxx, self.miny),
            (self.maxx, self.maxy),
        ]

        transformed_x, transformed_y = transformer.transform(
            [c[0] for c in corners],
            [c[1] for c in corners],
        )

        return BoundingBox(
            minx=min(transformed_x),
            miny=min(transformed_y),
            maxx=max(transformed_x),
            maxy=max(transformed_y),
        )

    def to_polygon_coords(self) -> list[tuple[float, float]]:
        """Return polygon coordinates (closed ring, CCW order).

        Returns:
            List of (x, y) tuples forming a closed polygon.
        """
        return [
            (self.minx, self.miny),
            (self.maxx, self.miny),
            (self.maxx, self.maxy),
            (self.minx, self.maxy),
            (self.minx, self.miny),  # Close the ring
        ]

    @classmethod
    def from_array(cls, data: np.ndarray) -> Self:
        """Create bounding box from a 4-element array.

        Args:
            data: Array with [minx, miny, maxx, maxy].

        Returns:
            BoundingBox instance.
        """
        if len(data) != 4:
            raise ValueError(f"Expected 4 elements, got {len(data)}")
        return cls(
            minx=float(data[0]),
            miny=float(data[1]),
            maxx=float(data[2]),
            maxy=float(data[3]),
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(minx={self.minx:.6f}, miny={self.miny:.6f}, "
            f"maxx={self.maxx:.6f}, maxy={self.maxy:.6f})"
        )
