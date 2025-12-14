"""Image tiling utilities for processing large rasters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ununennium.core.geotensor import GeoTensor

if TYPE_CHECKING:
    from collections.abc import Iterator


class Tiler:
    """Tile large images into smaller patches with optional overlap.

    Handles the common pattern of splitting large satellite images
    into patches for processing, then reassembling results.

    Example:
        >>> tiler = Tiler(tile_size=256, overlap=32)
        >>> for tile, window in tiler.tile(large_image):
        ...     result = model(tile)
        ...     tiler.add_result(result, window)
        >>> output = tiler.merge()
    """

    def __init__(
        self,
        tile_size: int | tuple[int, int] = 256,
        overlap: int = 0,
        padding_mode: str = "reflect",
    ):
        """Initialize tiler.

        Args:
            tile_size: Size of tiles (H, W) or single value for square.
            overlap: Overlap between adjacent tiles.
            padding_mode: Padding mode for edge tiles.
        """
        if isinstance(tile_size, int):
            self.tile_size = (tile_size, tile_size)
        else:
            self.tile_size = tile_size

        self.overlap = overlap
        self.padding_mode = padding_mode
        self._results: list[tuple[torch.Tensor, tuple[int, int, int, int]]] = []
        self._image_shape: tuple[int, ...] | None = None

    def tile(
        self,
        image: torch.Tensor | GeoTensor,
    ) -> Iterator[tuple[torch.Tensor, tuple[int, int, int, int]]]:
        """Generate tiles from an image.

        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W).

        Yields:
            Tuples of (tile, window) where window is (y, x, h, w).
        """
        data = image.data if isinstance(image, GeoTensor) else image

        # Handle batch dimension
        if data.ndim == 4:
            data = data[0]  # Process one image at a time

        _, h, w = data.shape
        self._image_shape = data.shape
        self._results = []

        th, tw = self.tile_size
        step_h = th - self.overlap
        step_w = tw - self.overlap

        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                # Calculate window
                y_end = min(y + th, h)
                x_end = min(x + tw, w)

                # Extract tile
                tile = data[:, y:y_end, x:x_end]

                # Pad if necessary
                if tile.shape[1] < th or tile.shape[2] < tw:
                    pad_h = th - tile.shape[1]
                    pad_w = tw - tile.shape[2]
                    tile = torch.nn.functional.pad(
                        tile, (0, pad_w, 0, pad_h), mode=self.padding_mode
                    )

                yield tile, (y, x, y_end - y, x_end - x)

    def add_result(
        self,
        result: torch.Tensor,
        window: tuple[int, int, int, int],
    ) -> None:
        """Add a processed tile result.

        Args:
            result: Processed tile.
            window: Original window (y, x, h, w).
        """
        self._results.append((result, window))

    def merge(self) -> torch.Tensor:
        """Merge all tile results into a single image.

        Returns:
            Merged output tensor.
        """
        if self._image_shape is None:
            raise ValueError("Must call tile() before merge()")

        # Determine output shape from first result
        first_result = self._results[0][0]
        out_channels = first_result.shape[0]
        _, h, w = self._image_shape

        # Initialize output and weight tensors
        output = torch.zeros(out_channels, h, w, device=first_result.device)
        weights = torch.zeros(1, h, w, device=first_result.device)

        for result, (y, x, rh, rw) in self._results:
            # Only use valid (non-padded) region
            valid_result = result[:, :rh, :rw]

            # Average overlapping regions
            output[:, y : y + rh, x : x + rw] += valid_result
            weights[:, y : y + rh, x : x + rw] += 1

        # Normalize by weights
        output = output / weights.clamp(min=1)

        return output


def tile_image(
    image: torch.Tensor,
    tile_size: int | tuple[int, int] = 256,
    overlap: int = 0,
) -> list[torch.Tensor]:
    """Split an image into tiles.

    Args:
        image: Input image (C, H, W).
        tile_size: Size of tiles.
        overlap: Overlap between tiles.

    Returns:
        List of tile tensors.
    """
    tiler = Tiler(tile_size=tile_size, overlap=overlap)
    return [tile for tile, _ in tiler.tile(image)]


def untile_image(
    tiles: list[torch.Tensor],
    original_shape: tuple[int, int, int],
    tile_size: int | tuple[int, int] = 256,
    overlap: int = 0,
) -> torch.Tensor:
    """Reassemble tiles into a single image.

    Args:
        tiles: List of tile tensors.
        original_shape: Original image shape (C, H, W).
        tile_size: Size of tiles.
        overlap: Overlap between tiles.

    Returns:
        Merged image tensor.
    """
    tiler = Tiler(tile_size=tile_size, overlap=overlap)

    # Generate windows
    dummy = torch.zeros(original_shape)
    windows = [window for _, window in tiler.tile(dummy)]

    # Add results
    for tile, window in zip(tiles, windows, strict=False):
        tiler.add_result(tile, window)

    return tiler.merge()
