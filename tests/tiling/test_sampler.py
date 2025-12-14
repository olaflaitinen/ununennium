"""Tests for tiling."""

import pytest
import torch

from ununennium.tiling import Tiler, tile_image


class TestTiling:
    def test_tile_image(self):
        image = torch.randn(3, 512, 512)
        tiles = tile_image(image, tile_size=256)
        assert len(tiles) == 4

    def test_tiler_merge(self):
        tiler = Tiler(tile_size=256, overlap=0)
        image = torch.randn(3, 512, 512)

        results = []
        for tile, window in tiler.tile(image):
            results.append((tile + 1, window))  # Simple operation

        for tile, window in results:
            tiler.add_result(tile, window)

        merged = tiler.merge()
        assert merged.shape == image.shape
