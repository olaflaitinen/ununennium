"""Tests for I/O module."""

import pytest
import torch
import numpy as np
from pathlib import Path
import rasterio

from ununennium.io import read_geotiff, write_geotiff
from ununennium.core import GeoTensor


class TestGeoTIFF:
    def test_read_geotiff(self, tmp_path):
        # Create a dummy GeoTIFF
        path = tmp_path / "test.tif"
        data = np.random.rand(3, 100, 100).astype(np.float32)
        transform = rasterio.transform.from_origin(0, 0, 10, 10)
        
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=3,
            dtype=data.dtype,
            crs="EPSG:32632",
            transform=transform,
        ) as dst:
            dst.write(data)
            
        # Read back
        tensor = read_geotiff(path)
        assert isinstance(tensor, GeoTensor)
        assert tensor.shape == (3, 100, 100)
        assert tensor.crs.to_epsg() == 32632
        
    def test_write_geotiff(self, tmp_path):
        tensor = GeoTensor(
            data=torch.randn(3, 50, 50),
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 0, 0.001, 0.001),
            band_names=["R", "G", "B"],
            nodata=None
        )
        
        path = tmp_path / "output.tif"
        write_geotiff(tensor, path)
        assert path.exists()
        
        with rasterio.open(path) as src:
            assert src.count == 3
            assert src.shape == (50, 50)
            assert src.crs.to_epsg() == 4326
