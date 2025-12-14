"""Tests for core data structures."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ununennium.core import BoundingBox, GeoTensor, GeoBatch


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_creation(self):
        """Test basic bounding box creation."""
        bbox = BoundingBox(minx=0, miny=0, maxx=100, maxy=100)
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.area == 10000

    def test_validation(self):
        """Test that invalid bounds raise an error."""
        with pytest.raises(ValueError):
            BoundingBox(minx=100, miny=0, maxx=0, maxy=100)

    def test_intersection(self):
        """Test bounding box intersection."""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)

        intersection = bbox1.intersection(bbox2)
        assert intersection is not None
        assert intersection.minx == 50
        assert intersection.miny == 50
        assert intersection.maxx == 100
        assert intersection.maxy == 100

    def test_no_intersection(self):
        """Test non-intersecting bounding boxes."""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(200, 200, 300, 300)

        assert not bbox1.intersects(bbox2)
        assert bbox1.intersection(bbox2) is None

    def test_union(self):
        """Test bounding box union."""
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)

        union = bbox1.union(bbox2)
        assert union.minx == 0
        assert union.miny == 0
        assert union.maxx == 150
        assert union.maxy == 150

    def test_contains_point(self):
        """Test point containment."""
        bbox = BoundingBox(0, 0, 100, 100)
        assert bbox.contains(50, 50)
        assert bbox.contains(0, 0)
        assert bbox.contains(100, 100)
        assert not bbox.contains(150, 50)

    def test_buffer(self):
        """Test bounding box buffering."""
        bbox = BoundingBox(10, 10, 90, 90)
        buffered = bbox.buffer(10)
        assert buffered.minx == 0
        assert buffered.maxx == 100


class TestGeoTensor:
    """Tests for GeoTensor class."""

    def test_creation_from_numpy(self):
        """Test GeoTensor creation from numpy array."""
        data = np.random.randn(12, 256, 256).astype(np.float32)
        tensor = GeoTensor(data=data)

        assert isinstance(tensor.data, torch.Tensor)
        assert tensor.shape == (12, 256, 256)
        assert tensor.num_bands == 12
        assert tensor.height == 256
        assert tensor.width == 256

    def test_creation_from_torch(self):
        """Test GeoTensor creation from torch tensor."""
        data = torch.randn(12, 256, 256)
        tensor = GeoTensor(data=data)

        assert tensor.shape == (12, 256, 256)
        assert tensor.dtype == torch.float32

    def test_device_transfer(self, device):
        """Test moving GeoTensor between devices."""
        tensor = GeoTensor(data=torch.randn(12, 256, 256))

        moved = tensor.to(device)
        assert str(moved.device).startswith(device)

    def test_numpy_conversion(self):
        """Test conversion back to numpy."""
        original = np.random.randn(12, 256, 256).astype(np.float32)
        tensor = GeoTensor(data=original)
        result = tensor.numpy()

        np.testing.assert_allclose(result, original)

    def test_band_names_validation(self):
        """Test band names length validation."""
        data = torch.randn(12, 256, 256)

        with pytest.raises(ValueError, match="band_names length"):
            GeoTensor(data=data, band_names=["a", "b", "c"])  # Only 3 names for 12 bands

    def test_select_bands_by_index(self):
        """Test band selection by index."""
        data = torch.randn(12, 256, 256)
        tensor = GeoTensor(data=data)

        selected = tensor.select_bands([0, 3, 7])
        assert selected.num_bands == 3

    def test_nodata_mask(self):
        """Test nodata masking."""
        data = torch.randn(3, 100, 100)
        data[0, 50, 50] = -9999
        tensor = GeoTensor(data=data, nodata=-9999)

        mask = tensor.mask_nodata()
        assert mask.shape == (100, 100)
        assert not mask[50, 50]  # This pixel is nodata


class TestGeoBatch:
    """Tests for GeoBatch class."""

    def test_creation(self):
        """Test GeoBatch creation."""
        images = torch.randn(4, 12, 256, 256)
        labels = torch.randint(0, 10, (4, 256, 256))

        batch = GeoBatch(images=images, labels=labels)

        assert batch.batch_size == 4
        assert batch.num_channels == 12
        assert batch.height == 256

    def test_shape_mismatch(self):
        """Test batch size mismatch raises error."""
        images = torch.randn(4, 12, 256, 256)
        labels = torch.randint(0, 10, (3, 256, 256))  # Wrong batch size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            GeoBatch(images=images, labels=labels)

    def test_device_transfer(self, device):
        """Test moving GeoBatch to device."""
        batch = GeoBatch(
            images=torch.randn(4, 12, 256, 256),
            labels=torch.randint(0, 10, (4, 256, 256)),
        )

        moved = batch.to(device)
        assert str(moved.images.device).startswith(device)
        assert str(moved.labels.device).startswith(device)

    def test_iteration(self):
        """Test iterating over batch."""
        batch = GeoBatch(images=torch.randn(4, 12, 256, 256))

        samples = list(batch)
        assert len(samples) == 4
        assert all(isinstance(s, GeoTensor) for s in samples)
