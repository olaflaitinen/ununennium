"""Tests for datasets."""

import pytest
import torch
from torch.utils.data import Dataset
from ununennium.datasets import GeoDataset, SyntheticDataset

class TestDatasetRegistry:
    def test_synthetic_dataset(self):
        ds = SyntheticDataset(num_samples=10, task="segmentation")
        assert len(ds) == 10
        img, lbl = ds[0]
        assert img.shape[0] == 12  # default channels
        
    def test_classification_task(self):
        ds = SyntheticDataset(num_samples=5, task="classification")
        img, lbl = ds[0]
        assert isinstance(lbl, int) or lbl.ndim == 0
