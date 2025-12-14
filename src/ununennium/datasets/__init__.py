"""Datasets module for data loading."""

from ununennium.datasets.base import GeoDataset
from ununennium.datasets.synthetic import SyntheticDataset

__all__ = [
    "GeoDataset",
    "SyntheticDataset",
]
