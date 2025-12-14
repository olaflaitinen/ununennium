"""Base dataset classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ununennium.core.geotensor import GeoTensor


class GeoDataset(ABC, Dataset):
    """Abstract base class for geospatial datasets.

    GeoDataset extends PyTorch's Dataset with geospatial awareness,
    providing CRS information and spatial sampling capabilities.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[GeoTensor, Any]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image, label) where image is a GeoTensor.
        """
        ...

    @property
    @abstractmethod
    def crs(self) -> Any:
        """Return the coordinate reference system."""
        ...

    @property
    def num_classes(self) -> int | None:
        """Return number of classes if applicable."""
        return None

    @property
    def class_names(self) -> list[str] | None:
        """Return class names if applicable."""
        return None
