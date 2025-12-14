"""Synthetic dataset for testing and debugging."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ununennium.core.geotensor import GeoTensor
from ununennium.datasets.base import GeoDataset


class SyntheticDataset(GeoDataset):
    """Synthetic dataset for testing and development.

    Generates random imagery with configurable properties.

    Example:
        >>> dataset = SyntheticDataset(
        ...     num_samples=1000,
        ...     num_channels=12,
        ...     image_size=256,
        ...     num_classes=10,
        ... )
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_channels: int = 12,
        image_size: tuple[int, int] | int = 256,
        num_classes: int = 10,
        task: str = "segmentation",
        seed: int | None = None,
    ):
        """Initialize synthetic dataset.

        Args:
            num_samples: Number of samples in the dataset.
            num_channels: Number of spectral bands.
            image_size: Spatial dimensions (H, W) or single value for square.
            num_classes: Number of segmentation/classification classes.
            task: Task type ('segmentation' or 'classification').
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.num_channels = num_channels

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        self._num_classes = num_classes
        self.task = task

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[GeoTensor, torch.Tensor]:
        # Generate random image
        image = torch.randn(self.num_channels, self.image_size[0], self.image_size[1])

        # Generate label based on task
        if self.task == "segmentation":
            label = torch.randint(0, self._num_classes, (self.image_size[0], self.image_size[1]))
        elif self.task == "classification":
            label = torch.randint(0, self._num_classes, (1,)).squeeze()
        else:
            raise ValueError(f"Unknown task: {self.task}")

        geotensor = GeoTensor(
            data=image,
            crs=None,
            transform=None,
        )

        return geotensor, label

    @property
    def crs(self) -> Any:
        return None

    @property
    def num_classes(self) -> int:
        return self._num_classes
