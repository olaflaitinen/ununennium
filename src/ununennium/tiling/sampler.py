"""Spatial sampling strategies for training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from ununennium.core.bounds import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Iterator


class Sampler(ABC):
    """Abstract base class for spatial samplers."""

    @abstractmethod
    def __iter__(self) -> Iterator[BoundingBox | tuple[int, int]]:
        """Iterate over sample locations."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples."""
        ...


class RandomSampler(Sampler):
    """Random spatial sampling within bounds."""

    def __init__(
        self,
        bounds: BoundingBox,
        sample_size: tuple[float, float],
        num_samples: int,
        seed: int | None = None,
    ):
        """Initialize random sampler.

        Args:
            bounds: Bounding box to sample within.
            sample_size: Size of each sample (width, height) in CRS units.
            num_samples: Number of samples to generate.
            seed: Random seed for reproducibility.
        """
        self.bounds = bounds
        self.sample_size = sample_size
        self.num_samples = num_samples

        if seed is not None:
            torch.manual_seed(seed)

    def __iter__(self) -> Iterator[BoundingBox]:
        for _ in range(self.num_samples):
            # Random center point
            x = torch.rand(1).item() * (self.bounds.width - self.sample_size[0]) + self.bounds.minx
            y = torch.rand(1).item() * (self.bounds.height - self.sample_size[1]) + self.bounds.miny

            yield BoundingBox(
                minx=x,
                miny=y,
                maxx=x + self.sample_size[0],
                maxy=y + self.sample_size[1],
            )

    def __len__(self) -> int:
        return self.num_samples


class GridSampler(Sampler):
    """Regular grid sampling."""

    def __init__(
        self,
        bounds: BoundingBox,
        sample_size: tuple[float, float],
        stride: tuple[float, float] | None = None,
    ):
        """Initialize grid sampler.

        Args:
            bounds: Bounding box to sample within.
            sample_size: Size of each sample (width, height).
            stride: Step size between samples. Defaults to sample_size.
        """
        self.bounds = bounds
        self.sample_size = sample_size
        self.stride = stride or sample_size

        # Calculate grid dimensions
        self.nx = int((bounds.width - sample_size[0]) / self.stride[0]) + 1
        self.ny = int((bounds.height - sample_size[1]) / self.stride[1]) + 1

    def __iter__(self) -> Iterator[BoundingBox]:
        for iy in range(self.ny):
            for ix in range(self.nx):
                x = self.bounds.minx + ix * self.stride[0]
                y = self.bounds.miny + iy * self.stride[1]

                yield BoundingBox(
                    minx=x,
                    miny=y,
                    maxx=x + self.sample_size[0],
                    maxy=y + self.sample_size[1],
                )

    def __len__(self) -> int:
        return self.nx * self.ny


class BalancedSampler(Sampler):
    """Class-balanced spatial sampling.

    Samples locations to balance class distribution in batches.
    """

    def __init__(
        self,
        class_locations: dict[int, list[tuple[float, float]]],
        sample_size: tuple[float, float],
        num_samples: int,
        seed: int | None = None,
    ):
        """Initialize balanced sampler.

        Args:
            class_locations: Mapping from class ID to list of (x, y) centers.
            sample_size: Size of each sample.
            num_samples: Total number of samples.
            seed: Random seed.
        """
        self.class_locations = class_locations
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.classes = list(class_locations.keys())

        if seed is not None:
            torch.manual_seed(seed)

    def __iter__(self) -> Iterator[BoundingBox]:
        for i in range(self.num_samples):
            # Round-robin through classes
            cls = self.classes[i % len(self.classes)]
            locations = self.class_locations[cls]

            # Random location for this class
            idx = int(torch.randint(len(locations), (1,)).item())
            x, y = locations[idx]

            # Add jitter
            x += (torch.rand(1).item() - 0.5) * self.sample_size[0] * 0.5
            y += (torch.rand(1).item() - 0.5) * self.sample_size[1] * 0.5

            yield BoundingBox(
                minx=x - self.sample_size[0] / 2,
                miny=y - self.sample_size[1] / 2,
                maxx=x + self.sample_size[0] / 2,
                maxy=y + self.sample_size[1] / 2,
            )

    def __len__(self) -> int:
        return self.num_samples
