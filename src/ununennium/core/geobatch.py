"""GeoBatch: Batched container for training with geospatial data."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11):
    pass
else:
    pass

import torch

from ununennium.core.geotensor import GeoTensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ununennium.core.types import Device


@dataclass
class GeoBatch:
    """A batch of geospatial images and labels for ML training.

    GeoBatch provides a structured container for training data that
    preserves geospatial metadata while being compatible with
    PyTorch DataLoader.

    Attributes:
        images: Batched image tensor, shape (B, C, H, W).
        labels: Optional labels (segmentation masks, class indices, etc.).
        metadata: Optional per-sample metadata dictionaries.
        crs: CRS of all samples (assumes homogeneous CRS).
        transforms: Per-sample affine transforms.

    Example:
        >>> batch = GeoBatch(
        ...     images=torch.randn(16, 12, 256, 256),
        ...     labels=torch.randint(0, 10, (16, 256, 256)),
        ... )
        >>> batch = batch.cuda()
        >>> predictions = model(batch.images)
    """

    images: torch.Tensor
    labels: torch.Tensor | None = None
    metadata: list[dict[str, Any]] | None = None
    crs: Any | None = None
    transforms: list[Any] | None = None

    def __post_init__(self) -> None:
        """Validate batch structure."""
        if self.images.ndim != 4:
            raise ValueError(f"images must be 4D (B, C, H, W), got {self.images.ndim}D")

        if self.labels is not None and self.labels.shape[0] != self.images.shape[0]:
            raise ValueError(
                f"Batch size mismatch: images={self.images.shape[0]}, labels={self.labels.shape[0]}"
            )

        if self.metadata is not None and len(self.metadata) != self.images.shape[0]:
            raise ValueError(
                f"Metadata length ({len(self.metadata)}) must match "
                f"batch size ({self.images.shape[0]})"
            )

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self.images.shape[0]

    @property
    def num_channels(self) -> int:
        """Number of channels in the images."""
        return self.images.shape[1]

    @property
    def height(self) -> int:
        """Height of the images."""
        return self.images.shape[2]

    @property
    def width(self) -> int:
        """Width of the images."""
        return self.images.shape[3]

    @property
    def device(self) -> torch.device:
        """Device where the batch resides."""
        return self.images.device

    def to(self, device: Device) -> GeoBatch:
        """Move batch to specified device.

        Args:
            device: Target device.

        Returns:
            New GeoBatch on target device.
        """
        return self.__class__(
            images=self.images.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            metadata=self.metadata,
            crs=self.crs,
            transforms=self.transforms,
        )

    def cuda(self, device: int = 0) -> GeoBatch:
        """Move batch to CUDA."""
        return self.to(f"cuda:{device}")

    def cpu(self) -> GeoBatch:
        """Move batch to CPU."""
        return self.to("cpu")

    def float(self) -> GeoBatch:
        """Convert images to float32."""
        return self.__class__(
            images=self.images.float(),
            labels=self.labels,
            metadata=self.metadata,
            crs=self.crs,
            transforms=self.transforms,
        )

    def half(self) -> GeoBatch:
        """Convert images to float16."""
        return self.__class__(
            images=self.images.half(),
            labels=self.labels,
            metadata=self.metadata,
            crs=self.crs,
            transforms=self.transforms,
        )

    def __len__(self) -> int:
        """Return batch size."""
        return self.batch_size

    def __getitem__(self, idx: int) -> GeoTensor:
        """Get a single sample as GeoTensor.

        Args:
            idx: Sample index.

        Returns:
            GeoTensor for the sample.
        """
        transform = self.transforms[idx] if self.transforms else None
        return GeoTensor(
            data=self.images[idx],
            crs=self.crs,
            transform=transform,
        )

    def __iter__(self) -> Iterator[GeoTensor]:
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def collate(cls, samples: list[tuple[GeoTensor, Any]]) -> GeoBatch:
        """Collate function for PyTorch DataLoader.

        Args:
            samples: List of (image, label) tuples.

        Returns:
            Batched GeoBatch.
        """
        images = []
        labels = []
        transforms = []
        crs = None

        for geotensor, label in samples:
            images.append(geotensor.data)
            labels.append(label if label is not None else torch.tensor([]))
            transforms.append(geotensor.transform)
            if crs is None:
                crs = geotensor.crs

        images_tensor = torch.stack(images)

        # Handle labels
        labels_tensor = torch.stack(labels) if all(label.numel() > 0 for label in labels) else None

        return cls(
            images=images_tensor,
            labels=labels_tensor,
            crs=crs,
            transforms=transforms,
        )

    def __repr__(self) -> str:
        label_info = f", labels={self.labels.shape}" if self.labels is not None else ""
        return f"GeoBatch(images={self.images.shape}{label_info}, device={self.device})"
