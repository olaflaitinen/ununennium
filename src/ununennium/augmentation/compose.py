"""Compose multiple augmentations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
