"""Compose multiple augmentations."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
