"""Geometric augmentations."""

from __future__ import annotations

import torch


class RandomFlip:
    """Random horizontal and vertical flip."""

    def __init__(self, h_flip_p: float = 0.5, v_flip_p: float = 0.5):
        self.h_flip_p = h_flip_p
        self.v_flip_p = v_flip_p

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if torch.rand(1).item() < self.h_flip_p:
            image = torch.flip(image, [-1])
            if mask is not None:
                mask = torch.flip(mask, [-1])

        if torch.rand(1).item() < self.v_flip_p:
            image = torch.flip(image, [-2])
            if mask is not None:
                mask = torch.flip(mask, [-2])

        return image, mask


class RandomRotate:
    """Random 90-degree rotations."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if torch.rand(1).item() < self.p:
            k = int(torch.randint(1, 4, (1,)).item())
            image = torch.rot90(image, k=k, dims=[-2, -1])
            if mask is not None:
                mask = torch.rot90(mask, k=k, dims=[-2, -1])

        return image, mask


class RandomCrop:
    """Random crop to specified size."""

    def __init__(self, size: int | tuple[int, int]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, h, w = image.shape
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Image size ({h}, {w}) smaller than crop size {self.size}")

        y = torch.randint(0, h - th + 1, (1,)).item()
        x = torch.randint(0, w - tw + 1, (1,)).item()

        image = image[:, y : y + th, x : x + tw]
        if mask is not None:
            mask = mask[y : y + th, x : x + tw]

        return image, mask
