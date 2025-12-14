"""Tests for augmentation."""

import pytest
import torch

from ununennium.augmentation import RandomFlip, RandomRotate, Compose


class TestAugmentation:
    def test_random_flip(self):
        aug = RandomFlip(h_flip_p=1.0, v_flip_p=0.0)
        image = torch.arange(12).reshape(1, 3, 4).float()
        flipped, _ = aug(image, None)
        assert flipped.shape == image.shape

    def test_compose(self):
        augs = Compose([RandomFlip(), RandomRotate()])
        image = torch.randn(3, 64, 64)
        result, _ = augs(image, None)
        assert result.shape == image.shape
