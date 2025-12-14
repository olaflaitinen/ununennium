"""Tests for losses."""

import pytest
import torch

from ununennium.losses import DiceLoss, FocalLoss


class TestLosses:
    def test_dice_loss(self):
        loss_fn = DiceLoss()
        pred = torch.randn(2, 10, 64, 64)
        target = torch.randint(0, 10, (2, 64, 64))
        loss = loss_fn(pred, target)
        assert loss.item() >= 0

    def test_focal_loss(self):
        loss_fn = FocalLoss()
        pred = torch.randn(2, 10, 64, 64)
        target = torch.randint(0, 10, (2, 64, 64))
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
