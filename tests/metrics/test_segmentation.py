"""Tests for metrics."""

import pytest
import torch

from ununennium.metrics import iou_score, dice_score, pixel_accuracy


class TestMetrics:
    def test_iou_score(self):
        pred = torch.randint(0, 5, (2, 64, 64))
        target = torch.randint(0, 5, (2, 64, 64))
        iou = iou_score(pred, target, num_classes=5)
        assert iou.shape == (5,)

    def test_dice_score(self):
        pred = torch.randint(0, 5, (2, 64, 64))
        target = torch.randint(0, 5, (2, 64, 64))
        dice = dice_score(pred, target, num_classes=5)
        assert dice.shape == (5,)

    def test_pixel_accuracy(self):
        pred = torch.randint(0, 5, (2, 64, 64))
        acc = pixel_accuracy(pred, pred)
        assert acc == 1.0
