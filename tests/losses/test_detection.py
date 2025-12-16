"""Tests for detection losses."""

from __future__ import annotations

import pytest
import torch

from ununennium.losses import (
    DetectionLoss,
    FocalLossDetection,
    GIoULoss,
    SmoothL1Loss,
)


class TestFocalLossDetection:
    """Tests for FocalLossDetection."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = FocalLossDetection(alpha=0.25, gamma=2.0)
        pred = torch.randn(10, 5)  # 10 samples, 5 classes
        target = torch.randint(0, 5, (10,))

        loss = loss_fn(pred, target)

        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_forward_with_spatial(self):
        """Test with spatial dimensions."""
        loss_fn = FocalLossDetection()
        pred = torch.randn(2, 5, 32, 32)  # B, C, H, W
        target = torch.randint(0, 5, (2, 32, 32))

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert loss >= 0

    def test_gamma_effect(self):
        """Test that higher gamma focuses more on hard examples."""
        pred = torch.tensor([[2.0, 0.0], [-2.0, 0.0]])  # Easy and hard example
        target = torch.tensor([0, 0])

        loss_low = FocalLossDetection(gamma=0.0)(pred, target)
        loss_high = FocalLossDetection(gamma=2.0)(pred, target)

        # Higher gamma should reduce loss for easy examples more
        assert loss_low.item() != loss_high.item()


class TestSmoothL1Loss:
    """Tests for SmoothL1Loss."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = SmoothL1Loss(beta=1.0)
        pred = torch.randn(10, 4)
        target = torch.randn(10, 4)

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert loss >= 0

    def test_with_weights(self):
        """Test with per-element weights."""
        loss_fn = SmoothL1Loss()
        pred = torch.randn(10, 4)
        target = torch.randn(10, 4)
        weights = torch.rand(10, 4)

        loss = loss_fn(pred, target, weights)

        assert loss.dim() == 0
        assert loss >= 0

    def test_zero_loss(self):
        """Test that identical predictions have zero loss."""
        loss_fn = SmoothL1Loss()
        pred = torch.randn(10, 4)

        loss = loss_fn(pred, pred)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestGIoULoss:
    """Tests for GIoULoss."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = GIoULoss()
        # Boxes in (x1, y1, x2, y2) format
        pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        target = torch.tensor([[1.0, 1.0, 11.0, 11.0], [5.0, 5.0, 15.0, 15.0]])

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert loss >= 0
        assert loss <= 2  # GIoU ranges from -1 to 1, so loss is 0 to 2

    def test_perfect_overlap(self):
        """Test that identical boxes have zero loss."""
        loss_fn = GIoULoss()
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])

        loss = loss_fn(boxes, boxes)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_no_overlap(self):
        """Test non-overlapping boxes."""
        loss_fn = GIoULoss()
        pred = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        target = torch.tensor([[10.0, 10.0, 15.0, 15.0]])

        loss = loss_fn(pred, target)

        # Should be > 1 for non-overlapping boxes
        assert loss.item() > 1


class TestDetectionLoss:
    """Tests for combined DetectionLoss."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        loss_fn = DetectionLoss()
        cls_pred = torch.randn(10, 5)
        cls_target = torch.randint(0, 5, (10,))
        reg_pred = torch.randn(10, 4)
        reg_target = torch.randn(10, 4)

        losses = loss_fn(cls_pred, cls_target, reg_pred, reg_target)

        assert "cls_loss" in losses
        assert "reg_loss" in losses
        assert "total_loss" in losses
        assert losses["total_loss"] >= 0

    def test_with_mask(self):
        """Test with regression mask."""
        loss_fn = DetectionLoss()
        cls_pred = torch.randn(10, 5)
        cls_target = torch.randint(0, 5, (10,))
        reg_pred = torch.randn(10, 4)
        reg_target = torch.randn(10, 4)
        reg_mask = torch.tensor([True, True, False, True, False, True, True, False, True, True])

        losses = loss_fn(cls_pred, cls_target, reg_pred, reg_target, reg_mask)

        assert losses["total_loss"] >= 0

    def test_custom_weights(self):
        """Test with custom loss weights."""
        loss_fn = DetectionLoss(cls_weight=2.0, reg_weight=0.5)
        cls_pred = torch.randn(10, 5)
        cls_target = torch.randint(0, 5, (10,))
        reg_pred = torch.randn(10, 4)
        reg_target = torch.randn(10, 4)

        losses = loss_fn(cls_pred, cls_target, reg_pred, reg_target)

        expected_total = 2.0 * losses["cls_loss"] + 0.5 * losses["reg_loss"]
        assert losses["total_loss"].item() == pytest.approx(expected_total.item(), rel=1e-5)
