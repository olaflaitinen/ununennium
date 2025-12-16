"""Tests for detection metrics."""

from __future__ import annotations

import pytest
import torch

from ununennium.metrics import (
    average_precision_at_iou,
    compute_iou_boxes,
    mean_average_precision,
)


class TestComputeIoUBoxes:
    """Tests for compute_iou_boxes function."""

    def test_identical_boxes(self):
        """Test IoU of identical boxes is 1."""
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        iou = compute_iou_boxes(boxes, boxes)

        assert iou.shape == (1, 1)
        assert iou[0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_no_overlap(self):
        """Test IoU of non-overlapping boxes is 0."""
        boxes1 = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
        boxes2 = torch.tensor([[10.0, 10.0, 15.0, 15.0]])

        iou = compute_iou_boxes(boxes1, boxes2)

        assert iou[0, 0].item() == pytest.approx(0.0, abs=1e-6)

    def test_partial_overlap(self):
        """Test IoU of partially overlapping boxes."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        boxes2 = torch.tensor([[5.0, 5.0, 15.0, 15.0]])

        iou = compute_iou_boxes(boxes1, boxes2)

        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        expected_iou = 25 / 175
        assert iou[0, 0].item() == pytest.approx(expected_iou, abs=1e-5)

    def test_pairwise_computation(self):
        """Test pairwise IoU computation."""
        boxes1 = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        boxes2 = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 15.0],
            [20.0, 20.0, 30.0, 30.0],
        ])

        iou = compute_iou_boxes(boxes1, boxes2)

        assert iou.shape == (2, 3)
        # boxes1[0] vs boxes2[0] should be 1
        assert iou[0, 0].item() == pytest.approx(1.0, abs=1e-6)
        # boxes1[1] vs boxes2[2] should be 1
        assert iou[1, 2].item() == pytest.approx(1.0, abs=1e-6)


class TestAveragePrecisionAtIoU:
    """Tests for average_precision_at_iou function."""

    def test_perfect_predictions(self):
        """Test AP with perfect predictions."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        pred_scores = torch.tensor([0.9, 0.8])
        pred_labels = torch.tensor([0, 0])
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        gt_labels = torch.tensor([0, 0])

        result = average_precision_at_iou(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5
        )

        assert "mAP" in result
        assert result["mAP"] == pytest.approx(1.0, abs=0.1)

    def test_no_predictions(self):
        """Test AP with no predictions."""
        pred_boxes = torch.zeros(0, 4)
        pred_scores = torch.zeros(0)
        pred_labels = torch.zeros(0, dtype=torch.long)
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt_labels = torch.tensor([0])

        result = average_precision_at_iou(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        )

        assert result["mAP"] == 0.0

    def test_no_ground_truth(self):
        """Test AP with no ground truth (should be 0 for false positives)."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        gt_boxes = torch.zeros(0, 4)
        gt_labels = torch.zeros(0, dtype=torch.long)

        result = average_precision_at_iou(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=1
        )

        # All predictions are false positives
        assert result["mAP"] == 0.0


class TestMeanAveragePrecision:
    """Tests for mean_average_precision function."""

    def test_basic_computation(self):
        """Test mAP computation."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        pred_scores = torch.tensor([0.9, 0.8])
        pred_labels = torch.tensor([0, 0])
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
        gt_labels = torch.tensor([0, 0])

        result = mean_average_precision(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        )

        assert "mAP" in result
        assert "AP50" in result
        assert "AP75" in result
        assert result["mAP"] >= 0 and result["mAP"] <= 1.0 + 1e-6

    def test_custom_iou_thresholds(self):
        """Test with custom IoU thresholds."""
        pred_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        pred_scores = torch.tensor([0.9])
        pred_labels = torch.tensor([0])
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt_labels = torch.tensor([0])

        result = mean_average_precision(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels,
            iou_thresholds=[0.3, 0.5, 0.7]
        )

        assert "mAP@0.30" in result
        assert "mAP@0.50" in result
        assert "mAP@0.70" in result

    def test_multiclass(self):
        """Test with multiple classes."""
        pred_boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        pred_scores = torch.tensor([0.9, 0.8])
        pred_labels = torch.tensor([0, 1])  # Two different classes
        gt_boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
        ])
        gt_labels = torch.tensor([0, 1])

        result = mean_average_precision(
            pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, num_classes=2
        )

        assert "mAP" in result
