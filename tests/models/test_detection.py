"""Tests for object detection models."""

from __future__ import annotations

import pytest
import torch

from ununennium.models import (
    FCOS,
    FasterRCNN,
    RetinaNet,
    DetectionHead,
    create_model,
    list_models,
)


class TestDetectionHead:
    """Tests for DetectionHead."""

    def test_forward_shape(self):
        """Test output shapes."""
        head = DetectionHead(in_channels=256, num_classes=10, num_anchors=9)
        features = [
            torch.randn(2, 256, 32, 32),
            torch.randn(2, 256, 16, 16),
            torch.randn(2, 256, 8, 8),
        ]

        cls_logits, box_regs = head(features)

        assert len(cls_logits) == 3
        assert len(box_regs) == 3
        # Check classification output: B x (A*C) x H x W
        assert cls_logits[0].shape == (2, 9 * 10, 32, 32)
        # Check regression output: B x (A*4) x H x W
        assert box_regs[0].shape == (2, 9 * 4, 32, 32)

    def test_different_anchor_counts(self):
        """Test with different number of anchors."""
        head = DetectionHead(in_channels=128, num_classes=5, num_anchors=3)
        features = [torch.randn(1, 128, 16, 16)]

        cls_logits, box_regs = head(features)

        assert cls_logits[0].shape == (1, 3 * 5, 16, 16)
        assert box_regs[0].shape == (1, 3 * 4, 16, 16)


class TestRetinaNet:
    """Tests for RetinaNet detector."""

    def test_forward_shape(self):
        """Test output structure."""
        model = RetinaNet(in_channels=3, num_classes=5, backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert hasattr(output, "class_logits")
        assert hasattr(output, "box_regression")
        assert len(output.class_logits) > 0
        assert len(output.box_regression) > 0

    def test_multispectral_input(self):
        """Test with 12-band satellite imagery input."""
        model = RetinaNet(in_channels=12, num_classes=10, backbone="resnet18", pretrained=False)
        x = torch.randn(1, 12, 256, 256)

        output = model(x)

        assert len(output.class_logits) > 0
        # Should have multiple FPN levels
        assert len(output.class_logits) >= 4

    def test_registry_creation(self):
        """Test creating RetinaNet through registry."""
        # Ensure module is imported to register models
        from ununennium.models.architectures import detection  # noqa: F401

        models = list_models()
        assert "retinanet" in models

        model = create_model("retinanet", in_channels=3, num_classes=5, backbone="resnet18", pretrained=False)
        assert model is not None


class TestFasterRCNN:
    """Tests for Faster R-CNN detector."""

    def test_forward_shape(self):
        """Test output structure."""
        model = FasterRCNN(in_channels=3, num_classes=5, backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert hasattr(output, "class_logits")
        assert hasattr(output, "box_regression")

    def test_multispectral_input(self):
        """Test with satellite imagery input."""
        model = FasterRCNN(in_channels=12, num_classes=20, backbone="resnet18", pretrained=False)
        x = torch.randn(1, 12, 512, 512)

        output = model(x)

        assert len(output.class_logits) > 0


class TestFCOS:
    """Tests for FCOS anchor-free detector."""

    def test_forward_shape(self):
        """Test output structure."""
        model = FCOS(in_channels=3, num_classes=10, backbone="resnet18", pretrained=False)
        x = torch.randn(2, 3, 256, 256)

        output = model(x)

        assert hasattr(output, "class_logits")
        assert hasattr(output, "box_regression")
        # FCOS outputs per-pixel predictions (no anchors dimension)
        for cls_out in output.class_logits:
            assert cls_out.shape[1] == 10  # num_classes

    def test_multispectral_input(self):
        """Test with 12-band input."""
        model = FCOS(in_channels=12, num_classes=5, backbone="resnet18", pretrained=False)
        x = torch.randn(1, 12, 256, 256)

        output = model(x)

        assert len(output.class_logits) > 0
        for box_reg in output.box_regression:
            assert box_reg.shape[1] == 4  # l, t, r, b


class TestDetectionModelRegistry:
    """Tests for detection model registry."""

    def test_all_detection_models_registered(self):
        """Test that all detection models are in registry."""
        from ununennium.models.architectures import detection  # noqa: F401

        models = list_models()
        assert "retinanet" in models
        assert "fasterrcnn" in models
        assert "fcos" in models

    def test_output_keys_method(self):
        """Test DetectionOutput.keys() method."""
        model = RetinaNet(in_channels=3, num_classes=5, backbone="resnet18", pretrained=False)
        x = torch.randn(1, 3, 128, 128)

        output = model(x)
        keys = output.keys()

        assert "class_logits" in keys
        assert "box_regression" in keys
