"""Tests for model architectures."""

from __future__ import annotations

import pytest
import torch

from ununennium.models import create_model, list_models
from ununennium.models.backbones import ResNetBackbone, BasicBlock, Bottleneck
from ununennium.models.heads import ClassificationHead, SegmentationHead


class TestResNetBackbone:
    """Tests for ResNet backbone."""

    @pytest.mark.parametrize("variant", ["resnet18", "resnet34", "resnet50"])
    def test_forward_shape(self, variant):
        """Test output shapes for different variants."""
        backbone = ResNetBackbone(variant=variant, in_channels=3, pretrained=False)
        x = torch.randn(2, 3, 224, 224)

        features = backbone(x)

        assert len(features) == 4
        assert all(isinstance(f, torch.Tensor) for f in features)
        # Check spatial dimensions reduce by factors of 4, 8, 16, 32
        for i, f in enumerate(features):
            expected_size = 224 // (4 * (2 ** i))
            assert f.shape[2] == expected_size

    def test_multispectral_input(self):
        """Test with multi-spectral (12-band) input."""
        backbone = ResNetBackbone(
            variant="resnet50", in_channels=12, pretrained=False
        )
        x = torch.randn(2, 12, 256, 256)

        features = backbone(x)

        assert len(features) == 4
        assert features[0].shape[1] == 256  # ResNet50 channel count

    def test_out_channels(self):
        """Test out_channels property."""
        backbone = ResNetBackbone(variant="resnet50", pretrained=False)

        assert backbone.out_channels == [256, 512, 1024, 2048]
        assert backbone.out_strides == [4, 8, 16, 32]


class TestClassificationHead:
    """Tests for classification head."""

    def test_forward_shape(self):
        """Test output shape."""
        head = ClassificationHead(in_channels=512, num_classes=10)
        features = [torch.randn(2, 512, 7, 7)]

        output = head(features)

        assert output.shape == (2, 10)

    def test_with_dropout(self):
        """Test with dropout."""
        head = ClassificationHead(in_channels=512, num_classes=10, dropout=0.5)
        head.train()
        features = [torch.randn(2, 512, 7, 7)]

        output = head(features)
        assert output.shape == (2, 10)


class TestSegmentationHead:
    """Tests for segmentation head."""

    def test_forward_shape(self):
        """Test output shape with multi-scale features."""
        encoder_channels = [256, 512, 1024, 2048]
        head = SegmentationHead(
            encoder_channels=encoder_channels, num_classes=10
        )

        # Create features at different scales
        features = [
            torch.randn(2, 256, 64, 64),
            torch.randn(2, 512, 32, 32),
            torch.randn(2, 1024, 16, 16),
            torch.randn(2, 2048, 8, 8),
        ]

        output = head(features)

        # Output should be at highest resolution feature level
        assert output.shape[:2] == (2, 10)


class TestModelRegistry:
    """Tests for model registry."""

    def test_list_models(self):
        """Test listing available models."""
        # Import architectures to register them
        from ununennium.models.architectures import unet  # noqa: F401

        models = list_models()
        assert "unet" in models

    def test_create_model(self):
        """Test creating a model by name."""
        from ununennium.models.architectures import unet  # noqa: F401

        model = create_model("unet", in_channels=12, num_classes=10)

        assert model is not None
        x = torch.randn(2, 12, 256, 256)
        output = model(x)
        assert output.shape == (2, 10, 256, 256)

    def test_unknown_model(self):
        """Test that unknown model raises KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            create_model("nonexistent_model")
