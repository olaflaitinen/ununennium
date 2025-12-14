"""End-to-end integration tests."""

import pytest
import torch

from ununennium.models import create_model
from ununennium.models.architectures import unet  # noqa: F401
from ununennium.datasets import SyntheticDataset


class TestEndToEnd:
    def test_segmentation_pipeline(self):
        # Dataset
        ds = SyntheticDataset(num_samples=4, image_size=64, num_classes=5)

        # Model
        model = create_model("unet", in_channels=12, num_classes=5)

        # Forward pass
        image, label = ds[0]
        with torch.no_grad():
            output = model(image.data.unsqueeze(0))

        assert output.shape == (1, 5, 64, 64)
