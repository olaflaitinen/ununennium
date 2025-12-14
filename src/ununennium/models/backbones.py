"""Backbone networks for feature extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn


class Backbone(ABC, nn.Module):
    """Abstract base class for feature extraction backbones.

    Backbones extract multi-scale features from input images for
    downstream tasks like segmentation or detection.
    """

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Number of output channels at each feature level."""
        ...

    @property
    @abstractmethod
    def out_strides(self) -> list[int]:
        """Downsampling stride at each feature level."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature maps at different scales.
        """
        ...


class ResNetBackbone(Backbone):
    """ResNet backbone with configurable depth and input channels.

    Supports ResNet-18, 34, 50, 101, and 152 variants.
    """

    CONFIGS: ClassVar[dict] = {
        "resnet18": {"layers": [2, 2, 2, 2], "expansion": 1},
        "resnet34": {"layers": [3, 4, 6, 3], "expansion": 1},
        "resnet50": {"layers": [3, 4, 6, 3], "expansion": 4},
        "resnet101": {"layers": [3, 4, 23, 3], "expansion": 4},
        "resnet152": {"layers": [3, 8, 36, 3], "expansion": 4},
    }

    def __init__(
        self,
        variant: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet50",
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        """Initialize ResNet backbone.

        Args:
            variant: ResNet variant to use.
            in_channels: Number of input channels.
            pretrained: Whether to load pretrained weights (for 3-channel input).
        """
        super().__init__()
        self.variant = variant
        self.in_channels = in_channels

        config = self.CONFIGS[variant]
        expansion = config["expansion"]

        # Base channels for each stage
        base_channels = [64, 128, 256, 512]
        self._out_channels = [c * expansion for c in base_channels]
        self._out_strides = [4, 8, 16, 32]

        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build stages
        self.layer1 = self._make_layer(64, base_channels[0], config["layers"][0], expansion)
        self.layer2 = self._make_layer(
            base_channels[0] * expansion, base_channels[1], config["layers"][1], expansion, stride=2
        )
        self.layer3 = self._make_layer(
            base_channels[1] * expansion, base_channels[2], config["layers"][2], expansion, stride=2
        )
        self.layer4 = self._make_layer(
            base_channels[2] * expansion, base_channels[3], config["layers"][3], expansion, stride=2
        )

        # Load pretrained weights
        if pretrained and in_channels == 3:
            self._load_pretrained()

    def _make_layer(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        expansion: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a ResNet stage."""
        layers = []

        # First block may downsample
        if expansion == 1:
            layers.append(BasicBlock(in_planes, planes, stride))
        else:
            layers.append(Bottleneck(in_planes, planes, stride, expansion))

        # Remaining blocks
        for _ in range(1, num_blocks):
            if expansion == 1:
                layers.append(BasicBlock(planes * expansion, planes, 1))
            else:
                layers.append(Bottleneck(planes * expansion, planes, 1, expansion))

        return nn.Sequential(*layers)

    def _load_pretrained(self) -> None:
        """Load pretrained ImageNet weights."""
        from torchvision import models  # noqa: PLC0415

        pretrained_models = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
        }

        pretrained = pretrained_models[self.variant](weights="IMAGENET1K_V1")
        self.load_state_dict(pretrained.state_dict(), strict=False)

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    @property
    def out_strides(self) -> list[int]:
        return self._out_strides

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return [c1, c2, c3, c4]


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1, expansion: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class EfficientNetBackbone(Backbone):
    """EfficientNet backbone using timm."""

    def __init__(
        self,
        variant: str = "efficientnet_b0",
        in_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            import timm  # noqa: PLC0415  # noqa: PLC0415
        except ImportError as err:
            raise ImportError(
                "timm is required for EfficientNet. Install with: pip install timm"
            ) from err

        self.model: Any = timm.create_model(
            variant,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        self._out_channels: list[int] = self.model.feature_info.channels()
        self._out_strides: list[int] = self.model.feature_info.reduction()

    @property
    def out_channels(self) -> list[int]:
        return self._out_channels

    @property
    def out_strides(self) -> list[int]:
        return self._out_strides

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.model(x)  # type: ignore[return-value]
