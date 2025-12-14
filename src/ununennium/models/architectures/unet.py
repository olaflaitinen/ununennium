"""U-Net architecture for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn

from ununennium.models.backbones import EfficientNetBackbone, ResNetBackbone
from ununennium.models.heads import SegmentationHead
from ununennium.models.registry import register_model


@register_model("unet")
class UNet(nn.Module):
    """U-Net architecture for semantic segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        backbone: str = "resnet50",
        pretrained: bool = True,
        decoder_channels: list[int] | None = None,
    ):
        """Initialize U-Net.

        Args:
            in_channels: Number of input channels.
            num_classes: Number of output segmentation classes.
            backbone: Backbone architecture ('resnet18', 'resnet50', etc.).
            pretrained: Whether to use pretrained backbone weights.
            decoder_channels: Number of channels at each decoder level.
        """
        super().__init__()

        if backbone.startswith("efficientnet"):
            self.encoder = EfficientNetBackbone(
                variant=backbone,
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )
        else:
            self.encoder = ResNetBackbone(
                variant=backbone,
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )

        self.decoder = SegmentationHead(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
        )

        # Final upsampling to match input resolution
        self.upsample = nn.Upsample(
            scale_factor=self.encoder.out_strides[0],  # Typically 4 for ResNet, but varies
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Segmentation logits of shape (B, num_classes, H, W).
        """
        features = self.encoder(x)
        out = self.decoder(features)
        return self.upsample(out)


@register_model("unet_resnet18")
class UNetResNet18(UNet):
    """U-Net with ResNet-18 backbone."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone="resnet18",
            **kwargs,
        )


@register_model("unet_resnet50")
class UNetResNet50(UNet):
    """U-Net with ResNet-50 backbone."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10, **kwargs):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            backbone="resnet50",
            **kwargs,
        )
