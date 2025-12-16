"""Task-specific heads for different remote sensing tasks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ClassificationHead(nn.Module):
    """Global pooling + linear classifier head.

    Suitable for scene classification tasks.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.0,
        pooling: str = "avg",
    ):
        """Initialize classification head.

        Args:
            in_channels: Number of input channels from backbone.
            num_classes: Number of output classes.
            dropout: Dropout probability before final linear layer.
            pooling: Pooling type ('avg' or 'max').
        """
        super().__init__()

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from backbone (uses last one).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = features[-1]
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


class SegmentationHead(nn.Module):
    """U-Net style decoder for semantic segmentation.

    Uses skip connections from encoder features.
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int] | None = None,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        """Initialize segmentation head.

        Args:
            encoder_channels: Channels at each encoder level (from backbone).
            decoder_channels: Channels at each decoder level.
                If None, uses [256, 128, 64, 32].
            num_classes: Number of segmentation classes.
            dropout: Dropout probability in decoder blocks.
        """
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]

        # Ensure we have enough decoder channels
        n_stages = len(encoder_channels)
        if len(decoder_channels) < n_stages:
            decoder_channels = decoder_channels + [decoder_channels[-1]] * (
                n_stages - len(decoder_channels)
            )

        self.blocks = nn.ModuleList()
        in_ch = encoder_channels[-1]

        for _i, (enc_ch, dec_ch) in enumerate(
            zip(reversed(encoder_channels[:-1]), decoder_channels, strict=False)
        ):
            self.blocks.append(DecoderBlock(in_ch + enc_ch, dec_ch, dropout=dropout))
            in_ch = dec_ch

        self.final_conv = nn.Conv2d(decoder_channels[len(encoder_channels) - 2], num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from backbone
                (low to high resolution).

        Returns:
            Segmentation logits of shape (B, num_classes, H', W').
        """
        x = features[-1]

        for i, block in enumerate(self.blocks):
            skip = features[-(i + 2)]
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.final_conv(x)


class DecoderBlock(nn.Module):
    """Single decoder block with two convolutions."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class FPNHead(nn.Module):
    """Feature Pyramid Network head for multi-scale segmentation."""

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        num_classes: int = 10,
    ):
        """Initialize FPN head.

        Args:
            in_channels: Channels at each feature level.
            out_channels: Channels in the FPN.
            num_classes: Number of segmentation classes.
        """
        super().__init__()

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ch, out_channels, 1) for ch in in_channels])

        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

        self.final_conv = nn.Conv2d(out_channels * len(in_channels), num_classes, 1)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: List of feature maps from backbone.

        Returns:
            Segmentation logits.
        """
        # Lateral connections
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features, strict=False)]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )

        # Output convolutions
        outputs = [conv(lat) for conv, lat in zip(self.output_convs, laterals, strict=False)]

        # Upsample all to highest resolution
        target_size = outputs[0].shape[2:]
        outputs = [
            F.interpolate(o, size=target_size, mode="bilinear", align_corners=False)
            for o in outputs
        ]

        # Concatenate and final conv
        return self.final_conv(torch.cat(outputs, dim=1))


class DetectionHead(nn.Module):
    """Detection head for object detection tasks.

    Outputs classification scores and bounding box regressions for
    anchor-based detectors (RetinaNet, FasterRCNN) or center-based
    detectors (FCOS).

    The head applies shared convolutions followed by separate
    classification and regression branches.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 9,
        num_convs: int = 4,
        prior_prob: float = 0.01,
    ):
        """Initialize detection head.

        Args:
            in_channels: Number of input channels from FPN.
            num_classes: Number of object classes (excluding background).
            num_anchors: Number of anchors per spatial location.
            num_convs: Number of convolution layers in each branch.
            prior_prob: Prior probability for focal loss initialization.
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Classification branch
        cls_convs = []
        for _ in range(num_convs):
            cls_convs.extend(
                [
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.cls_convs = nn.Sequential(*cls_convs)
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)

        # Regression branch
        reg_convs = []
        for _ in range(num_convs):
            reg_convs.extend(
                [
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.reg_convs = nn.Sequential(*reg_convs)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Initialize classification bias with prior probability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.cls_pred.bias is not None:
            nn.init.constant_(self.cls_pred.bias, bias_value)
        nn.init.normal_(self.reg_pred.weight, std=0.01)
        if self.reg_pred.bias is not None:
            nn.init.constant_(self.reg_pred.bias, 0)

    def forward(
        self, features: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass.

        Args:
            features: List of feature maps from FPN at different scales.

        Returns:
            Tuple of (class_logits, box_regressions) where each is a list
            of tensors corresponding to each feature level.
            - class_logits[i]: (B, num_anchors * num_classes, H_i, W_i)
            - box_regressions[i]: (B, num_anchors * 4, H_i, W_i)
        """
        class_logits = []
        box_regressions = []

        for feature in features:
            cls_feat = self.cls_convs(feature)
            cls_logit = self.cls_pred(cls_feat)
            class_logits.append(cls_logit)

            reg_feat = self.reg_convs(feature)
            box_reg = self.reg_pred(reg_feat)
            box_regressions.append(box_reg)

        return class_logits, box_regressions
