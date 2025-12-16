"""Object detection architectures for remote sensing.

Implements popular detection models adapted for satellite imagery:
- RetinaNet: One-stage detector with Feature Pyramid Network and Focal Loss
- FasterRCNN: Two-stage detector with Region Proposal Network
- FCOS: Anchor-free Fully Convolutional One-Stage detector
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from ununennium.models.backbones import Backbone, EfficientNetBackbone, ResNetBackbone
from ununennium.models.heads import DetectionHead
from ununennium.models.registry import register_model


@dataclass
class DetectionOutput:
    """Container for detection model outputs.

    Attributes:
        class_logits: Classification logits per feature level.
        box_regression: Bounding box regression per feature level.
        proposals: Optional region proposals (for two-stage detectors).
        losses: Optional loss dictionary (during training).
    """

    class_logits: list[torch.Tensor] = field(default_factory=list)
    box_regression: list[torch.Tensor] = field(default_factory=list)
    proposals: list[torch.Tensor] | None = None
    losses: dict[str, torch.Tensor] | None = None

    def keys(self) -> list[str]:
        """Return available output keys for compatibility."""
        keys = ["class_logits", "box_regression"]
        if self.proposals is not None:
            keys.append("proposals")
        if self.losses is not None:
            keys.append("losses")
        return keys


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction.

    Takes multi-scale features from backbone and creates top-down
    pathway with lateral connections for object detection.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 256,
        extra_levels: int = 2,
    ):
        """Initialize FPN.

        Args:
            in_channels: Channels at each backbone level.
            out_channels: Number of output channels at each FPN level.
            extra_levels: Number of additional levels beyond backbone.
        """
        super().__init__()

        self.out_channels = out_channels
        self.num_levels = len(in_channels) + extra_levels

        # Lateral connections
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])

        # Output convolutions
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

        # Extra levels (P6, P7 for RetinaNet)
        self.extra_convs = nn.ModuleList()
        for i in range(extra_levels):
            if i == 0:
                self.extra_convs.append(
                    nn.Conv2d(in_channels[-1], out_channels, 3, stride=2, padding=1)
                )
            else:
                self.extra_convs.append(
                    nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: Multi-scale features from backbone.

        Returns:
            FPN features at all levels (P3-P7 typically).
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

        # Extra levels
        extra_input = features[-1]
        for i, conv in enumerate(self.extra_convs):
            if i == 0:
                outputs.append(conv(extra_input))
            else:
                outputs.append(conv(F.relu(outputs[-1])))

        return outputs


class AnchorGenerator(nn.Module):
    """Generate anchors for object detection.

    Creates anchor boxes at each spatial location across all FPN levels.
    """

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
        scales: tuple[float, ...] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ):
        """Initialize anchor generator.

        Args:
            sizes: Base anchor sizes for each FPN level.
            aspect_ratios: Anchor aspect ratios (height/width).
            scales: Anchor scale multipliers.
        """
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.scales = scales

    @property
    def num_anchors_per_location(self) -> int:
        """Number of anchors per spatial location."""
        return len(self.aspect_ratios) * len(self.scales)

    def forward(
        self,
        features: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        """Generate anchors for all feature levels.

        Args:
            features: Feature maps from FPN.
            image_size: (height, width) of input image.

        Returns:
            List of anchor tensors, one per feature level.
            Each tensor has shape (H_i * W_i * A, 4) where A is num_anchors.
        """
        device = features[0].device
        dtype = features[0].dtype
        anchors_per_level = []

        for level_idx, feature in enumerate(features):
            _, _, h, w = feature.shape
            size_idx = min(level_idx, len(self.sizes) - 1)
            base_size = self.sizes[size_idx][0]

            # Generate anchor boxes
            anchors = []
            for scale in self.scales:
                for ratio in self.aspect_ratios:
                    # Compute anchor dimensions
                    anchor_size = base_size * scale
                    anchor_h = anchor_size * (ratio**0.5)
                    anchor_w = anchor_size / (ratio**0.5)

                    anchors.append(
                        [
                            -anchor_w / 2,
                            -anchor_h / 2,
                            anchor_w / 2,
                            anchor_h / 2,
                        ]
                    )

            base_anchors = torch.tensor(anchors, device=device, dtype=dtype)

            # Compute stride
            stride_h = image_size[0] / h
            stride_w = image_size[1] / w

            # Generate grid
            shifts_x = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride_w
            shifts_y = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride_h
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=-1)
            shifts = shifts.reshape(-1, 1, 4)

            # Combine with base anchors
            level_anchors = (shifts + base_anchors.unsqueeze(0)).reshape(-1, 4)
            anchors_per_level.append(level_anchors)

        return anchors_per_level


@register_model("retinanet")
class RetinaNet(nn.Module):
    """RetinaNet: One-stage object detector with Focal Loss.

    Architecture:
        Backbone (ResNet/EfficientNet) -> FPN -> Detection Head

    RetinaNet uses a Feature Pyramid Network for multi-scale detection
    and Focal Loss to handle class imbalance in dense detection.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = True,
        fpn_channels: int = 256,
        num_anchors: int = 9,
    ):
        """Initialize RetinaNet.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of object classes (excluding background).
            backbone: Backbone architecture name.
            pretrained: Whether to use pretrained backbone.
            fpn_channels: Number of channels in FPN.
            num_anchors: Number of anchors per location.
        """
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        if backbone.startswith("efficientnet"):
            self.backbone: Backbone = EfficientNetBackbone(
                variant=backbone,
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )
        else:
            self.backbone = ResNetBackbone(
                variant=backbone,  # type: ignore
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )

        # FPN
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
            extra_levels=2,
        )

        # Detection head
        self.head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_anchors=num_anchors,
        )

        # Anchor generator
        self.anchor_generator = AnchorGenerator()

    def forward(
        self,
        x: torch.Tensor,
        _targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> DetectionOutput:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).
            targets: Optional list of target dictionaries for training.
                Each dict should have 'boxes' (N, 4) and 'labels' (N,).

        Returns:
            DetectionOutput with class_logits and box_regression.
        """
        # Extract features
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        # Detection head
        class_logits, box_regression = self.head(fpn_features)

        return DetectionOutput(
            class_logits=class_logits,
            box_regression=box_regression,
        )


@register_model("fasterrcnn")
class FasterRCNN(nn.Module):
    """Faster R-CNN: Two-stage object detector.

    Architecture:
        Backbone -> FPN -> RPN (proposals) -> RoI Head (detection)

    First stage generates region proposals, second stage refines
    boxes and predicts classes.

    Reference:
        Ren et al., "Faster R-CNN", NeurIPS 2015.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = True,
        fpn_channels: int = 256,
    ):
        """Initialize Faster R-CNN.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of object classes (excluding background).
            backbone: Backbone architecture name.
            pretrained: Whether to use pretrained backbone.
            fpn_channels: Number of channels in FPN.
        """
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        if backbone.startswith("efficientnet"):
            self.backbone: Backbone = EfficientNetBackbone(
                variant=backbone,
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )
        else:
            self.backbone = ResNetBackbone(
                variant=backbone,  # type: ignore
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )

        # FPN
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
            extra_levels=1,
        )

        # Region Proposal Network
        self.rpn_head = DetectionHead(
            in_channels=fpn_channels,
            num_classes=1,  # Objectness score only
            num_anchors=3,
            num_convs=1,
        )

        # RoI head (simplified - box classifier)
        self.roi_pool_size = 7
        self.roi_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fpn_channels * self.roi_pool_size**2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(1024, num_classes + 1)  # +1 for background
        self.bbox_pred = nn.Linear(1024, (num_classes + 1) * 4)

    def forward(
        self,
        x: torch.Tensor,
        _targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> DetectionOutput:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).
            targets: Optional list of target dictionaries for training.

        Returns:
            DetectionOutput with class_logits, box_regression, and proposals.
        """
        # Extract features
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        # RPN forward
        rpn_class_logits, rpn_box_regression = self.rpn_head(fpn_features)

        return DetectionOutput(
            class_logits=rpn_class_logits,
            box_regression=rpn_box_regression,
            proposals=None,  # Would be computed from RPN outputs
        )


@register_model("fcos")
class FCOS(nn.Module):
    """FCOS: Fully Convolutional One-Stage Object Detection.

    Anchor-free detector that predicts bounding boxes at each spatial
    location using center-ness and regression targets.

    Reference:
        Tian et al., "FCOS: Fully Convolutional One-Stage Object Detection",
        ICCV 2019.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = True,
        fpn_channels: int = 256,
    ):
        """Initialize FCOS.

        Args:
            in_channels: Number of input image channels.
            num_classes: Number of object classes.
            backbone: Backbone architecture name.
            pretrained: Whether to use pretrained backbone.
            fpn_channels: Number of channels in FPN.
        """
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        if backbone.startswith("efficientnet"):
            self.backbone: Backbone = EfficientNetBackbone(
                variant=backbone,
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )
        else:
            self.backbone = ResNetBackbone(
                variant=backbone,  # type: ignore
                in_channels=in_channels,
                pretrained=pretrained and in_channels == 3,
            )

        # FPN
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=fpn_channels,
            extra_levels=2,
        )

        # Shared classification and regression convolutions
        cls_convs = []
        reg_convs = []
        for _ in range(4):
            cls_convs.extend(
                [
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, fpn_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            reg_convs.extend(
                [
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(32, fpn_channels),
                    nn.ReLU(inplace=True),
                ]
            )

        self.cls_convs = nn.Sequential(*cls_convs)
        self.reg_convs = nn.Sequential(*reg_convs)

        # Prediction heads
        self.cls_pred = nn.Conv2d(fpn_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(fpn_channels, 4, 3, padding=1)  # l, t, r, b
        self.centerness_pred = nn.Conv2d(fpn_channels, 1, 3, padding=1)

        # Learnable scale parameters per level
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(5)])

    def forward(
        self,
        x: torch.Tensor,
        _targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> DetectionOutput:
        """Forward pass.

        Args:
            x: Input images of shape (B, C, H, W).
            targets: Optional list of target dictionaries for training.

        Returns:
            DetectionOutput with class_logits and box_regression.
            Box regression contains (l, t, r, b) distances from each location.
        """
        # Extract features
        backbone_features = self.backbone(x)
        fpn_features = self.fpn(backbone_features)

        class_logits = []
        box_regressions = []

        for level_idx, feature in enumerate(fpn_features):
            cls_feat = self.cls_convs(feature)
            reg_feat = self.reg_convs(feature)

            # Classification
            cls_logit = self.cls_pred(cls_feat)
            class_logits.append(cls_logit)

            # Regression (with level-specific scale)
            scale = self.scales[min(level_idx, len(self.scales) - 1)]
            box_reg = scale * self.reg_pred(reg_feat)
            box_regressions.append(box_reg)

        return DetectionOutput(
            class_logits=class_logits,
            box_regression=box_regressions,
        )
