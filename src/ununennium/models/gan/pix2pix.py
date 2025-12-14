"""Pix2Pix for paired image-to-image translation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ununennium.models.gan.generators import UNetGenerator
from ununennium.models.gan.discriminators import PatchDiscriminator
from ununennium.models.gan.losses import AdversarialLoss
from ununennium.models.registry import register_model


@register_model("pix2pix")
class Pix2Pix(nn.Module):
    """Pix2Pix model for paired image translation.

    Uses a U-Net generator and PatchGAN discriminator with L1 loss
    for high-quality conditional image generation.

    Reference:
        Isola et al., "Image-to-Image Translation with Conditional
        Adversarial Networks", CVPR 2017.

    Example:
        >>> model = Pix2Pix(in_channels=12, out_channels=3)
        >>> fake = model(input_image)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        lambda_l1: float = 100.0,
        adversarial_mode: str = "lsgan",
    ):
        super().__init__()

        self.generator = UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
        )
        self.discriminator = PatchDiscriminator(
            in_channels=in_channels + out_channels,
            base_channels=base_channels,
        )

        self.adv_loss = AdversarialLoss(mode=adversarial_mode)
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = lambda_l1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output image.

        Args:
            x: Input image.

        Returns:
            Generated image.
        """
        return self.generator(x)

    def compute_generator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses.

        Args:
            real_a: Input domain image.
            real_b: Target domain image.

        Returns:
            Dictionary with loss components and generated image.
        """
        fake_b = self.generator(real_a)

        # Adversarial loss
        fake_ab = torch.cat([real_a, fake_b], dim=1)
        pred_fake = self.discriminator(fake_ab)
        loss_adv = self.adv_loss(pred_fake, target_is_real=True, for_discriminator=False)

        # L1 reconstruction loss
        loss_l1 = self.l1_loss(fake_b, real_b)

        loss_total = loss_adv + self.lambda_l1 * loss_l1

        return {
            "total": loss_total,
            "adversarial": loss_adv,
            "l1": loss_l1,
            "fake_b": fake_b,
        }

    def compute_discriminator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_b: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses.

        Args:
            real_a: Input domain image.
            real_b: Target domain image.
            fake_b: Generated image (detached).

        Returns:
            Dictionary with loss components.
        """
        # Real pair
        real_ab = torch.cat([real_a, real_b], dim=1)
        pred_real = self.discriminator(real_ab)
        loss_real = self.adv_loss(pred_real, target_is_real=True, for_discriminator=True)

        # Fake pair
        fake_ab = torch.cat([real_a, fake_b.detach()], dim=1)
        pred_fake = self.discriminator(fake_ab)
        loss_fake = self.adv_loss(pred_fake, target_is_real=False, for_discriminator=True)

        loss_total = (loss_real + loss_fake) * 0.5

        return {
            "total": loss_total,
            "real": loss_real,
            "fake": loss_fake,
        }
