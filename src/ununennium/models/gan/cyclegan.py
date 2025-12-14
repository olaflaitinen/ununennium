"""CycleGAN for unpaired image-to-image translation."""

from __future__ import annotations

import torch
from torch import nn

from ununennium.models.gan.discriminators import PatchDiscriminator
from ununennium.models.gan.generators import ResNetGenerator
from ununennium.models.gan.losses import AdversarialLoss
from ununennium.models.registry import register_model


@register_model("cyclegan")
class CycleGAN(nn.Module):
    """CycleGAN model for unpaired image translation.

    Uses cycle consistency loss to learn mappings between domains
    without paired examples.

    Reference:
        Zhu et al., "Unpaired Image-to-Image Translation using
        Cycle-Consistent Adversarial Networks", ICCV 2017.

    Example:
        >>> model = CycleGAN(in_channels_a=2, in_channels_b=3)  # SAR to optical
        >>> fake_b = model(sar_image, direction="A2B")
    """

    def __init__(
        self,
        in_channels_a: int = 3,
        in_channels_b: int = 3,
        base_channels: int = 64,
        num_res_blocks: int = 9,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ):
        super().__init__()

        # Generators
        self.G_A2B = ResNetGenerator(in_channels_a, in_channels_b, base_channels, num_res_blocks)
        self.G_B2A = ResNetGenerator(in_channels_b, in_channels_a, base_channels, num_res_blocks)

        # Discriminators
        self.D_A = PatchDiscriminator(in_channels_a, base_channels)
        self.D_B = PatchDiscriminator(in_channels_b, base_channels)

        # Losses
        self.adv_loss = AdversarialLoss(mode="lsgan")
        self.cycle_loss = nn.L1Loss()
        self.identity_loss = nn.L1Loss()

        # Weights
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def forward(self, x: torch.Tensor, direction: str = "A2B") -> torch.Tensor:
        """Translate image to target domain.

        Args:
            x: Input image.
            direction: "A2B" or "B2A".

        Returns:
            Translated image.
        """
        if direction == "A2B":
            return self.G_A2B(x)
        else:
            return self.G_B2A(x)

    def compute_generator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute generator losses.

        Args:
            real_a: Domain A image.
            real_b: Domain B image.

        Returns:
            Dictionary with loss components and generated images.
        """
        # Forward cycle: A -> B -> A
        fake_b = self.G_A2B(real_a)
        rec_a = self.G_B2A(fake_b)

        # Backward cycle: B -> A -> B
        fake_a = self.G_B2A(real_b)
        rec_b = self.G_A2B(fake_a)

        # Adversarial losses
        loss_adv_a2b = self.adv_loss(self.D_B(fake_b), target_is_real=True, for_discriminator=False)
        loss_adv_b2a = self.adv_loss(self.D_A(fake_a), target_is_real=True, for_discriminator=False)
        loss_adv = loss_adv_a2b + loss_adv_b2a

        # Cycle consistency losses
        loss_cycle_a = self.cycle_loss(rec_a, real_a)
        loss_cycle_b = self.cycle_loss(rec_b, real_b)
        loss_cycle = (loss_cycle_a + loss_cycle_b) * self.lambda_cycle

        # Identity losses (optional regularization)
        loss_idt = torch.tensor(0.0, device=real_a.device)
        if self.lambda_identity > 0:
            idt_a = self.G_B2A(real_a)
            idt_b = self.G_A2B(real_b)
            loss_idt = (
                (self.identity_loss(idt_a, real_a) + self.identity_loss(idt_b, real_b))
                * self.lambda_identity
                * self.lambda_cycle
            )

        loss_total = loss_adv + loss_cycle + loss_idt

        return {
            "total": loss_total,
            "adversarial": loss_adv,
            "cycle": loss_cycle,
            "identity": loss_idt,
            "fake_b": fake_b,
            "fake_a": fake_a,
            "rec_a": rec_a,
            "rec_b": rec_b,
        }

    def compute_discriminator_loss(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute discriminator losses.

        Args:
            real_a: Real domain A image.
            real_b: Real domain B image.
            fake_a: Generated domain A image (detached).
            fake_b: Generated domain B image (detached).

        Returns:
            Dictionary with loss components.
        """
        # D_A loss
        loss_d_a_real = self.adv_loss(self.D_A(real_a), target_is_real=True, for_discriminator=True)
        loss_d_a_fake = self.adv_loss(
            self.D_A(fake_a.detach()), target_is_real=False, for_discriminator=True
        )
        loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

        # D_B loss
        loss_d_b_real = self.adv_loss(self.D_B(real_b), target_is_real=True, for_discriminator=True)
        loss_d_b_fake = self.adv_loss(
            self.D_B(fake_b.detach()), target_is_real=False, for_discriminator=True
        )
        loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

        return {
            "total": loss_d_a + loss_d_b,
            "d_a": loss_d_a,
            "d_b": loss_d_b,
        }
