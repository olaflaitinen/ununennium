"""Loss functions for GAN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AdversarialLoss(nn.Module):
    """Configurable adversarial loss supporting multiple variants.

    Supports vanilla GAN, LSGAN, WGAN, and hinge losses.
    """

    def __init__(self, mode: str = "lsgan"):
        super().__init__()
        self.mode = mode
        self.loss: nn.Module | None = None

        if mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.loss = nn.MSELoss()
        elif mode in ("wgan", "hinge"):
            self.loss = None
        else:
            raise ValueError(f"Unknown adversarial loss mode: {mode}")

    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True,
    ) -> torch.Tensor:
        """Compute adversarial loss.

        Args:
            pred: Discriminator output.
            target_is_real: Whether the target should be real.
            for_discriminator: Whether this is for D or G training.

        Returns:
            Loss value.
        """
        if self.mode in ("vanilla", "lsgan"):
            target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
            return self.loss(pred, target)  # type: ignore

        elif self.mode == "wgan":
            return -pred.mean() if target_is_real else pred.mean()

        elif self.mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    return F.relu(1 - pred).mean()
                else:
                    return F.relu(1 + pred).mean()
            else:
                return -pred.mean()

        raise ValueError(f"Unknown mode: {self.mode}")


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for high-quality image generation."""

    def __init__(
        self,
        feature_layers: list[int] | None = None,
        weights: list[float] | None = None,
    ):
        super().__init__()

        if feature_layers is None:
            feature_layers = [2, 7, 12, 21, 30]
        if weights is None:
            weights = [1.0] * len(feature_layers)

        try:
            from torchvision import models  # noqa: PLC0415

            vgg = models.vgg19(weights="IMAGENET1K_V1").features
        except Exception:
            # Fallback: use VGG without pretrained weights
            from torchvision import models  # noqa: PLC0415

            vgg = models.vgg19().features

        self.blocks = nn.ModuleList()
        prev = 0
        for layer in feature_layers:
            self.blocks.append(nn.Sequential(*list(vgg.children())[prev:layer]))
            prev = layer

        self.weights = weights

        # Freeze VGG
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss.

        Args:
            pred: Generated image [-1, 1].
            target: Target image [-1, 1].

        Returns:
            Perceptual loss value.
        """
        # Normalize to VGG input range
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        pred = (pred - self.mean) / self.std  # type: ignore[operator]
        target = (target - self.mean) / self.std  # type: ignore[operator]

        loss = torch.tensor(0.0, device=pred.device)
        for block, weight in zip(self.blocks, self.weights, strict=True):
            pred = block(pred)
            with torch.no_grad():
                target = block(target)
            loss = loss + weight * F.l1_loss(pred, target)

        return loss


class SpectralAngleLoss(nn.Module):
    """Spectral angle mapper loss for multi-spectral consistency."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral angle between predicted and target.

        Args:
            pred: Predicted multi-band image (B, C, H, W).
            target: Target multi-band image.

        Returns:
            Mean spectral angle in radians.
        """
        pred_norm = F.normalize(pred, dim=1)
        target_norm = F.normalize(target, dim=1)

        cos_sim = (pred_norm * target_norm).sum(dim=1)
        angle = torch.acos(torch.clamp(cos_sim, -1 + 1e-7, 1 - 1e-7))

        return angle.mean()


class GradientPenalty(nn.Module):
    """Gradient penalty for WGAN-GP."""

    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp

    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty.

        Args:
            discriminator: Discriminator network.
            real: Real samples.
            fake: Generated samples.

        Returns:
            Gradient penalty value.
        """
        batch_size = real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)

        interpolated = alpha * real + (1 - alpha) * fake.detach()
        interpolated.requires_grad_(True)

        d_interpolated = discriminator(interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return self.lambda_gp * penalty
