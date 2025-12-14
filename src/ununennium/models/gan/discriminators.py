"""Discriminator architectures for GANs."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies NxN patches.

    Instead of classifying the entire image as real/fake, outputs
    a grid where each value represents real/fake likelihood for
    a local patch (70x70 receptive field by default).
    """

    def __init__(
        self,
        in_channels: int = 6,
        base_channels: int = 64,
        num_layers: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        norm_layer = spectral_norm if use_spectral_norm else lambda x: x

        layers = [
            norm_layer(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_ch = base_channels
        for i in range(1, num_layers):
            out_ch = min(base_channels * (2**i), 512)
            layers.extend(
                [
                    norm_layer(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)),
                    nn.BatchNorm2d(out_ch) if not use_spectral_norm else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_ch = out_ch

        # Final layers
        out_ch = min(base_channels * (2**num_layers), 512)
        layers.extend(
            [
                norm_layer(nn.Conv2d(in_ch, out_ch, 4, stride=1, padding=1)),
                nn.BatchNorm2d(out_ch) if not use_spectral_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                norm_layer(nn.Conv2d(out_ch, 1, 4, stride=1, padding=1)),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator using multiple PatchGANs.

    Applies discriminators at different scales to capture both
    fine-grained and coarse patterns.
    """

    def __init__(
        self,
        in_channels: int = 6,
        num_scales: int = 3,
        base_channels: int = 64,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [PatchDiscriminator(in_channels, base_channels) for _ in range(num_scales)]
        )
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs


class SpectralNormDiscriminator(nn.Module):
    """Discriminator with spectral normalization for training stability."""

    def __init__(self, in_channels: int = 6, base_channels: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
