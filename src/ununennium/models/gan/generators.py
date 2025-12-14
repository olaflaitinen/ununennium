"""Generator architectures for GANs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetGenerator(nn.Module):
    """U-Net generator with skip connections for image translation.

    Architecture uses encoder-decoder with skip connections to preserve
    spatial details during translation.

    Example:
        >>> generator = UNetGenerator(in_channels=12, out_channels=3)
        >>> output = generator(torch.randn(4, 12, 256, 256))
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_downs: int = 8,
        use_dropout: bool = True,
        norm_type: str = "batch",
    ):
        super().__init__()

        self.num_downs = num_downs

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_downs):
            out_ch = min(base_channels * (2**i), 512)
            self.encoders.append(
                self._encoder_block(in_ch, out_ch, normalize=(i > 0), norm_type=norm_type)
            )
            in_ch = out_ch

        # Decoder with skip connections
        self.decoders = nn.ModuleList()
        for i in range(num_downs - 1, -1, -1):
            if i == num_downs - 1:
                in_ch = min(base_channels * (2**i), 512)
            else:
                in_ch = min(base_channels * (2**i), 512) * 2

            if i == 0:
                out_ch = out_channels
            else:
                out_ch = min(base_channels * (2 ** (i - 1)), 512)

            self.decoders.append(
                self._decoder_block(
                    in_ch, out_ch,
                    dropout=(use_dropout and num_downs - 1 - i < 3),
                    norm_type=norm_type,
                    is_last=(i == 0),
                )
            )

    def _encoder_block(
        self, in_ch: int, out_ch: int, normalize: bool, norm_type: str
    ) -> nn.Module:
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            if norm_type == "batch":
                layers.append(nn.BatchNorm2d(out_ch))
            else:
                layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(
        self, in_ch: int, out_ch: int, dropout: bool, norm_type: str, is_last: bool
    ) -> nn.Module:
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=is_last)
        ]
        if not is_last:
            if norm_type == "batch":
                layers.append(nn.BatchNorm2d(out_ch))
            else:
                layers.append(nn.InstanceNorm2d(out_ch))
            if dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Decode with skip connections
        skips = skips[:-1][::-1]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x)
            if i < len(skips):
                x = torch.cat([x, skips[i]], dim=1)

        return x


class ResNetGenerator(nn.Module):
    """ResNet-based generator for higher resolution translation.

    Uses residual blocks in the bottleneck for improved gradient flow.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 9,
    ):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.down1 = self._downsample(base_channels, base_channels * 2)
        self.down2 = self._downsample(base_channels * 2, base_channels * 4)

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels * 4) for _ in range(num_blocks)]
        )

        # Upsampling
        self.up1 = self._upsample(base_channels * 4, base_channels * 2)
        self.up2 = self._upsample(base_channels * 2, base_channels)

        # Output
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, 7),
            nn.Tanh(),
        )

    def _downsample(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _upsample(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
