"""CNN encoder and mirrored decoder for surface-field embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SurfaceEncoder(nn.Module):
    """U-Net-style convolutional encoder returning one vector per surface day."""

    def __init__(self, input_channels: int, embedding_dim: int, base_channels: int = 32):
        super().__init__()
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.base_channels = base_channels
        self.down1 = ConvBlock(input_channels, base_channels)
        self.down2 = ConvBlock(base_channels, base_channels * 2, stride=2)
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(base_channels * 4, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, _, _, embedding = self.forward_features(inputs)
        return embedding

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return U-Net skip features followed by the bottleneck embedding."""
        skip1 = self.down1(inputs)
        skip2 = self.down2(skip1)
        bottleneck = self.down3(skip2)
        pooled = self.pool(bottleneck).flatten(1)
        return skip1, skip2, bottleneck, self.projection(pooled)


class SurfaceDecoder(nn.Module):
    """Decoder mirroring the encoder and restoring the configured grid size."""

    def __init__(self, output_channels: int, embedding_dim: int, output_size: tuple[int, int], base_channels: int = 32):
        super().__init__()
        self.output_channels = output_channels
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.base_channels = base_channels
        self.latent_size = (math.ceil(output_size[0] / 4), math.ceil(output_size[1] / 4))
        latent_channels = base_channels * 4
        self.expand = nn.Linear(embedding_dim, latent_channels * self.latent_size[0] * self.latent_size[1])
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.output = nn.Conv2d(base_channels, output_channels, 3, padding=1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        latent_channels = self.base_channels * 4
        features = self.expand(embedding).view(
            embedding.shape[0], latent_channels, self.latent_size[0], self.latent_size[1]
        )
        features = self.up1(features)
        features = self.up2(features)
        reconstruction = self.output(features)
        return F.interpolate(reconstruction, size=self.output_size, mode="bilinear", align_corners=False)


class SurfaceAutoencoder(nn.Module):
    """Composition of the independently usable surface encoder and decoder."""

    def __init__(self, input_channels: int, embedding_dim: int, output_size: tuple[int, int], base_channels: int = 32):
        super().__init__()
        self.encoder = SurfaceEncoder(input_channels, embedding_dim, base_channels)
        self.decoder = SurfaceDecoder(input_channels, embedding_dim, output_size, base_channels)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(inputs)
        return self.decoder(embedding), embedding