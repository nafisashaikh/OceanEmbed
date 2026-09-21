"""Full OceanEmbed reconstruction model with encoder skip connections."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.embedding_encoder import ConvBlock, SurfaceEncoder


class SkipSurfaceDecoder(nn.Module):
    """Decode an embedding with the encoder's two spatial skip features."""

    def __init__(self, output_channels: int, embedding_dim: int, base_channels: int = 32):
        super().__init__()
        self.base_channels = base_channels
        self.expand = nn.Linear(embedding_dim, base_channels * 4)
        self.bridge = ConvBlock(base_channels * 4, base_channels * 2)
        self.merge2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.merge1 = ConvBlock(base_channels * 3, base_channels)
        self.output = nn.Conv2d(base_channels, output_channels, 3, padding=1)

    def forward(
        self,
        embedding: torch.Tensor,
        skip1: torch.Tensor,
        skip2: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        features = self.expand(embedding).unsqueeze(-1).unsqueeze(-1)
        features = F.interpolate(features, size=bottleneck.shape[-2:], mode="bilinear", align_corners=False)
        features = self.bridge(features + bottleneck)
        features = F.interpolate(features, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        features = self.merge2(torch.cat([features, skip2], dim=1))
        features = F.interpolate(features, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        features = self.merge1(torch.cat([features, skip1], dim=1))
        return self.output(features)


class OceanEmbedReconstructionModel(nn.Module):
    """Encode surface variables and reconstruct 15 depth-resolved temperatures."""

    def __init__(self, input_channels: int, output_channels: int, embedding_dim: int, base_channels: int = 32):
        super().__init__()
        self.encoder = SurfaceEncoder(input_channels, embedding_dim, base_channels)
        self.decoder = SkipSurfaceDecoder(output_channels, embedding_dim, base_channels)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip1, skip2, bottleneck, embedding = self.encoder.forward_features(inputs)
        output = self.decoder(embedding, skip1, skip2, bottleneck)
        output = F.interpolate(output, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        return output, embedding