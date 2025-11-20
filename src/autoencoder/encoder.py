import torch
import torch.nn as nn

from typing import List

from src.blocks.resblock_autoencoder import AE_ResBlock
from src.blocks.sample import DownSample

class Encoder(nn.Module):
    """Encoder module for autoencoder architecture."""
    def __init__(self, image_channels: int, latent_dim: int, base_channels: int, channel_multipliers = List[int], group_norm_slices: int = 32, num_res_blocks: int = 2):
        super().__init__()

        self.image_channels = image_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        self.channel_multipliers = channel_multipliers
        self.group_norm_slices = group_norm_slices
        self.num_res_blocks = num_res_blocks

        self.in_channels = channel_multipliers[0] * base_channels
        self.init_conv = nn.Conv2d(self.image_channels, self.in_channels, kernel_size=3, padding=1)

        self.encoder = nn.ModuleList()
        for layer, multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * multiplier

            for _ in range(num_res_blocks):
                self.encoder.append(AE_ResBlock(in_channels=self.in_channels, out_channels=out_channels, group_norm_slices=group_norm_slices))
                self.in_channels = out_channels
            
            if layer != len(channel_multipliers) - 1:
                self.encoder.append(DownSample(channels=self.in_channels))

        self.to_mu_log_var =  nn.Conv2d(self.in_channels, latent_dim * 2, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)

        for layer in self.encoder:
            x = layer(x)
        
        x = self.to_mu_log_var(x)
        mu, log_var = torch.chunk(x, 2, dim=1)

        # Reparameterization
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # (B, latent_dim, H/4, W/4)

        return z, mu, log_var