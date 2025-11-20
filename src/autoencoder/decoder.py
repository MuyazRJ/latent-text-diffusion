import torch
import torch.nn as nn

from typing import List

from src.blocks.resblock_autoencoder import AE_ResBlock
from src.blocks.sample import UpSample


class Decoder(nn.Module):
    def __init__(self, image_channels: int, latent_dim: int, base_channels: int, channel_multipliers = List[int], group_norm_slices: int = 32, num_res_blocks: int = 2):
        super().__init__()

        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.base_channels = base_channels

        self.channel_multipliers = channel_multipliers
        self.group_norm_slices = group_norm_slices
        self.num_res_blocks = num_res_blocks

        self.in_channels = channel_multipliers[-1] * base_channels
        self.init_conv = nn.Conv2d(latent_dim, self.in_channels, kernel_size=3, padding=1)

        self.decoder = nn.ModuleList()
        for layer, multiplier in reversed(list(enumerate(channel_multipliers))):
            out_channels = base_channels * multiplier

            for _ in range(num_res_blocks):
                self.decoder.append(AE_ResBlock(in_channels=self.in_channels, out_channels=out_channels, group_norm_slices=group_norm_slices))
                self.in_channels = out_channels
            
            if layer != 0:
                self.decoder.append(UpSample(channels=self.in_channels))

        self.final = nn.Sequential(
            nn.GroupNorm(group_norm_slices, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, image_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(z)

        for block in self.decoder:
            x = block(x)

        x = self.final_conv(x)

        # Stable Diffusion clamps output to [-1,1]
        x = torch.tanh(x)
        return x