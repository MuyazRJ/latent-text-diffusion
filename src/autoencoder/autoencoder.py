# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

import torch
import torch.nn as nn

from typing import Dict, Any

from src.autoencoder.encoder import Encoder
from src.autoencoder.decoder import Decoder


class Autoencoder(nn.Module):
    """Autoencoder model combining Encoder and Decoder."""

    def __init__(self, ae_config: Dict[str, Any]):
        super().__init__()

        self.image_channels = ae_config["image_channels"]   
        self.latent_dim = ae_config["latent_dim"]
        self.base_channels = ae_config["base_channels"]
        self.channel_multipliers = ae_config["channel_multipliers"]
        self.group_norm_slices = ae_config["group_norm_slices"]
        self.num_res_blocks = ae_config["num_res_blocks"]

        self.encoder = Encoder(
            image_channels=self.image_channels,
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            channel_multipliers=self.channel_multipliers,
            group_norm_slices=self.group_norm_slices,
            num_res_blocks=self.num_res_blocks
        )

        self.decoder = Decoder(
            image_channels=self.image_channels,
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            channel_multipliers=self.channel_multipliers,
            group_norm_slices=self.group_norm_slices,
            num_res_blocks=self.num_res_blocks
        )
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes input images into latent representations."""
        z, mu, log_var = self.encoder(x)
        return z, mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent representations back into images."""
        x_recon = self.decoder(z)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full autoencoder forward pass: encode and then decode."""
        z, mu, log_var = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mu, log_var