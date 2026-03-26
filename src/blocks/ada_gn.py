# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Adaptive Group Normalization (AdaGN) layer for diffusion model conditioning.
# Applies Group Normalization without affine parameters, then modulates the
# normalized activations using learned scale and shift terms generated from an
# embedding vector, such as a timestep or conditioning embedding.
#
# This conditioning mechanism allows external embeddings to control feature
# statistics throughout the network, which is commonly used in diffusion U-Net
# architectures for timestep-aware feature modulation.
#
# Based on:
# - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
#   https://arxiv.org/abs/1709.07871
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752

import torch.nn as nn


class AdaGn(nn.Module):
    def __init__(self, num_channels, embed_dim, num_groups=32):
        """Adaptive Group Normalization layer."""
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.linear = nn.Linear(embed_dim, num_channels * 2)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x, emb):
        """Forward pass of AdaGn."""
        # t_emb: [B, embed_dim]
        scale, shift = self.linear(emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = self.gn(x)
        return x * (1 + scale) + shift