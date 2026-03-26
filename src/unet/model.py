# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Latent diffusion U-Net denoiser for CUB-200-2011.
# Operates on 4x32x32 latents from a frozen VAE and conditions generation
# on CLIP text embeddings via cross-attention SpatialTransformer modules.
#
# Based on:
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239

import torch
import torch.nn as nn

from src.embedding.sinusoidal import SinusoidalTimeEmbedding
from src.diffusion.transformer.spatial_transformer import SpatialTransformer
from src.blocks.resblock_diffusion import ResBlock
from src.blocks.sample import DownSample, UpSample

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, context=None):
        # Passes the input through each layer while handling timestep and text conditioning
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class SOTADiffusion(nn.Module):
    def __init__(self, config, context_dim, latent_channels = 4):
        super().__init__()
        self.latent_channels = latent_channels

        # Store key model settings from the config
        self.embedding_dim = config["sinusoidal_dim"]
        self.time_embedding_dim = config["time_embedding_dim"]
        self.num_res_blocks = config["num_res_blocks"]

        self.channel_mults = config["channel_mults"]
        self.base_channels = config["base_channels"]
        self.attention_layers = config["attention_res"]

        self.dropout_rate = config["dropout"]
        self.heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.context_dim = context_dim

        # Creates sinusoidal timestep embeddings and projects them into a higher-dimensional space
        self.embedder = SinusoidalTimeEmbedding(embed_dim = self.embedding_dim)
        self.time_embedding = nn.Sequential(
        nn.Linear(self.embedding_dim, self.time_embedding_dim),
        nn.SiLU(),
        nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # Initial convolution to map latent input channels to the first feature dimension
        in_channels = int(self.channel_mults[0] * self.base_channels)
        self.init_conv = nn.Conv2d(self.latent_channels, in_channels, kernel_size=3, padding=1)

        # Downsampling path of the U-Net
        self.down_blocks = nn.ModuleList()
        for layer, multiplier in enumerate(self.channel_mults):
            out_channels = int(self.base_channels * multiplier)
            res_layers = []

            # Add residual blocks at the current resolution
            for _ in range(self.num_res_blocks):
                res_layers.append(
                    ResBlock(in_channels=in_channels, out_channels=out_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim)
                )
                in_channels = out_channels

                # Add attention block if this resolution is marked for attention
                if layer in self.attention_layers:
                    res_layers.append(SpatialTransformer(channels = out_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim))
            
            self.down_blocks.append(TimestepEmbedSequential(*res_layers))

            # Add downsampling between resolution levels, except at the final level
            if layer != len(self.channel_mults) - 1:
                self.down_blocks.append(DownSample(out_channels))

        # Bottleneck block at the lowest resolution
        self.bottleneck = TimestepEmbedSequential(
            ResBlock(in_channels=in_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim),
            SpatialTransformer(channels = in_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim),
            ResBlock(in_channels=in_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim),
        )

        # Upsampling path of the U-Net
        self.up_blocks = nn.ModuleList()
        for layer, multiplier in reversed(list(enumerate(self.channel_mults))):
            out_channels = int(self.base_channels * multiplier)
            res_layers = []

            for i in range(self.num_res_blocks + 1):
                # First residual block takes concatenated skip connection + current feature map
                if i == 0:
                    res_layers.append(
                        ResBlock(in_channels + out_channels, out_channels=out_channels,
                                dropout=self.dropout_rate, embed_dim=self.time_embedding_dim)
                    )
                else:
                    res_layers.append(
                        ResBlock(out_channels, out_channels=out_channels,
                                dropout=self.dropout_rate, embed_dim=self.time_embedding_dim)
                    )

                # Add attention block if this resolution uses attention
                if layer in self.attention_layers:
                    res_layers.append(SpatialTransformer(channels = out_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim))

            # Add upsampling between resolution levels, except at the final output level
            if layer != 0:
                res_layers.append(UpSample(out_channels))

            # Store all blocks for this level inside a timestep-aware sequential block
            self.up_blocks.append(TimestepEmbedSequential(*res_layers))

            # Update channel count for the next level
            in_channels = out_channels

        # Final normalisation, activation, and projection back to latent channel size
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, context = None):
        # Create timestep embeddings for the current diffusion step
        t_emb = self.embedder(timesteps)
        t_emb = self.time_embedding(t_emb)

        # Project input latents into the first feature space
        x = self.init_conv(x)

        # Store intermediate outputs for skip connections during downsampling
        skip_connections = []
        for block in self.down_blocks:
            if not isinstance(block, DownSample):
                x = block(x, t_emb, context)
                skip_connections.append(x)
            else:
                x = block(x)

        # Process the lowest-resolution representation through the bottleneck
        x = self.bottleneck(x, t_emb, context)

        # Upsampling path with skip connections from the encoder
        for block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = torch.cat((x, skip_x), dim=1)
            x = block(x, t_emb, context)

        # Map features back to the latent output space
        x = self.out(x)

        return x