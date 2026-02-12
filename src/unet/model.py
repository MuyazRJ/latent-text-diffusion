import torch
import torch.nn as nn

from src.embedding.sinusoidal import SinusoidalTimeEmbedding
from src.diffusion.transformer.spatial_transformer import SpatialTransformer
from src.blocks.resblock_diffusion import ResBlock
from src.blocks.sample import DownSample, UpSample

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, context=None):
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

        self.embedder = SinusoidalTimeEmbedding(embed_dim = self.embedding_dim)
        self.time_embedding = nn.Sequential(
        nn.Linear(self.embedding_dim, self.time_embedding_dim),
        nn.SiLU(),
        nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        in_channels = int(self.channel_mults[0] * self.base_channels)
        self.init_conv = nn.Conv2d(self.latent_channels, in_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        for layer, multiplier in enumerate(self.channel_mults):
            out_channels = int(self.base_channels * multiplier)
            res_layers = []

            # Add ResBlocks
            for _ in range(self.num_res_blocks):
                res_layers.append(
                    ResBlock(in_channels=in_channels, out_channels=out_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim)
                )
                in_channels = out_channels

                if layer in self.attention_layers:
                    res_layers.append(SpatialTransformer(channels = out_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim))
            
            self.down_blocks.append(TimestepEmbedSequential(*res_layers))

            # Add Downsample except at last level
            if layer != len(self.channel_mults) - 1:
                self.down_blocks.append(DownSample(out_channels))

        self.bottleneck = TimestepEmbedSequential(
            ResBlock(in_channels=in_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim),
            SpatialTransformer(channels = in_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim),
            ResBlock(in_channels=in_channels, dropout = self.dropout_rate, embed_dim = self.time_embedding_dim),
        )

        self.up_blocks = nn.ModuleList()
        for layer, multiplier in reversed(list(enumerate(self.channel_mults))):
            out_channels = int(self.base_channels * multiplier)
            res_layers = []

            for i in range(self.num_res_blocks + 1):
                # --- First ResBlock merges skip connection ---
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

                if layer in self.attention_layers:
                    res_layers.append(SpatialTransformer(channels = out_channels, heads = self.heads, head_dim = self.head_dim, context_dim = self.context_dim))

            # --- Add Upsample except final level ---
            if layer != 0:
                res_layers.append(UpSample(out_channels))

            # Add this level as a timestep-aware sequential
            self.up_blocks.append(TimestepEmbedSequential(*res_layers))

            # Update for next loop
            in_channels = out_channels

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps, context = None):
        # Time embedding
        t_emb = self.embedder(timesteps)
        t_emb = self.time_embedding(t_emb)

        # Initial convolution
        x = self.init_conv(x)

        # Downsampling path
        skip_connections = []
        for block in self.down_blocks:
            if not isinstance(block, DownSample):
                x = block(x, t_emb, context)
                skip_connections.append(x)
            else:
                x = block(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb, context)

        # Upsampling path
        for block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = torch.cat((x, skip_x), dim=1)
            x = block(x, t_emb, context)

        # Final output layer
        x = self.out(x)

        return x