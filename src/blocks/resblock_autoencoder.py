# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

import torch.nn as nn


class AE_ResBlock(nn.Module):
    """Residual block for autoencoder architecture."""
    def __init__(self, in_channels: int, out_channels: int = None, group_norm_slices: int = 32):
        super().__init__()

        # Use the input channel size if no output size is provided
        out_channels = out_channels or in_channels

        # First normalisation and convolution layer
        self.norm1 = nn.GroupNorm(group_norm_slices, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Second normalisation and convolution layer
        self.norm2 = nn.GroupNorm(group_norm_slices, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Use a 1x1 convolution on the skip path if channel sizes differ
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        # SiLU activation used after each normalisation step
        self.act = nn.SiLU()

    def forward(self, x):
        # Apply two conv blocks and add the skip connection
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)