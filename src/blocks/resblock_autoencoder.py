import torch.nn as nn


class AE_ResBlock(nn.Module):
    """Residual block for autoencoder architecture."""
    def __init__(self, in_channels: int, out_channels: int = None, group_norm_slices: int = 32):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(group_norm_slices, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(group_norm_slices, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)
