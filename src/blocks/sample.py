import torch.nn as nn
from torch.nn import functional as F

"""Downsampling block."""
class DownSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()

        # Store input and output channel sizes
        self.channels = channels
        self.out_channels = out_channels or channels

        # Strided convolution halves the spatial resolution
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Check that the input tensor has the expected channel size
        assert x.shape[1] == self.channels
        return self.conv(x)

"""Upsampling block."""
class UpSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()

        # Store input and output channel sizes
        self.channels = channels
        self.out_channels = out_channels or channels

        # Convolution applied after nearest-neighbour upsampling
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Check that the input tensor has the expected channel size
        assert x.shape[1] == self.channels

        # Double the spatial resolution
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)