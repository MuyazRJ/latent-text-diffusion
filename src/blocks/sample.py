import torch.nn as nn
from torch.nn import functional as F

"""Downsampling block."""
class DownSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.conv(x)

"""Upsampling block."""
class UpSample(nn.Module):
    def __init__(self, channels, out_channels = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)