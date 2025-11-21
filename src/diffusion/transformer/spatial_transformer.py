import torch.nn as nn

from src.diffusion.transformer.basic_transformer import BasicTransformerBlock
from einops import rearrange

class SpatialTransformer(nn.Module):
    """
    Applies a transformer over spatial feature maps.
    Converts [B, C, H, W] -> [B, HW, C]
    Runs transformer
    Converts back to [B, C, H, W]
    """
    def __init__(self, channels, heads, head_dim, depth=1, context_dim=None):
        super().__init__()
        inner_dim = heads * head_dim

        self.norm = nn.GroupNorm(1, channels)
        self.to_tokens = nn.Conv2d(channels, inner_dim, 1)
        self.to_image  = nn.Conv2d(inner_dim, channels, 1)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, heads, head_dim, context_dim)
            for _ in range(depth)
        ])

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        x = self.to_tokens(x)             # [B, inner_dim, H, W]
        x = rearrange(x, "b c h w -> b (h w) c")

        for block in self.blocks:
            x = block(x, context)

        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.to_image(x)

        return x + residual