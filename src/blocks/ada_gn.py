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