import torch.nn as nn

from src.blocks.ada_gn import AdaGn

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, out_channels=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.dropout = dropout

        # --- First conv ---
        self.norm1 = AdaGn(in_channels, embed_dim=embed_dim)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

        # --- Second conv ---
        self.norm2 = AdaGn(self.out_channels, embed_dim=embed_dim)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)

        # --- Activation & dropout ---
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # --- Skip connection ---
        if self.in_channels != self.out_channels:
            self.skip_connection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x, t_emb):
        # --- First half ---
        h = self.norm1(x, t_emb)
        h = self.act(h)
        h = self.conv1(h)

        # --- Second half ---
        h = self.norm2(h, t_emb)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        # --- Residual add ---
        return self.skip_connection(x) + h