# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# BasicTransformerBlock that performs self-attention, optional cross-attention,
# and feedforward processing with residual connections.


import torch.nn as nn

from src.blocks.cross_attention import CrossAttention
from src.blocks.feedforward import FeedForward

class BasicTransformerBlock(nn.Module):
    """
    Custom transformer block:
    1. Self-attention
    2. Cross-attention (optional)
    3. FeedForward
    """
    def __init__(self, dim, heads, head_dim, context_dim=None, dropout=0.0):
        super().__init__()

        self.self_attn = CrossAttention(dim, dim, heads, head_dim, dropout)
        self.cross_attn = CrossAttention(dim, context_dim, heads, head_dim, dropout)
        self.ff = FeedForward(dim, dropout=dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        # self-attention
        x = x + self.self_attn(self.norm1(x))

        # cross-attention (or self-attention if context=None)
        x = x + self.cross_attn(self.norm2(x), context)

        # feedforward
        x = x + self.ff(self.norm3(x))

        return x
