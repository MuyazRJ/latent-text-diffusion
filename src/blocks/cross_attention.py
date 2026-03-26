# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Multi-head cross-attention module for diffusion transformer blocks.
# Projects input queries and optional conditioning context into query, key,
# and value representations, computes scaled dot-product attention across
# multiple heads, and returns attended features in the original query space.
#
# When no external context is provided, the module defaults to self-attention.
# This design is used in latent diffusion architectures to inject text or
# other conditioning information into spatial or token-based feature maps.
#
# Based on:
# - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
#   https://arxiv.org/abs/1706.03762
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752

import torch
import torch.nn as nn

from einops import rearrange

class CrossAttention(nn.Module):
    """
    Custom cross-attention module.
    Mathematically equivalent to SD's attention but rewritten cleanly.
    """
    def __init__(self, query_dim, context_dim=None, num_heads=4, head_dim=64, dropout=0.0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.inner_dim = num_heads * head_dim

        # if no context is provided → self-attention
        self.context_dim = context_dim or query_dim

        # query comes from x, keys/values come from context
        self.q_proj = nn.Linear(query_dim,   self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.context_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.context_dim, self.inner_dim, bias=False)

        # output projection
        self.out_proj = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.scale = head_dim ** -0.5

    def forward(self, x, context=None):
        """
        x:       [B, N, C]
        context: [B, M, C_ctx] or None
        """
        context = context if context is not None else x

        B, N, _ = x.shape
        _, M, _ = context.shape

        # project to Q,K,V
        q = self.q_proj(x)          # [B, N, inner_dim]
        k = self.k_proj(context)    # [B, M, inner_dim]
        v = self.v_proj(context)    # [B, M, inner_dim]

        # split into heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.num_heads)

        # attention scores
        attn_scores = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)

        # attention applied to values
        out = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)

        # merge heads
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.out_proj(out)

