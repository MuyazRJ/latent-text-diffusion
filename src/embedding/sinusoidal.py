import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    Generates sinusoidal timestep embeddings as described in
    'Attention Is All You Need' (Vaswani et al., 2017) and used in
    DDPM/Improved DDPM.

    Given a scalar timestep t ∈ [1, T], this produces a vector
    of dimension embed_dim containing sinusoids of different frequencies.
    """

    def __init__(self, embed_dim: int, max_period: float = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for integer timesteps.

        Args:
            timesteps (torch.Tensor): Shape (B,) — integer timestep values.

        Returns:
            torch.Tensor: Shape (B, embed_dim) — sinusoidal embeddings.
        """
        # Ensure float and add batch dimension if needed
        half_dim = self.embed_dim // 2
        device = timesteps.device

        # Compute the geometric sequence of frequencies
        freq = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )

        # Outer product: [B, 1] * [half_dim] → [B, half_dim]
        args = timesteps[:, None].float() * freq[None, :]

        # Concatenate sin and cos embeddings
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Pad if embed_dim is odd
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb
