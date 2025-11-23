import torch


def q_sample(x0, t, alpha_bars):
    """
    Forward diffusion (q-sample) step: adds noise to x0 at timestep t.
    
    Args:
        x0 (torch.Tensor): Original images, shape [B, C, H, W].
        t (torch.Tensor): Timesteps, shape [B] (1 <= t <= T).
        alpha_bars (torch.Tensor): Precomputed cumulative products of alphas, shape [T].
    
    Returns:
        x_t (torch.Tensor): Noisy image at timestep t.
        eps (torch.Tensor): The noise added.
    """
    # Gather corresponding alpha_bar_t for each batch element
    alpha_bar_t = alpha_bars[t].reshape(-1, 1, 1, 1)
    
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps
    return x_t, eps