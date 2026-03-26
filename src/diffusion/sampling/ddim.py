# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# DDIM reverse sampling function for latent diffusion models.
# Performs deterministic or semi-stochastic denoising in latent space by
# iteratively transforming Gaussian noise into a predicted latent sample using
# a reduced set of diffusion timesteps and a trained noise prediction model.
#
# Designed for latent diffusion inference where the denoiser operates on
# VAE latents and optional text context is provided for conditional generation.
# The returned latent remains in the scaled latent space and must be unscaled
# before decoding with the VAE.
#
# Based on:
# - Song et al., "Denoising Diffusion Implicit Models", ICLR 2021
#   https://arxiv.org/abs/2010.02502
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752

import torch
import numpy as np

@torch.no_grad()
def reverse_ddim_ldm(
    model,
    alpha_bars,        # The full schedule from training (length 1000)
    T,
    image_size,        # (B, C, H, W)
    device,
    context=None,
    num_steps=51,      # DDIM steps
    eta=0.0,           # 0 = Deterministic (Recommended)
    scale_latent=0.18215 # Used ONLY at the very end
):
    B, C, H, W = image_size
    model = model.to(device)

    # 1. TIMESTEP SELECTION
    # Create a linear spacing (e.g., [0, 20, 40, ... 980])
    c = T // num_steps
    ddim_timesteps = np.asarray(list(range(0, T, c))) + 1 # offset by 1 to match 1-indexing of alphas
    
    # Reverse for sampling: [981, ..., 41, 21, 1]
    # We use 1-based indexing for alpha_bars to make the math easier
    ddim_timesteps = np.flip(ddim_timesteps)
    
    # 2. START WITH STANDARD GAUSSIAN NOISE
    # DO NOT scale this by 0.18215. The model expects N(0,1).
    x = torch.randn((B, C, H, W), device=device, dtype=torch.float32)

    if context is not None:
        context = context.to(device=device, dtype=torch.float32)

    for i, step in enumerate(ddim_timesteps):
        # The model expects timestep indices 0...999
        t_tensor = torch.full((B,), step - 1, device=device, dtype=torch.long)

        # 3. PREDICT NOISE (Epsilon)
        eps_pred = model(x, t_tensor, context)

        # 4. GET ALPHAS
        # Current alpha_bar
        alpha_bar_t = alpha_bars[step - 1]
        
        # Previous alpha_bar
        # If we are at the last step, the "previous" alpha is 1.0 (fully clean)
        if i < len(ddim_timesteps) - 1:
            prev_step = ddim_timesteps[i + 1]
            alpha_bar_prev = alpha_bars[prev_step - 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device)

        # 5. DDIM EQUATION
        
        # Predict x0 (Denoised Latent)
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        # Clamp x0 (Optional but recommended for stability)
        # Latents roughly -4 to 4
        pred_x0 = torch.clamp(pred_x0, -4, 4)

        # Direction to x_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        )
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred
        noise = sigma_t * torch.randn_like(x)

        # Update x
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

    # 6. RETURN
    # The result is currently *scaled*. Must be unscaled before VAE decoding.
    return x