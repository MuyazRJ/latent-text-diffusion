# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# DDPM reverse sampling function for diffusion and latent diffusion models.
# Iteratively denoises a Gaussian noise sample by applying the learned reverse
# diffusion process over all timesteps, using the model’s noise prediction to
# estimate the posterior mean and variance at each step.
#
# Supports conditional generation through an optional context input and returns
# the final denoised sample after traversing the full diffusion chain from
# timestep T-1 down to 0.
#
# Based on:
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239
# - Nichol and Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
#   https://arxiv.org/abs/2102.09672
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752

import torch 


def reverse(model, alphas, alpha_bars, betas, T, image_size, device, context=None):
    B, C, H, W = image_size
    x_t = torch.randn((B, C, H, W), device=device)

    model = model.to(device)

    if context is None:
        labels = torch.randint(0, C, (B,), device=device, dtype=torch.long)
    else:
        context =  context.to(device=device, dtype=x_t.dtype)

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred = model(x_t, t_tensor, context)

        # === OpenAI 2021: Predict x0 ===
        alpha_bar_t = alpha_bars[t].view(1,1,1,1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        # === Compute coefficients ===
        alpha_t     = alphas[t].view(1,1,1,1)
        beta_t      = betas[t].view(1,1,1,1)
        if t > 0:
            alpha_bar_prev = alpha_bars[t-1].view(1,1,1,1)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device).view(1,1,1,1)

        # ========== μ_t (posterior mean) ==========
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x_t

        # ========== Learned variance σ_t² ==========
        posterior_variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_t = mean

    return x_t