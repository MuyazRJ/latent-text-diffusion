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
            eps_pred, var_raw = model(x_t, t_tensor, context)

        # === OpenAI 2021: Predict x0 ===
        alpha_bar_t = alpha_bars[t].view(1,1,1,1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Clip x0 prediction to [-1,1] as in the paper
        x0_pred = x0_pred.clamp(-1, 1)
        dtype = torch.float16
        # === Compute coefficients ===
        alpha_t     = alphas[t].to(dtype).view(1,1,1,1)
        beta_t      = betas[t].to(dtype).view(1,1,1,1)
        if t > 0:
            alpha_bar_prev = alpha_bars[t-1].to(dtype).view(1,1,1,1)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=device).to(dtype).view(1,1,1,1)

        # ========== μ_t (posterior mean) ==========
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x_t

        # ========== Learned variance σ_t² ==========
        tilde_beta_t = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

        frac = (var_raw + 1) / 2
        frac = frac.clamp(0, 1)

        min_log = torch.log(tilde_beta_t)
        max_log = torch.log(beta_t)

        model_log_sigma2 = frac * max_log + (1 - frac) * min_log
        model_sigma2 = torch.exp(model_log_sigma2)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(model_sigma2) * noise
        else:
            x_t = mean

    x_t = x_t.clamp(-1, 1)
    return (x_t + 1) / 2