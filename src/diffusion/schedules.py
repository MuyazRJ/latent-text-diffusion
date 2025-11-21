import torch, math

def make_beta_schedule(schedule_type, num_steps, start=None, end=None):
    if schedule_type == "linear":
        return torch.linspace(start, end, num_steps)

    elif schedule_type == "cosine":
        # ---- OpenAI Improved DDPM cosine schedule ----
        s = 0.008

        # t from 0 to 1 (num_steps steps)
        steps = num_steps + 1
        t = torch.linspace(0, 1, steps)

        # alpha_bar(t) = cos^2(...)
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # normalize

        # Convert alpha_bar to betas: beta_t = 1 - (alpha_bar[t+1] / alpha_bar[t])
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

        # Safety clamp (OpenAI uses max_beta=0.999)
        return betas.clamp(1e-5, 0.999)

    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")


def compute_alphas(betas: torch.Tensor):
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars