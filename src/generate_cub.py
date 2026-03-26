"""
generate_cub.py
---------------
Loads the trained CUB-200 latent text diffusion model and produces:
  - final_image : PIL.Image  (256x256 RGB, decoded from latent)
  - frames      : list[PIL.Image]  (10 frames, noisy -> clean, timestep stamped)

Usage in main.py
----------------
    from generate_cub import load_cub_model, generate_cub

    cub_bundle = load_cub_model()                              # once at startup
    img, frames = generate_cub(cub_bundle, prompt="a red bird", steps=50)
"""

import sys, os

_CUB_ROOT = os.path.dirname(__file__)
if _CUB_ROOT not in sys.path:
    sys.path.insert(0, _CUB_ROOT)

# Evict any conflicting cached modules from mnist/cifar repos
for _mod in list(sys.modules.keys()):
    if _mod in ("config", "model", "diffusion", "utils", "src") or \
       _mod.startswith(("src.", "model.", "diffusion.", "utils.", "blocks.",
                        "embeddings.", "data.", "models.")):
        del sys.modules[_mod]

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL

from src.utils.config_loader import load_config
from src.unet.model import SOTADiffusion
from src.diffusion.train import DiffusionTrainer
from src.diffusion.schedules import make_beta_schedule, compute_alphas

# ── Config ────────────────────────────────────────────────────────────────────

_CONFIGS_DIR  = os.path.join(_CUB_ROOT, "configs")
DISPLAY_SIZE  = 256
LATENT_SCALE  = 0.18215

# ── Model loading ─────────────────────────────────────────────────────────────

def load_cub_model():
    """
    Load the EMA diffusion model, VAE, CLIP encoder/tokenizer and schedules.
    Returns a bundle dict — pass directly to generate_cub().
    Call once at server startup.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_config    = load_config(os.path.join(_CONFIGS_DIR, "autoencoder.yaml"))["autoencoder"]
    model_config = load_config(os.path.join(_CONFIGS_DIR, "model.yaml"))
    t_cfg        = model_config["training"]

    # CLIP
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)
    context_dim  = text_encoder.config.hidden_size

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Diffusion model + schedules
    model  = SOTADiffusion(model_config["unet"], context_dim=context_dim,
                           latent_channels=ae_config["latent_dim"])
    betas  = make_beta_schedule(
        schedule_type=t_cfg["beta_schedule"],
        num_steps=t_cfg["time_steps"],
        start=t_cfg["beta_start"],
        end=t_cfg["beta_end"],
    ).to(device).float()
    _, alpha_bars = compute_alphas(betas)

    trainer = DiffusionTrainer(
        model=model, config=t_cfg,
        betas=betas, alpha_bars=alpha_bars,
        train_dataloader=None, device=device,
    )
    trainer.load_checkpoint()
    ema_model = trainer.get_ema_model()
    ema_model.to(device).eval()

    print(f"[CUB] Model loaded on {device}")

    return {
        "model":        ema_model,
        "vae":          vae,
        "tokenizer":    tokenizer,
        "text_encoder": text_encoder,
        "alpha_bars":   alpha_bars,
        "T":            t_cfg["time_steps"],
        "latent_dim":   ae_config["latent_dim"],
        "device":       device,
    }

# ── Helper: latent tensor -> PIL ──────────────────────────────────────────────

def _decode_latent(vae, latent: torch.Tensor) -> Image.Image:
    """Decode a (1, C, H, W) scaled latent to a 256x256 RGB PIL Image."""
    with torch.no_grad():
        decoded = vae.decode(latent / LATENT_SCALE).sample
    arr = decoded.squeeze(0).clamp(-1, 1)
    arr = ((arr + 1.0) / 2.0).permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    return pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)

def _stamp(pil: Image.Image, label: str) -> Image.Image:
    """Stamp a timestep label onto a PIL image."""
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except IOError:
        font = ImageFont.load_default()
    draw.text((11, 11), label, fill=(0, 0, 0), font=font)
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return pil

# ── DDIM loop with frame capture ──────────────────────────────────────────────

def _ddim_with_frames(model, vae, alpha_bars, T, latent_shape,
                      context, device, num_steps, num_frames=10):
    """
    Run DDIM sampling, decoding and saving `num_frames` evenly-spaced frames.

    Returns
    -------
    x_final  : torch.Tensor  (1, C, H, W) scaled latent
    frames   : list[PIL.Image]  ordered noisy -> clean
    """
    B, C, H, W = latent_shape
    c = T // num_steps
    ddim_ts = np.asarray(list(range(0, T, c))) + 1
    ddim_ts = np.flip(ddim_ts)          # high -> low  (noisy -> clean direction)
    total   = len(ddim_ts)

    # Which indices to capture (evenly spread, always include first and last)
    capture_at = set(
        np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    )

    x = torch.randn((B, C, H, W), device=device, dtype=torch.float32)
    if context is not None:
        context = context.to(device=device, dtype=torch.float32)

    frames = []

    for i, step in enumerate(ddim_ts):
        t_tensor     = torch.full((B,), int(step) - 1, device=device, dtype=torch.long)
        alpha_bar_t  = alpha_bars[int(step) - 1]
        alpha_bar_prev = (alpha_bars[int(ddim_ts[i + 1]) - 1]
                          if i < total - 1
                          else torch.tensor(1.0, device=device))

        with torch.no_grad():
            eps_pred = model(x, t_tensor, context)

        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        pred_x0 = pred_x0.clamp(-4, 4)
        dir_xt  = torch.sqrt(1 - alpha_bar_prev) * eps_pred
        x       = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if i in capture_at:
            # Decode the current latent for the GIF frame
            approx_t = int(step)
            frame_pil = _decode_latent(vae, x.clone())
            frame_pil = _stamp(frame_pil, f"t = {approx_t}")
            frames.append(frame_pil)

    return x, frames

# ── Main generation function ──────────────────────────────────────────────────

def generate_cub(bundle: dict, prompt: str = "", steps: int = 50):
    """
    Generate one bird image conditioned on `prompt` using `steps` DDIM steps.

    Parameters
    ----------
    bundle  : dict returned by load_cub_model()
    prompt  : text description, e.g. "a small red bird with black wings"
    steps   : 50 (fast) or 1000 (higher quality)

    Returns
    -------
    final_image : PIL.Image   256x256 RGB
    frames      : list[PIL.Image]  10 frames, noisy -> clean, timestep stamped
    """
    model        = bundle["model"]
    vae          = bundle["vae"]
    tokenizer    = bundle["tokenizer"]
    text_encoder = bundle["text_encoder"]
    alpha_bars   = bundle["alpha_bars"]
    T            = bundle["T"]
    latent_dim   = bundle["latent_dim"]
    device       = bundle["device"]

    if not prompt:
        prompt = ""

    # Encode prompt with CLIP
    inputs = tokenizer(
        [prompt], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        context = text_encoder(**inputs).last_hidden_state.to(torch.float32)

    latent_shape = (1, latent_dim, 32, 32)

    x_final, frames = _ddim_with_frames(
        model=model,
        vae=vae,
        alpha_bars=alpha_bars,
        T=T,
        latent_shape=latent_shape,
        context=context,
        device=device,
        num_steps=steps,
        num_frames=10,
    )

    final_image = _decode_latent(vae, x_final)
    return final_image, frames