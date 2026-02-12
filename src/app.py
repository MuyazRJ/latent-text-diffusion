# app.py
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch

from src.diffusion.sampling.ddim import reverse_ddim_ldm


def create_diffusion_gui(
    model,
    vae,
    text_encoder,
    tokenizer,
    alpha_bars,
    timesteps,
    device,
    latent_shape,
    scale=0.18215,
):
    """
    Create and run a simple Tkinter GUI around your diffusion model.

    Parameters
    ----------
    model : nn.Module
        Your SOTADiffusion UNet (already loaded, on `device`, in eval mode).
    vae : AutoencoderKL or your Autoencoder
        VAE used during training.
    text_encoder : CLIPTextModel
        CLIP text encoder.
    tokenizer : CLIPTokenizer
        Tokenizer matching text_encoder.
    alpha_bars : torch.Tensor
        Cumulative alphas used by reverse_ddim_ldm.
    timesteps : int
        Total diffusion timesteps used during training.
    device : torch.device
        Device where inference is run (cuda/cpu).
    latent_shape : tuple[int, int, int]
        (C, H, W) for the latent space.
    scale : float
        Latent scaling factor used with SD VAE (0.18215 in your code).
    """

    model.eval()
    vae.eval()
    text_encoder.eval()
    latent_shape = tuple(latent_shape)

    # ----------------- CORE SAMPLING FUNCTION -----------------

    @torch.no_grad()
    def generate_image(prompt: str, steps: int = 50, eta: float = 0.0, seed: int = 0):
        """Run your DDIM sampler and return a PIL.Image."""
        if not prompt or prompt.strip() == "":
            prompt = "a cute pokemon-like creature"

        steps = int(steps)
        if steps > timesteps:
            steps = timesteps

        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # 1) TEXT → EMBEDDINGS
        inputs = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(**inputs).last_hidden_state  # [1, L, D]

        # 2) DDIM SAMPLING IN LATENT SPACE
        B = 1
        C, H, W = latent_shape
        noise_shape = (B, C, H, W)

        print(f"[GUI] Running DDIM Sampling ({steps} steps)...")
        latents_scaled = reverse_ddim_ldm(
            model=model,
            alpha_bars=alpha_bars,
            T=timesteps,
            image_size=noise_shape,
            device=device,
            context=text_embeddings,
            num_steps=steps,
            eta=0,
        )

        # 3) DECODE LATENTS
        latents_unscaled = latents_scaled / scale
        decoded = vae.decode(latents_unscaled).sample  # [1, 3, H, W]
        decoded = ((decoded.clamp(-1, 1) + 1) / 2).cpu()[0]  # [3, H, W]

        # 4) TENSOR → PIL
        img = decoded.permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")
        pil_img = Image.fromarray(img, mode="RGB")
        return pil_img

    # ----------------- TKINTER GUI -----------------

    root = tk.Tk()
    root.title("Latent Diffusion Demo")

    # Prompt input
    prompt_var = tk.StringVar(
        value="blue pokemon pikachu is a pokemon character with a big smile"
    )

    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")

    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    # Prompt label + entry
    ttk.Label(frame, text="Prompt:").grid(row=0, column=0, sticky="w")
    prompt_entry = ttk.Entry(frame, textvariable=prompt_var, width=60)
    prompt_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))

    # Status label
    status_var = tk.StringVar(value="Idle")
    status_label = ttk.Label(frame, textvariable=status_var)
    status_label.grid(row=2, column=0, sticky="w", pady=(0, 10))

    # Image display
    img_label = ttk.Label(frame)
    img_label.grid(row=3, column=0, columnspan=2, pady=(0, 10))

    # Make columns stretch
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=0)

    # -------------- BUTTON HANDLER (WITH THREAD) --------------

    def run_generation_thread():
        import random
        """Worker thread: run sampling and then update the UI."""
        try:
            prompt = prompt_var.get()
            status_var.set("Sampling...")
            btn_generate.config(state=tk.DISABLED)

            # You can tweak steps/eta/seed here if you want sliders later
            seed = random.randint(0, 2**31 - 1)
            pil_img = generate_image(prompt, steps=1000, eta=0.0, seed=seed)

            # Resize for display if huge
            display_img = pil_img.copy()
            display_img.thumbnail((512, 512))

            tk_img = ImageTk.PhotoImage(display_img)

            # Update GUI on the main thread
            def update_ui():
                img_label.configure(image=tk_img)
                img_label.image = tk_img  # keep reference
                status_var.set("Done")
                btn_generate.config(state=tk.NORMAL)

            root.after(0, update_ui)
        except Exception as e:
            print("[GUI] Error during generation:", e)

            def show_error():
                status_var.set(f"Error: {e}")
                btn_generate.config(state=tk.NORMAL)

            root.after(0, show_error)

    def on_generate_clicked():
        # Run diffusion in a separate thread so the window doesn't freeze
        threading.Thread(target=run_generation_thread, daemon=True).start()

    btn_generate = ttk.Button(frame, text="Generate", command=on_generate_clicked)
    btn_generate.grid(row=2, column=1, sticky="e")

    # Start the GUI loop
    root.mainloop()
