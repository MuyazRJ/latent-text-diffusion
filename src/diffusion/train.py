import torch
import torch.nn as nn
import os
import random
import math 
import torchvision

from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any
from torch.amp import GradScaler, autocast 
from torchvision.utils import make_grid

from src.unet.model import SOTADiffusion
from src.diffusion.forward import q_sample
from src.utils.image import save_image_grid 
from src.diffusion.sampling.ddpm import reverse
from src.diffusion.sampling.ddim import reverse_ddim_ldm
from src.diffusion.ema import EMA

class DiffusionTrainer:
    def __init__(self, model: SOTADiffusion, config: Dict[str, Any], betas, alpha_bars, train_dataloader, val_dataloader=None, device="cuda"):
        """
        model: your SOTADiffusion model
        config: training config dict
        train_dataloader: DataLoader for training
        val_dataloader: optional validation DataLoader
        """
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.epochs = config["epochs"]
        self.learning_rate = float(config["lr"])
        self.timesteps = config["time_steps"]  # number of diffusion steps
        self.loss_type = config.get("loss_type", "l2")  # 'l1' or 'l2'

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(config["save_dir"], timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.betas = betas
        self.alpha_bars = alpha_bars

        self.kl_weight = config.get("kl_weight", 1.0)
        self.scaler = GradScaler(self.device)

        self.global_step = 0
        self.global_step_ema = config.get("global_step_ema", 1000)
        
        self.ema = EMA(self.model, decay=config.get("ema_decay", 0.9999))

    def sample_timesteps(self, batch_size):
        """Randomly sample timesteps for DDPM training"""
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def compute_loss(self, predicted_noise, target_noise):
        """Standard DDPM noise prediction loss"""
        if self.loss_type == "l1":
            return nn.functional.l1_loss(predicted_noise, target_noise)
        else:
            return nn.functional.mse_loss(predicted_noise, target_noise)

    def train_step(self, images, captions=None):
        images = images.to(self.device)
        if captions is not None: captions = captions.to(self.device)

        batch_size = images.shape[0]
        timesteps = self.sample_timesteps(batch_size)

        # Create noisy input
        noisy_images, noise = q_sample(images, timesteps, self.alpha_bars)

        self.optimizer.zero_grad()

        with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            # Model outputs TWO things: (eps_pred, var_raw)
            predicted_noise = self.model(noisy_images, timesteps, captions)

            # Compute loss
            loss = self.compute_loss(predicted_noise, noise)
        
        # Backward + step with AMP
        self.scaler.scale(loss).backward()

        # 1. unscale gradients
        self.scaler.unscale_(self.optimizer)

        # 2. now clip them safely
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # 3. optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.global_step > self.global_step_ema:
            self.ema.update(self.model)

        self.global_step += 1

        return loss.item()

    def train(self, vae, text_encoder, tokenizer):
        print(f"Starting diffusion training for {self.epochs} epochs...")

        # === CRITICAL: FREEZE VAE & TEXT ENCODER ===
        # If you don't do this, you will run out of memory immediately.
        vae.eval()
        vae.requires_grad_(False)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

        if self.config["resume_from_checkpoint"]:
            self.load_checkpoint()

        for epoch in range(self.epochs):
            self.ema.get_model().train()
            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0

            for images, captions in pbar:
                with torch.no_grad():
                    images = images.to(self.device)

                    # 20% CAPTION DROPOUT (CFG PREPARATION)
                    captions_list = list(captions)
                    # per-sample (better quality)
                    clean_captions = []
                    for cap in captions_list:
                        if random.random() < 0.2: # 20% chance to drop
                            clean_captions.append("")
                        else:
                            clean_captions.append(cap)

                    inputs = tokenizer(clean_captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    text_embeddings = text_encoder(**inputs).last_hidden_state
                    latents = vae.encode(images).latent_dist.sample() * 0.18215

                loss = self.train_step(latents, text_embeddings)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            self.log_metrics(epoch_loss, epoch, vae, text_encoder, tokenizer)

    def log_metrics(self, epoch_loss, epoch, vae, text_encoder, tokenizer):
        """Log metrics to console or a logging service"""
        avg_loss = epoch_loss / len(self.train_dataloader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == self.epochs - 1:
            # Use EMA model for sampling if it's warm, otherwise raw model
            if self.global_step > self.global_step_ema:
                model = self.ema.get_model()
            else:
                model = self.model
                
            model.eval()
            vae.eval()
            text_encoder.eval()

            # === SAMPLE A BATCH FROM TRAIN ===
            images, captions = next(iter(self.train_dataloader))
            images = images.to(self.device)

            batch_size = images.size(0)
            num_pairs = min(7, batch_size)  # up to 7 examples

            # Random indices for images/captions
            idx = torch.randperm(batch_size, device=images.device)[:num_pairs]
            sel_images = images[idx]

            # Index captions in Python space (captions is usually a list of strings)
            idx_list = idx.cpu().tolist()
            sel_captions = [captions[i] for i in idx_list]

            # Override the 7th caption (if we have 7 samples)
            if num_pairs == 7:
                sel_captions[6] = ""

            with torch.no_grad():
                # === TEXT ENCODING ===
                inputs = tokenizer(
                    sel_captions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                text_embeddings = text_encoder(**inputs).last_hidden_state  # [B, L, D]

                # === ENCODE TRUE IMAGES TO LATENTS ===
                true_latents = vae.encode(sel_images).latent_dist.sample()  # [B, C, H', W']

                # Decode originals for top row
                decoded_true = vae.decode(true_latents).sample  # [B, 3, H, W]
                decoded_true = ((decoded_true.clamp(-1, 1) + 1) / 2).cpu()  # [B, 3, H, W]

                # ---- GENERATION ----
                bottom_images = []

                # 1) First image: DDIM with 1000 steps
                latents_1000 = true_latents[0:1]
                emb_1000 = text_embeddings[0:1]
                sampled_1000_scaled = reverse_ddim_ldm(
                    model=model,
                    alpha_bars=self.alpha_bars,
                    T=self.timesteps,
                    image_size=latents_1000.shape,
                    device=self.device,
                    context=emb_1000,
                    num_steps=1000,
                    eta=0.0,
                )
                sampled_1000_unscaled = sampled_1000_scaled / 0.18215
                decoded_1000 = vae.decode(sampled_1000_unscaled).sample  # [1, 3, H, W]
                decoded_1000 = ((decoded_1000.clamp(-1, 1) + 1) / 2).cpu()
                bottom_images.append(decoded_1000[0])  # [3, H, W]

                # 2) Remaining images (if any): DDIM with 50 steps
                if num_pairs > 1:
                    latents_50 = true_latents[1:num_pairs]
                    emb_50 = text_embeddings[1:num_pairs]

                    sampled_50_scaled = reverse_ddim_ldm(
                        model=model,
                        alpha_bars=self.alpha_bars,
                        T=self.timesteps,
                        image_size=latents_50.shape,
                        device=self.device,
                        context=emb_50,
                        num_steps=50,
                        eta=0.0,
                    )
                    sampled_50_unscaled = sampled_50_scaled / 0.18215
                    decoded_50 = vae.decode(sampled_50_unscaled).sample  # [num_pairs-1, 3, H, W]
                    decoded_50 = ((decoded_50.clamp(-1, 1) + 1) / 2).cpu()

                    for i in range(num_pairs - 1):
                        bottom_images.append(decoded_50[i])  # each [3, H, W]

                bottom_row = torch.stack(bottom_images, dim=0)          # [num_pairs, 3, H, W]
                top_row = decoded_true[:num_pairs]                       # [num_pairs, 3, H, W]

                # Combine: first all originals, then all generated
                combined = torch.cat([top_row, bottom_row], dim=0)       # [2*num_pairs, 3, H, W]

                save_image_grid(
                    combined,
                    out_dir=os.path.join(self.save_dir, f"recon_epoch_{epoch}.png"),
                    nrow=num_pairs  # top row originals, bottom row generated
                )

            # === SAVE MODEL + EMA ===
            torch.save({
                "model": self.model.state_dict(),        # training weights
                "ema_model": self.ema.state_dict(),      # EMA weights
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(self.save_dir, f"checkpoint_{epoch}.pt"))

            print(
                f"Saved EMA checkpoint and preview grid for epoch {epoch} "
                f"(col 1: 1000-step, cols 2–{num_pairs}: 50-step; col 7 uses Pikachu caption if present)."
            )

    def load_checkpoint(self):
        """Load model, optimizer, AND EMA state from checkpoint"""
        checkpoint_path = self.config["load_checkpoint"]
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 1. Load Standard Weights
        self.model.load_state_dict(checkpoint["model"])
        
        # 2. Load Optimizer
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # 3. Load EMA Weights (Crucial for resuming!)
        # We check if the key exists just in case you load an old checkpoint
        if "ema_model" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_model"])
            print("EMA state loaded successfully.")
        else:
            print("Warning: No EMA state found in checkpoint. Starting EMA from scratch.")

        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming at epoch {start_epoch}.")
        return start_epoch

    def generate_samples(self, vae, text_encoder, tokenizer, prompts=None, num_samples=50):
        """
        Generate samples conditioned on captions (or provided prompts) and
        save a single grid image of [original, generated] pairs.

        - Uses first batch from val_dataloader if available, otherwise train_dataloader
        - Saves ONE image file: samples_grid.png
        """
        self.model.eval()
        vae.eval()
        text_encoder.eval()

        # Pick a dataloader to pull "original" images from
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader

        # Get one batch
        images, captions = next(iter(dataloader))
        images = images.to(self.device)

        # Limit to num_samples and also to batch size
        batch_size = images.size(0)
        num_samples = min(num_samples, batch_size, 10)  # hard cap at 10 as requested
        images = images[:num_samples]
        captions = captions[:num_samples]

        # If user provided prompts, override captions (trim to num_samples)
        if prompts is not None and len(prompts) > 0:
            prompts = list(prompts)[:num_samples]
        else:
            prompts = list(captions)

        with torch.no_grad():
            # === TEXT ENCODING ===
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            text_embeddings = text_encoder(**inputs).last_hidden_state  # [B, L, D]

            # === ENCODE TRUE IMAGES TO LATENTS ===
            true_latents = vae.encode(images).latent_dist.sample()

            # === SAMPLE FROM DIFFUSION MODEL IN LATENT SPACE ===
            inference_model = self.ema.get_model()
            #inference_model =self.model
            inference_model.eval()

            sampled_latents_scaled = reverse_ddim_ldm(
                model=inference_model,  # <--- CHANGED from self.model to inference_model
                alpha_bars=self.alpha_bars,
                T=self.timesteps,
                image_size=true_latents.shape,
                device=self.device,
                context=text_embeddings,
                num_steps=1000,
                eta=0.0,
            )

            # Unscale for SD VAE decode
            sampled_latents_unscaled = sampled_latents_scaled / 0.18215

            # === DECODE BOTH LATENTS BACK TO IMAGE SPACE ===
            decoded_pred = vae.decode(sampled_latents_unscaled).sample
            decoded_true = vae.decode(true_latents).sample

            # Map from [-1, 1] → [0, 1]
            decoded_pred = ((decoded_pred.clamp(-1, 1) + 1) / 2).cpu()
            decoded_true = ((decoded_true.clamp(-1, 1) + 1) / 2).cpu()

        # === BUILD ONE BIG GRID: [orig0, gen0, orig1, gen1, ...] ===
        pairs = []
        for i in range(num_samples):
            pairs.append(decoded_true[i])
            pairs.append(decoded_pred[i])

        # Shape: [2 * num_samples, C, H, W]
        combined = torch.stack(pairs, dim=0)

        out_path = os.path.join(self.save_dir, "samples_grid.png")
        # 2 columns: [original | generated] per row
        save_image_grid(
            combined,
            out_dir=out_path,
            nrow=2
        )

        print(f"Saved {num_samples} [original, generated] pairs in one grid at: {out_path}")
    
    def get_ema_model(self):
        return self.ema.get_model()

    @torch.no_grad()
    def vae_recon_sanity_check(
        vae,
        dataloader,
        device,
        out_path="vae_recon_grid.png",
        max_images=8,
        latent_scale=0.18215,
    ):
        vae.eval()

        images, captions = next(iter(dataloader))
        images = images.to(device)  # expected in [-1, 1]
        images = images[:max_images]

        # Encode -> scale to "diffusion latent space"
        latents = vae.encode(images).latent_dist.sample() * latent_scale

        # Decode -> IMPORTANT: scale back before decode
        recon = vae.decode(latents / latent_scale).sample  # typically in [-1, 1]

        # Metrics (in [-1,1] space)
        mse = torch.mean((recon - images) ** 2).item()
        psnr = 10.0 * math.log10(4.0 / mse) if mse > 0 else float("inf")  # range is 2 => max^2=4
        print(f"VAE recon MSE:  {mse:.6f}")
        print(f"VAE recon PSNR: {psnr:.2f} dB")

        # Save side-by-side grid: originals on top, recon on bottom
        # Convert from [-1,1] -> [0,1] for saving
        orig_vis = (images.clamp(-1, 1) + 1) / 2
        recon_vis = (recon.clamp(-1, 1) + 1) / 2

        grid = make_grid(torch.cat([orig_vis, recon_vis], dim=0), nrow=max_images, padding=2)
        torchvision.utils.save_image(grid, out_path)
        print(f"Saved: {out_path}")

        # Quick shape checks
        print("images:", tuple(images.shape))
        print("latents:", tuple(latents.shape))
        print("recon:", tuple(recon.shape))


    def _read_captions_file(self, captions_path: str):
        """
        Supports lines like:
        image/path.jpg<TAB>caption...
        or just:
        caption...
        Returns: list[str] captions
        """
        captions = []
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    _, cap = line.split("\t", 1)
                    cap = cap.strip()
                else:
                    cap = line
                if cap:
                    captions.append(cap)
        return captions


    def generate_from_captions_file(
        self,
        vae,
        text_encoder,
        tokenizer,
        captions_path: str = "data/captions.txt",
        steps: int = 50,
        batch_size: int = 8,
        max_prompts: int | None = None,
        out_dir: str | None = None,
        seed: int = 0,
        eta: float = 0.0,
        latent_scale: float = 0.18215,  # matches your decode unscale
    ):
        """
        Generates ONE image per caption in captions.txt using DDIM with steps=50.

        - Infers latent shape from one batch of your dataloader (so it matches training resolution).
        - Uses EMA weights (same as your generate_samples).
        - Saves: gen_00000.png, gen_00001.png, ... + prompts_used.txt
        """
        from torchvision.utils import save_image
        from pathlib import Path
        self.model.eval()
        vae.eval()
        text_encoder.eval()

        # Where to save
        if out_dir is None:
            out_dir = os.path.join(self.save_dir, f"caption_generations_steps{steps}")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Read captions
        prompts = self._read_captions_file(captions_path)
        if max_prompts is not None:
            prompts = prompts[:max_prompts]

        if len(prompts) == 0:
            raise ValueError(f"No captions found in: {captions_path}")

        # Infer latent shape from a real batch (so we don't guess H/W)
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        if dataloader is None:
            raise ValueError("No dataloader found (val_dataloader/train_dataloader are both None).")

        images, _ = next(iter(dataloader))
        images = images.to(self.device)

        with torch.no_grad():
            example_latents = vae.encode(images[:1]).latent_dist.sample()
        latent_shape_1 = example_latents.shape[1:]  # (C, H, W)

        # Use EMA model for inference (as in your code)
        inference_model = self.ema.get_model()
        inference_model.eval()

        # Reproducibility
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # Save prompt index file (useful later for debugging / FID bookkeeping)
        with open(out_path / "prompts_used.txt", "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                f.write(f"{i}\t{p}\n")

        total = len(prompts)
        idx = 0

        for start in tqdm(range(0, total, batch_size), desc="Generating", unit="batch"):
            batch_prompts = prompts[start : start + batch_size]
            B = len(batch_prompts)

            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                text_embeddings = text_encoder(**inputs).last_hidden_state  # [B, L, D]

            sampled_latents_scaled = reverse_ddim_ldm(
                model=inference_model,
                alpha_bars=self.alpha_bars,
                T=self.timesteps,
                image_size=(B, *latent_shape_1),
                device=self.device,
                context=text_embeddings,
                num_steps=steps,   # 50
                eta=eta,
            )

            sampled_latents_unscaled = sampled_latents_scaled / latent_scale

            with torch.no_grad():
                decoded = vae.decode(sampled_latents_unscaled).sample
                decoded = ((decoded.clamp(-1, 1) + 1) / 2)

            for b in range(B):
                img = decoded[b].detach().cpu()
                save_image(img, out_path / f"gen_{idx:05d}.png")
                idx += 1

        print(f"Saved {idx} generated images to: {str(out_path)}")
        print(f"Prompts saved to: {str(out_path / 'prompts_used.txt')}")