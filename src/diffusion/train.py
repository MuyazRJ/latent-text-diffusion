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
        model: SOTADiffusion model
        config: training config dict
        train_dataloader: DataLoader for training
        val_dataloader: optional validation DataLoader
        """
        # Move the model onto the selected device
        self.model = model.to(device)
        self.device = device
        self.config = config

        # Store main training settings from the config
        self.epochs = config["epochs"]
        self.learning_rate = float(config["lr"])
        self.timesteps = config["time_steps"]
        self.loss_type = config.get("loss_type", "l2")

        # Optimiser used for training the diffusion model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Create a timestamped save directory for checkpoints and outputs
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(config["save_dir"], timestamp)
        os.makedirs(self.save_dir, exist_ok=True)

        # Store diffusion schedule values
        self.betas = betas
        self.alpha_bars = alpha_bars

        self.kl_weight = config.get("kl_weight", 1.0)

        # Gradient scaler used for mixed precision training
        self.scaler = GradScaler(self.device)

        # Track the total number of optimisation steps
        self.global_step = 0
        self.global_step_ema = config.get("global_step_ema", 1000)
        
        # Exponential moving average model used for more stable inference
        self.ema = EMA(self.model, decay=config.get("ema_decay", 0.9999))

    def sample_timesteps(self, batch_size):
        # Randomly sample a timestep for each item in the batch
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def compute_loss(self, predicted_noise, target_noise):
        # Computes either L1 or L2 noise prediction loss
        if self.loss_type == "l1":
            return nn.functional.l1_loss(predicted_noise, target_noise)
        else:
            return nn.functional.mse_loss(predicted_noise, target_noise)

    def train_step(self, images, captions=None):
        # Move images to the correct device
        images = images.to(self.device)

        # Move caption embeddings if they are provided
        if captions is not None:
            captions = captions.to(self.device)

        batch_size = images.shape[0]

        # Sample random timesteps for the current batch
        timesteps = self.sample_timesteps(batch_size)

        # Add noise to the clean images according to the diffusion schedule
        noisy_images, noise = q_sample(images, timesteps, self.alpha_bars)

        # Reset gradients before the backward pass
        self.optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
            predicted_noise = self.model(noisy_images, timesteps, captions)

            # Compare predicted noise against the true sampled noise
            loss = self.compute_loss(predicted_noise, noise)
        
        # Backpropagate the scaled loss
        self.scaler.scale(loss).backward()

        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)

        # Clip gradients to improve training stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Apply optimiser step and update scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Start updating the EMA model after the warmup threshold
        if self.global_step > self.global_step_ema:
            self.ema.update(self.model)

        self.global_step += 1

        return loss.item()

    def train(self, vae, text_encoder, tokenizer):
        print(f"Starting diffusion training for {self.epochs} epochs...")

        # Set VAE and text encoder to evaluation mode and freeze their parameters
        vae.eval()
        vae.requires_grad_(False)
        text_encoder.eval()
        text_encoder.requires_grad_(False)

        # Resume training if a checkpoint is specified
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

                    # Apply caption dropout for classifier-free guidance training
                    captions_list = list(captions)
                    clean_captions = []
                    for cap in captions_list:
                        if random.random() < 0.2:
                            clean_captions.append("")
                        else:
                            clean_captions.append(cap)

                    # Tokenise captions and encode them using CLIP
                    inputs = tokenizer(clean_captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    text_embeddings = text_encoder(**inputs).last_hidden_state

                    # Encode images into latent space using the VAE
                    latents = vae.encode(images).latent_dist.sample() * 0.18215

                # Run one optimisation step on the diffusion model
                loss = self.train_step(latents, text_embeddings)
                epoch_loss += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

            self.log_metrics(epoch_loss, epoch, vae, text_encoder, tokenizer)

    def log_metrics(self, epoch_loss, epoch, vae, text_encoder, tokenizer):
        # Compute and print average loss for the epoch
        avg_loss = epoch_loss / len(self.train_dataloader)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

        # Periodically generate previews and save checkpoints
        if epoch % 5 == 0 or epoch == self.epochs - 1:
            # Use EMA weights for sampling once available
            if self.global_step > self.global_step_ema:
                model = self.ema.get_model()
            else:
                model = self.model
                
            model.eval()
            vae.eval()
            text_encoder.eval()

            # Take one batch from the training set for preview generation
            images, captions = next(iter(self.train_dataloader))
            images = images.to(self.device)

            batch_size = images.size(0)
            num_pairs = min(7, batch_size)

            # Randomly select examples for the preview grid
            idx = torch.randperm(batch_size, device=images.device)[:num_pairs]
            sel_images = images[idx]

            idx_list = idx.cpu().tolist()
            sel_captions = [captions[i] for i in idx_list]

            # Replace the final caption with an empty prompt when 7 samples are used
            if num_pairs == 7:
                sel_captions[6] = ""

            with torch.no_grad():
                # Encode the selected captions
                inputs = tokenizer(
                    sel_captions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                text_embeddings = text_encoder(**inputs).last_hidden_state

                # Encode selected images into latent space
                true_latents = vae.encode(sel_images).latent_dist.sample()

                # Decode originals so they can be shown in the preview grid
                decoded_true = vae.decode(true_latents).sample
                decoded_true = ((decoded_true.clamp(-1, 1) + 1) / 2).cpu()

                bottom_images = []

                # Generate the first sample using 1000 DDIM steps
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
                decoded_1000 = vae.decode(sampled_1000_unscaled).sample
                decoded_1000 = ((decoded_1000.clamp(-1, 1) + 1) / 2).cpu()
                bottom_images.append(decoded_1000[0])

                # Generate the remaining samples using 50 DDIM steps
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
                    decoded_50 = vae.decode(sampled_50_unscaled).sample
                    decoded_50 = ((decoded_50.clamp(-1, 1) + 1) / 2).cpu()

                    for i in range(num_pairs - 1):
                        bottom_images.append(decoded_50[i])

                # Arrange originals on the top row and generated outputs on the bottom row
                bottom_row = torch.stack(bottom_images, dim=0)
                top_row = decoded_true[:num_pairs]
                combined = torch.cat([top_row, bottom_row], dim=0)

                save_image_grid(
                    combined,
                    out_dir=os.path.join(self.save_dir, f"recon_epoch_{epoch}.png"),
                    nrow=num_pairs
                )

            # Save training model state, EMA weights, optimiser state, and epoch number
            torch.save({
                "model": self.model.state_dict(),
                "ema_model": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(self.save_dir, f"checkpoint_{epoch}.pt"))

            print(
                f"Saved EMA checkpoint and preview grid for epoch {epoch} "
                f"(col 1: 1000-step, cols 2–{num_pairs}: 50-step; col 7 uses Pikachu caption if present)."
            )

    def load_checkpoint(self):
        # Load model, optimiser, and EMA state from a checkpoint file
        checkpoint_path = self.config["load_checkpoint"]
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load main model weights
        self.model.load_state_dict(checkpoint["model"])
        
        # Load optimiser state for resumed training
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Load EMA weights if they are present in the checkpoint
        if "ema_model" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_model"])
            print("EMA state loaded successfully.")
        else:
            print("Warning: No EMA state found in checkpoint. Starting EMA from scratch.")

        start_epoch = checkpoint.get("epoch", 0)
        print(f"Resuming at epoch {start_epoch}.")
        return start_epoch

    def generate_samples(self, vae, text_encoder, tokenizer, prompts=None, num_samples=10):
        """
        Generate samples conditioned on captions (or provided prompts) and:
        (1) save a single grid image of [original, generated] pairs
        (2) save each original and generated image separately

        - Uses first batch from val_dataloader if available, otherwise train_dataloader
        - Saves:
            samples_grid.png
            samples/orig_000.png, samples/gen_000.png, ...
            (optional) samples/pair_000.png, ...
        """
        import os
        import torch
        from torchvision.utils import save_image

        # Set all models to evaluation mode for inference
        self.model.eval()
        vae.eval()
        text_encoder.eval()

        # Use validation data if available, otherwise fall back to training data
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader

        # Get a batch of real images and captions
        images, captions = next(iter(dataloader))
        images = images.to(self.device)

        # Restrict the number of samples to the batch size and the hard cap
        batch_size = images.size(0)
        num_samples = min(num_samples, batch_size, 10)
        images = images[:num_samples]
        captions = captions[:num_samples]

        # Use provided prompts if given, otherwise use dataset captions
        if prompts is not None and len(prompts) > 0:
            prompts = list(prompts)[:num_samples]
        else:
            prompts = list(captions)

        with torch.no_grad():
            # Convert prompts into text embeddings
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            text_embeddings = text_encoder(**inputs).last_hidden_state

            # Encode the original images into latent space
            true_latents = vae.encode(images).latent_dist.sample()

            # Use EMA weights for sample generation
            inference_model = self.ema.get_model()
            inference_model.eval()

            # Generate latent samples using DDIM
            sampled_latents_scaled = reverse_ddim_ldm(
                model=inference_model,
                alpha_bars=self.alpha_bars,
                T=self.timesteps,
                image_size=true_latents.shape,
                device=self.device,
                context=text_embeddings,
                num_steps=1000,
                eta=0.0,
            )

            # Undo latent scaling before decoding
            sampled_latents_unscaled = sampled_latents_scaled / 0.18215

            # Decode both generated and original latents into image space
            decoded_pred = vae.decode(sampled_latents_unscaled).sample
            decoded_true = vae.decode(true_latents).sample

            # Convert outputs from [-1,1] to [0,1] for saving
            decoded_pred = ((decoded_pred.clamp(-1, 1) + 1) / 2).cpu()
            decoded_true = ((decoded_true.clamp(-1, 1) + 1) / 2).cpu()

        # Create output directory for generated samples
        samples_dir = os.path.join(self.save_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        for i in range(num_samples):
            # Save original and generated image separately
            orig_path = os.path.join(samples_dir, f"orig_{i:03d}.png")
            gen_path  = os.path.join(samples_dir, f"gen_{i:03d}.png")

            save_image(decoded_true[i], orig_path)
            save_image(decoded_pred[i], gen_path)

        # Build a grid in the order [orig0, gen0, orig1, gen1, ...]
        pairs = []
        for i in range(num_samples):
            pairs.append(decoded_true[i])
            pairs.append(decoded_pred[i])

        combined = torch.stack(pairs, dim=0)

        out_path = os.path.join(self.save_dir, "samples_grid.png")
        save_image_grid(
            combined,
            out_dir=out_path,
            nrow=2
        )

        print(f"Saved grid at: {out_path}")
        print(f"Saved individual images in: {samples_dir} (orig_###.png, gen_###.png)")

    def get_ema_model(self):
        # Returns the EMA version of the model for inference
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
        # Set VAE to evaluation mode
        vae.eval()

        # Take one batch of images and limit the number shown
        images, captions = next(iter(dataloader))
        images = images.to(device)
        images = images[:max_images]

        # Encode images into latent space and apply latent scaling
        latents = vae.encode(images).latent_dist.sample() * latent_scale

        # Decode the scaled latents back into image space
        recon = vae.decode(latents / latent_scale).sample

        # Compute reconstruction quality metrics
        mse = torch.mean((recon - images) ** 2).item()
        psnr = 10.0 * math.log10(4.0 / mse) if mse > 0 else float("inf")
        print(f"VAE recon MSE:  {mse:.6f}")
        print(f"VAE recon PSNR: {psnr:.2f} dB")

        # Convert images into [0,1] range before saving
        orig_vis = (images.clamp(-1, 1) + 1) / 2
        recon_vis = (recon.clamp(-1, 1) + 1) / 2

        # Save originals on the top row and reconstructions on the bottom row
        grid = make_grid(torch.cat([orig_vis, recon_vis], dim=0), nrow=max_images, padding=2)
        torchvision.utils.save_image(grid, out_path)
        print(f"Saved: {out_path}")

        # Print tensor shapes for quick inspection
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
        # Read captions from file and support both path-caption and caption-only formats
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
        latent_scale: float = 0.18215,
    ):
        """
        Generates ONE image per caption in captions.txt using DDIM with steps=50.

        - Infers latent shape from one batch of your dataloader (so it matches training resolution).
        - Uses EMA weights (same as your generate_samples).
        - Saves: gen_00000.png, gen_00001.png, ... + prompts_used.txt
        """
        from torchvision.utils import save_image
        from pathlib import Path

        # Set all models to evaluation mode
        self.model.eval()
        vae.eval()
        text_encoder.eval()

        # Create output directory for generated images
        if out_dir is None:
            out_dir = os.path.join(self.save_dir, f"caption_generations_steps{steps}")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Read prompts from the captions file
        prompts = self._read_captions_file(captions_path)
        if max_prompts is not None:
            prompts = prompts[:max_prompts]

        if len(prompts) == 0:
            raise ValueError(f"No captions found in: {captions_path}")

        # Use a real batch to infer the latent shape
        dataloader = self.val_dataloader if self.val_dataloader is not None else self.train_dataloader
        if dataloader is None:
            raise ValueError("No dataloader found (val_dataloader/train_dataloader are both None).")

        images, _ = next(iter(dataloader))
        images = images.to(self.device)

        with torch.no_grad():
            example_latents = vae.encode(images[:1]).latent_dist.sample()
        latent_shape_1 = example_latents.shape[1:]

        # Use the EMA model for generation
        inference_model = self.ema.get_model()
        inference_model.eval()

        # Set up a generator with a fixed seed
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # Save the prompt list alongside the generated outputs
        with open(out_path / "prompts_used.txt", "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                f.write(f"{i}\t{p}\n")

        total = len(prompts)
        idx = 0

        # Process prompts in batches
        for start in tqdm(range(0, total, batch_size), desc="Generating", unit="batch"):
            batch_prompts = prompts[start : start + batch_size]
            B = len(batch_prompts)

            # Tokenise prompts and move them to the correct device
            inputs = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                text_embeddings = text_encoder(**inputs).last_hidden_state

            # Generate latent samples from the prompts
            sampled_latents_scaled = reverse_ddim_ldm(
                model=inference_model,
                alpha_bars=self.alpha_bars,
                T=self.timesteps,
                image_size=(B, *latent_shape_1),
                device=self.device,
                context=text_embeddings,
                num_steps=steps,
                eta=eta,
            )

            # Undo scaling before decoding with the VAE
            sampled_latents_unscaled = sampled_latents_scaled / latent_scale

            with torch.no_grad():
                decoded = vae.decode(sampled_latents_unscaled).sample
                decoded = ((decoded.clamp(-1, 1) + 1) / 2)

            # Save each generated image individually
            for b in range(B):
                img = decoded[b].detach().cpu()
                save_image(img, out_path / f"gen_{idx:05d}.png")
                idx += 1

        print(f"Saved {idx} generated images to: {str(out_path)}")
        print(f"Prompts saved to: {str(out_path / 'prompts_used.txt')}")