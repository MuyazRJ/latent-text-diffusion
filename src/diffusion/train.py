import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any
from torch.amp import GradScaler, autocast 

from src.unet.model import SOTADiffusion
from src.diffusion.forward import q_sample

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
            predicted_noise, var_raw = self.model(noisy_images, timesteps, captions)

            # Compute loss
            loss = self.compute_loss(predicted_noise, noise)

            # map raw var prediction [-1,1] → [0,1]
            frac = (var_raw + 1) / 2
            frac = frac.clamp(0,1)

            # get per-sample beta_t values
            beta_t = self.betas[timesteps].view(-1, 1, 1, 1)
            alpha_bar_t = self.alpha_bars[timesteps].view(-1, 1, 1, 1)

            # compute alpha_bar_prev safely
            alpha_bar_prev = self.alpha_bars[(timesteps - 1).clamp(min=0)].view(-1,1,1,1)

            # posterior variance (tilde beta)
            tilde_beta_t = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

            # bounds for variance (exact 2021 logic)
            min_log = torch.log(tilde_beta_t)
            max_log = torch.log(beta_t)

            # predicted log σ²_t
            model_log_sigma2 = frac * max_log + (1 - frac) * min_log

            # Variational KL term between true and predicted variance
            # KL(N(0, σ²_true) || N(0, σ²_pred))
            kl_loss = 0.5 * (model_log_sigma2 - min_log).mean()

            # Weight for KL (small weight gives stable results)
            loss = loss + self.kl_weight * kl_loss
        
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
            pass
            #self.ema_model.update(self.model)

        self.global_step += 1

        return loss.item()

    def train(self, vae, text_encoder, tokenizer):
        print(f"Starting diffusion training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0

            for images, captions in pbar:
                images = images.to(self.device)
                inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
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
        from src.diffusion.sampling.ddpm import reverse
        if epoch % 5 == 0 or epoch == self.epochs - 1:
            images, captions = next(iter(self.train_dataloader))
            images = images.to(self.device)
            inputs = tokenizer(captions, padding=True, truncation=True, return_tensors="pt").to(self.device)
            text_embeddings = text_encoder(**inputs).last_hidden_state
            latents = vae.encode(images).latent_dist.sample()
            latents = reverse(model=self.model, alphas=1 - self.betas, alpha_bars=self.alpha_bars, betas=self.betas, T=self.timesteps, image_size=latents.shape, device=self.device, context=text_embeddings)
            from matplotlib import pyplot as plt
            with torch.no_grad():
                decoded_images = vae.decode(latents).sample
                decoded_images = (decoded_images.clamp(-1,1) + 1) / 2
                decoded_images = decoded_images.cpu()

                # Plot first image in the batch
                plt.figure(figsize=(4,4))
                plt.imshow(decoded_images[0].permute(1,2,0))  # CHW -> HWC
                plt.axis("off")
                plt.title(f"Epoch {epoch+1}")
                plt.show()

                # Save model every epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"diffusion_epoch_{epoch+1}.pt")
        )
