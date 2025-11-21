import torch 
import torch.nn as nn 
import os

from datetime import datetime
from tqdm import tqdm

from src.autoencoder.autoencoder import Autoencoder


class AutoencoderTrainer():
    def __init__(self, model: Autoencoder, config, train_dataloader, val_dataloader = None, device = "cuda"):
        self.model = model.to(device)
        self.l1 = nn.L1Loss()

        self.config = config 
        self.device = device

        self.learning_rate = config["learning_rate"]
        self.kl_weight = float(config["kl_weight"])
        self.epochs = config["epochs"]

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9)
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = val_dataloader

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(self.config["save_dir"], timestamp)

        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self):
        print(f"Starting AE training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for batch in pbar:
                loss, recon_loss, kl_loss = self.train_step(batch)

                epoch_loss += loss
                epoch_recon += recon_loss
                epoch_kl += kl_loss

                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "kl": f"{kl_loss:.4f}"
                })

            # Save model every epoch
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, f"autoencoder_epoch_{epoch+1}.pt")
            )

            print(f"[Epoch {epoch+1}] Loss: {epoch_loss/len(self.train_dataloader):.4f}, "
                  f"Recon: {epoch_recon/len(self.train_dataloader):.4f}, "
                  f"KL: {epoch_kl/len(self.train_dataloader):.6f}")

    def train_step(self, x):
        x = x.to(self.device)

        x_recon, z, mu, log_var = self.model(x)

        # Reconstruction loss (L1)
        recon_loss = torch.nn.functional.l1_loss(x_recon, x)

        # KL divergence
        kl = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(),
            dim=[1,2,3]
        ).mean()

        loss = recon_loss + self.kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kl.item()

