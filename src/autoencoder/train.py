# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Training utility for variational autoencoder learning in latent diffusion.
# Defines the AutoencoderTrainer class, which handles reconstruction and KL
# divergence optimisation, epoch-based training, progress logging, and
# checkpoint saving for an autoencoder used to compress images into a latent
# representation for downstream diffusion modelling.
#
# Designed for training the latent autoencoder component of a latent diffusion
# pipeline, where reconstruction fidelity and latent regularisation must be
# balanced to produce a compact and well-structured latent space.
#
# Based on:
# - Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022
#   https://arxiv.org/abs/2112.10752

import torch 
import torch.nn as nn 
import os

from datetime import datetime
from tqdm import tqdm

from src.autoencoder.autoencoder import Autoencoder


class AutoencoderTrainer():
    def __init__(self, model: Autoencoder, config, train_dataloader, val_dataloader = None, device = "cuda"):
        # Move the model to the selected device
        self.model = model.to(device)

        # L1 loss is used for reconstruction quality
        self.l1 = nn.L1Loss()

        self.config = config 
        self.device = device

        # Store key training settings
        self.learning_rate = config["learning_rate"]
        self.kl_weight = float(config["kl_weight"])
        self.epochs = config["epochs"]

        # Optimiser used to train the autoencoder
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.5, 0.9)
        )

        self.train_dataloader = train_dataloader
        self.test_dataloader = val_dataloader

        # Create a timestamped folder for saving checkpoints
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = os.path.join(self.config["save_dir"], timestamp)

        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self):
        print(f"Starting AE training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")

            # Track total losses across the epoch
            epoch_loss = 0
            epoch_recon = 0
            epoch_kl = 0

            for batch in pbar:
                # Run one training step on the current batch
                loss, recon_loss, kl_loss = self.train_step(batch)

                epoch_loss += loss
                epoch_recon += recon_loss
                epoch_kl += kl_loss

                # Show live loss values in the progress bar
                pbar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "recon": f"{recon_loss:.4f}",
                    "kl": f"{kl_loss:.4f}"
                })

            # Save model weights after each epoch
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir, f"autoencoder_epoch_{epoch+1}.pt")
            )

            # Print average losses for the epoch
            print(f"[Epoch {epoch+1}] Loss: {epoch_loss/len(self.train_dataloader):.4f}, "
                  f"Recon: {epoch_recon/len(self.train_dataloader):.4f}, "
                  f"KL: {epoch_kl/len(self.train_dataloader):.6f}")

    def train_step(self, x):
        # Move input batch to the selected device
        x = x.to(self.device)

        # Forward pass through the autoencoder
        x_recon, z, mu, log_var = self.model(x)

        # Reconstruction loss measures how close the output is to the input
        recon_loss = torch.nn.functional.l1_loss(x_recon, x)

        # KL divergence regularises the latent space distribution
        kl = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp(),
            dim=[1,2,3]
        ).mean()

        # Total loss combines reconstruction and KL terms
        loss = recon_loss + self.kl_weight * kl

        # Standard optimisation step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), recon_loss.item(), kl.item()