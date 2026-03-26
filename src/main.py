# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

import torch 

from src.utils.config_loader import load_config
from src.load.data_loader import BirdCaptionDataset

from src.autoencoder.autoencoder import Autoencoder
from src.autoencoder.train import AutoencoderTrainer

from src.diffusion.train import DiffusionTrainer
from src.unet.model import SOTADiffusion

from src.diffusion.schedules import make_beta_schedule, compute_alphas

from src.app import create_diffusion_gui 

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL

def main():
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config files for the autoencoder, main model, and dataset
    ae_config = load_config("./configs/autoencoder.yaml")["autoencoder"]
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")["data"]

    # dataset = PokemonCaptionDataset(data_config["captions_file"])
    # dataloader = dataset.get_dataloader(data_config["batch_size"])

    # Load the bird caption dataset using the captions file and image root directory
    dataset = BirdCaptionDataset(
        captions_file=data_config["captions_file"],
        root_dir=data_config.get("root_dir", "data/CUB_200_2011"),
    )

    # Create dataloader for batching the dataset during training
    dataloader = dataset.get_dataloader(data_config["batch_size"], shuffle=True, num_workers=4)

    # Load CLIP tokenizer and text encoder for turning text prompts into embeddings
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.float16, use_safetensors=True)

    # Move text encoder to the correct device
    text_encoder = text_encoder.to(device)

    # Get the embedding size from CLIP so the diffusion model knows the context dimension
    context_dim = text_encoder.config.hidden_size

    # Create the diffusion U-Net model
    model = SOTADiffusion(model_config["unet"], context_dim=context_dim, latent_channels=ae_config["latent_dim"])

    # Create beta schedule for the diffusion process
    betas = make_beta_schedule(
        schedule_type=model_config["training"]["beta_schedule"],
        num_steps=model_config["training"]["time_steps"],
        start=model_config["training"]["beta_start"],
        end=model_config["training"]["beta_end"],
    ).to(device).float()

    # Compute cumulative alpha values used in the forward and reverse diffusion steps
    _, alpha_bars = compute_alphas(betas)

    # Set up the diffusion trainer with the model, config, scheduler values, and dataloader
    diffusion_trainer = DiffusionTrainer(
        model=model,
        config=model_config["training"],
        betas=betas,
        alpha_bars=alpha_bars,
        train_dataloader=dataloader,
        device=device
    )
    
    # Decide whether to use a pretrained VAE or the custom autoencoder
    if ae_config["use_pretrained"]:
        # Load pretrained Stable Diffusion VAE
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    else:
        # Create custom autoencoder
        vae = Autoencoder(ae_config=ae_config).to(device)

        # Train the autoencoder if required by the config
        if ae_config["train"]:
            train_loader = DataLoaderBuilder(data_config).load()
            trainer  = AutoencoderTrainer(vae, ae_config, train_loader)
            trainer.train()
        else:
            # Otherwise load the saved checkpoint
            ckpt = torch.load(ae_config["checkpoint"], map_location=device)
            vae.load_state_dict(ckpt)
    
    # Set VAE to eval mode since it is being used for inference
    vae.eval()

    # Train diffusion model if needed
    # diffusion_trainer.train(vae, text_encoder, tokenizer)

    # Load saved diffusion checkpoint instead of training from scratch
    diffusion_trainer.load_checkpoint()

    # Use the EMA version of the model for inference because it usually gives smoother results
    inference_model = diffusion_trainer.get_ema_model()
    inference_model.to(device).eval()

    # Latent shape expected by the model for 256x256 images
    latent_shape = (ae_config["latent_dim"], 32, 32) 

    # Launch the local GUI for generating images
    create_diffusion_gui(
        model=inference_model,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        alpha_bars=alpha_bars,
        timesteps=model_config["training"]["time_steps"],
        device=device,
        latent_shape=latent_shape,
    )

if __name__ == "__main__":
    main()