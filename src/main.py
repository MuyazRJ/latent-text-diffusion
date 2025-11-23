import torch 

from src.utils.config_loader import load_config
from src.load.data_loader import PokemonCaptionDataset

from src.autoencoder.autoencoder import Autoencoder
from src.autoencoder.train import AutoencoderTrainer

from src.diffusion.train import DiffusionTrainer
from src.unet.model import SOTADiffusion

from src.diffusion.schedules import make_beta_schedule, compute_alphas

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_config = load_config("./configs/autoencoder.yaml")["autoencoder"]
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")["data"]

    dataset = PokemonCaptionDataset(data_config["captions_file"])
    dataloader = dataset.get_dataloader(data_config["batch_size"])

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.float16, use_safetensors=True)

    text_encoder = text_encoder.to(device)
    context_dim = text_encoder.config.hidden_size

    model = SOTADiffusion(model_config["unet"], context_dim=context_dim, latent_channels=ae_config["latent_dim"])

    betas = make_beta_schedule(
        schedule_type=model_config["training"]["beta_schedule"],
        num_steps=model_config["training"]["time_steps"],
        start=model_config["training"]["beta_start"],
        end=model_config["training"]["beta_end"],
    ).to(device).float()
    alphas, alpha_bars = compute_alphas(betas)

    diffusion_trainer = DiffusionTrainer(
        model=model,
        config=model_config["training"],
        betas=betas,
        alpha_bars=alpha_bars,
        train_dataloader=dataloader,
        device=device
    )
    
    if ae_config["use_pretrained"]:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    else:
        vae = Autoencoder(ae_config=ae_config).to(device)
        if ae_config["train"]:
            train_loader = DataLoaderBuilder(data_config).load()
            trainer  = AutoencoderTrainer(vae, ae_config, train_loader)
            trainer.train()
        else:
            ckpt = torch.load(ae_config["checkpoint"], map_location=device)
            vae.load_state_dict(ckpt)
    
    vae.eval()

    diffusion_trainer.train(vae, text_encoder, tokenizer)

if __name__ == "__main__":
    main()