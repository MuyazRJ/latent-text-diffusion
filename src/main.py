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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae_config = load_config("./configs/autoencoder.yaml")["autoencoder"]
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")["data"]

    # dataset = PokemonCaptionDataset(data_config["captions_file"])
    # dataloader = dataset.get_dataloader(data_config["batch_size"])

    dataset = BirdCaptionDataset(
        captions_file=data_config["captions_file"],
        root_dir=data_config.get("root_dir", "data/CUB_200_2011"),
    )
    dataloader = dataset.get_dataloader(data_config["batch_size"], shuffle=True, num_workers=4)

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
    _, alpha_bars = compute_alphas(betas)

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

    #diffusion_trainer.train(vae, text_encoder, tokenizer)
    diffusion_trainer.load_checkpoint()
    #diffusion_trainer.generate_from_captions_file(vae, text_encoder, tokenizer)
    diffusion_trainer.generate_samples(vae, text_encoder, tokenizer, num_samples=9, prompts=["Anna Hummingbird, bill shape needle, wing color grey, wing color black, upperparts color grey, upperparts color black", "", "", "Common Yellowthroat, bill shape all-purpose, wing brown, wing grey, wing buff, upperparts brown", "Least Tern, bill shape dagger, wing grey, upperparts grey, underparts white, breast solid", "Least Tern, bill shape all-purpose, wing brown, upperparts black, underparts buff, breast striped", "Blue Grosbeak", "Blue Grosbeak, wing color blue, upperparts color blue, underparts color blue, breast pattern solid, back color blue", "Blue Grosbeak, wing color blue, wing color brown, wing color black, upperparts color blue, upperparts color white"])

    # --------- LOAD DIFFUSION CHECKPOINT FOR INFERENCE ---------
    # If you have an existing checkpoint:
    # diffusion_trainer.load_checkpoint("path/to/checkpoint.pt")

    # Choose which model to use for inference:
    #  - raw model: diffusion_trainer.model
    #  - EMA model: diffusion_trainer.ema.get_model()
    inference_model = diffusion_trainer.get_ema_model()
    #inference_model = diffusion_trainer.model
    inference_model.to(device).eval()

    # --------- CREATE AND LAUNCH GUI ---------
    # For SD VAE at 256x256, latent shape is typically (4, 32, 32)
    latent_shape = (ae_config["latent_dim"], 32, 32)  # adjust if your latent res is different

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