import torch 

from src.utils.config_loader import load_config
from src.autoencoder.autoencoder import Autoencoder

from src.autoencoder.train import AutoencoderTrainer
from src.load.data_loader import DataLoaderBuilder

from diffusers import AutoencoderKL

def main():
    ae_config = load_config("./configs/autoencoder.yaml")["autoencoder"]
    model_config = load_config("configs/model.yaml")
    data_config = load_config("configs/data.yaml")["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == "__main__":
    main()