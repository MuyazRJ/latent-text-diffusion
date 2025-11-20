import torch 

from utils.config_loader import load_config

from src.autoencoder.autoencoder import Autoencoder

def main():
    ae_config = load_config("configs/autoencoder.yaml")
    model_config = load_config("configs/model.yaml")

    autoencoder = Autoencoder(ae_config=ae_config)

if __name__ == "__main__":
    main()