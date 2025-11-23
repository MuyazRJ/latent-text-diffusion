# pokemon_dataset.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class PokemonCaptionDataset(Dataset):
    def __init__(self, captions_file):
        """
        captions_file: path to your captions.txt where each line is:
        <image_path>\t<caption>
        """
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        
        self.data = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                img_path, caption = line.strip().split("\t")
                self.data.append((img_path, caption))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption

    def get_dataloader(self, batch_size=8, shuffle=True, num_workers=4):
        """Return a PyTorch DataLoader directly from the dataset."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
