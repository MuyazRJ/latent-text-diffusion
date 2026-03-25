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
        # Image preprocessing pipeline used before feeding images into the model
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        
        # Stores pairs of image paths and captions
        self.data = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line in f:
                img_path, caption = line.strip().split("\t", 1)
                caption = caption.strip()
                img_path = img_path.strip()
                self.data.append((img_path, caption))

    def __len__(self):
        # Returns total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Loads a single image-caption pair by index
        img_path, caption = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, caption

    def get_dataloader(self, batch_size=8, shuffle=True, num_workers=4):
        # Creates a DataLoader for batching and shuffling the dataset
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# cub_dataset.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
from pathlib import Path


class BirdCaptionDataset(Dataset):
    def __init__(
        self,
        captions_file: str,
        root_dir: str | None = None,
        augment: bool = False,
        verify_size: bool = False,
        return_path: bool = False,
    ):
        """
        captions_file: path to captions.txt where each line is:
            <image_path>\t<caption>

        root_dir: optional path to your CUB_200_2011 folder.

        augment: whether to apply augmentation.

        verify_size: whether to check image dimensions.

        return_path: whether to also return the image path.
        """
        # Stores dataset settings
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.verify_size = verify_size
        self.return_path = return_path

        # Build transform pipeline for the bird images
        tfms = []

        if augment:
            tfms.append(transforms.RandomHorizontalFlip(p=0.5))

        tfms += [
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        self.transform = transforms.Compose(tfms)

        # Stores pairs of image paths and captions
        self.data: list[tuple[str, str]] = []
        with open(captions_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    img_path, caption = line.split("\t", 1)
                except ValueError:
                    raise ValueError(
                        f"Bad line {line_num} in {captions_file}: expected '<path>\\t<caption>'"
                    )

                img_path = img_path.strip()
                caption = caption.strip()
                self.data.append((img_path, caption))

        # Makes sure the dataset is not empty
        if len(self.data) == 0:
            raise ValueError(f"No samples found in {captions_file}")

    def __len__(self):
        # Returns total number of samples in the dataset
        return len(self.data)

    def _resolve_path(self, img_path: str) -> Path:
        # Resolves image paths relative to the dataset root directory if needed
        p = Path(img_path)
        if self.root_dir is not None and not p.is_absolute():
            p = self.root_dir / p
        return p

    def __getitem__(self, idx):
        # Loads a single image-caption pair by index
        img_path, caption = self.data[idx]
        p = self._resolve_path(img_path)

        if not p.exists():
            raise FileNotFoundError(f"Missing image: {p}")

        # Opens the image, fixes orientation, and converts it to RGB
        image = Image.open(p)
        image = ImageOps.exif_transpose(image).convert("RGB")

        # Optionally checks that the image size matches the expected dimensions
        if self.verify_size:
            if image.size != (256, 256):
                raise ValueError(f"Expected 256x256, got {image.size} for {p}")

        # Apply preprocessing transforms
        image = self.transform(image)

        # Optionally include the file path in the returned sample
        if self.return_path:
            return image, caption, str(p)
        return image, caption

    def get_dataloader(self, batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True):
        # Creates a DataLoader for batching, shuffling, and loading samples efficiently
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )