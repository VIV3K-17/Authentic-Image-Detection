# dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DefactifyDataset(Dataset):
    """Binary real/fake classification dataset."""

    def __init__(self, manifest_path, transform=None):
        import pandas as pd
        self.df = pd.read_csv(manifest_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        label = int(row["label_a"])  # 0=Real, 1=AI-Generated

        if self.transform:
            image = self.transform(image)

        return image, label