# feature_dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import cv2


class HandcraftedDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["label_a"])

        # 1. Load image and resize
        img = Image.open(img_path).convert("RGB")
        img = img.resize((256, 256))
        img_np = np.array(img)

        # SIGNAL 1: FREQUENCY DOMAIN (FFT)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1)
        fft_feat = cv2.resize(magnitude, (32, 32)).flatten() / 255.0

        # SIGNAL 2: TEXTURE DOMAIN (Sobel Gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        texture_stats = np.array([
            mag.mean(), mag.std(), np.percentile(mag, 25),
            np.percentile(mag, 75), np.percentile(mag, 95)
        ]) / 100.0

        # SIGNAL 3: COLOR / NOISE STATISTICS
        b, g, r = cv2.split(img_np)
        color_stats = np.array([
            r.mean(), g.mean(), b.mean(),
            r.std(), g.std(), b.std()
        ]) / 255.0

        # COLLECTIVE FUSION
        # 1024 (FFT) + 5 (Texture) + 6 (Color) = 1035 features
        combined_features = np.concatenate([fft_feat, texture_stats, color_stats])

        return torch.tensor(combined_features, dtype=torch.float32), label