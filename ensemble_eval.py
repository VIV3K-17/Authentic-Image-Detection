# ensemble_eval.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from PIL import Image
import numpy as np
import cv2

from dataset import DefactifyDataset
from model import CLIPDetector
from feature_dataset import HandcraftedDataset
from scratch_model import ScratchMLP


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP
    clip_ds = DefactifyDataset("data/manifests/test_manifest.csv")
    clip_loader = DataLoader(clip_ds, batch_size=32, shuffle=False)
    clip_model = CLIPDetector().to(device)
    clip_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    clip_model.eval()

    # Load Scratch
    scratch_ds = HandcraftedDataset("data/manifests/test_manifest.csv")
    scratch_loader = DataLoader(scratch_ds, batch_size=32, shuffle=False)
    scratch_model = ScratchMLP().to(device)
    scratch_model.load_state_dict(torch.load("models/best_scratch_model.pth", map_location=device))
    scratch_model.eval()

    all_labels = []
    clip_preds = []
    scratch_preds = []
    ensemble_preds = []

    print("Running Collective Decision...")
    for (clip_imgs, labels), (scratch_feats, _) in zip(clip_loader, scratch_loader):
        labels_np = labels.numpy()
        all_labels.extend(labels_np)

        # CLIP probs
        with torch.no_grad():
            c_logits = clip_model(clip_imgs.to(device))
            c_probs = torch.softmax(c_logits, dim=1)[:, 1].cpu().numpy()

        # Scratch probs
        with torch.no_grad():
            s_logits = scratch_model(scratch_feats.to(device))
            s_probs = torch.softmax(s_logits, dim=1)[:, 1].cpu().numpy()

        clip_preds.extend((c_probs > 0.5).astype(int))
        scratch_preds.extend((s_probs > 0.5).astype(int))

        # Collective Decision (Average Probability)
        avg_probs = (c_probs + s_probs) / 2.0
        ensemble_preds.extend((avg_probs > 0.5).astype(int))

    print("\n=== COLLECTIVE DECISION RESULTS ===")
    print(f"CLIP Model F1:    {f1_score(all_labels, clip_preds, average='binary'):.4f}")
    print(f"Scratch Model F1: {f1_score(all_labels, scratch_preds, average='binary'):.4f}")
    print(f"Ensemble F1:      {f1_score(all_labels, ensemble_preds, average='binary'):.4f}")

    print("\n=== Ensemble Classification Report ===")
    print(classification_report(all_labels, ensemble_preds, target_names=["Real", "AI"]))


if __name__ == "__main__":
    main()