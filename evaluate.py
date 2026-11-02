# evaluate.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from dataset import DefactifyDataset
from model import CLIPDetector
import numpy as np


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = DefactifyDataset("data/manifests/test_manifest.csv")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = CLIPDetector().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n=== Test Set Results ===")
    print(classification_report(all_labels, all_preds, target_names=["Real", "AI-Generated"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()