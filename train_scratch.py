# train_scratch.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score
from feature_dataset import HandcraftedDataset
from scratch_model import ScratchMLP
from tqdm import tqdm
import os


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for feats, labels in tqdm(loader, desc="Train"):
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(feats)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds,
                                                                                     average="binary")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for feats, labels in tqdm(loader, desc="Eval"):
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds,
                                                                                     average="binary")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs("models", exist_ok=True)

    train_ds = HandcraftedDataset("data/manifests/train_manifest.csv")
    val_ds = HandcraftedDataset("data/manifests/val_manifest.csv")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    model = ScratchMLP(input_size=1035, num_classes=2).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device))

    num_epochs = 15
    best_f1 = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "models/best_scratch_model.pth")
            print(f"  → Saved best model (F1: {best_f1:.4f})")

    print(f"\nTraining complete. Best Val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()