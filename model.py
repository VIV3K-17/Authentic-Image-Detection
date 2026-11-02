# model.py
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class CLIPDetector(nn.Module):
    """
    CLIP ViT backbone + lightweight classification head.
    Freeze CLIP; train only the head for fast iteration.
    """

    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", num_classes=2):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)

        # Freeze CLIP backbone
        for param in self.clip.parameters():
            param.requires_grad = False

        hidden_size = self.clip.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    def forward(self, pixel_values):
        # pixel_values: [B, 3, 224, 224]
        with torch.no_grad():
            outputs = self.clip(pixel_values=pixel_values)
            # Use CLS token (pooler_output)
            features = outputs.pooler_output  # [B, hidden_size]

        logits = self.classifier(features)
        return logits