# app_ensemble.py
import gradio as gr
import torch
from PIL import Image
import numpy as np
import cv2

from model import CLIPDetector
from scratch_model import ScratchMLP
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CLIP Model
clip_model = CLIPDetector().to(device)
clip_model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
clip_model.eval()
clip_processor = clip_model.processor

# 2. Load Scratch Model
scratch_model = ScratchMLP().to(device)
scratch_model.load_state_dict(torch.load("models/best_scratch_model.pth", map_location=device))
scratch_model.eval()


def extract_features(image):
    img = image.convert("RGB").resize((256, 256))
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    fft_feat = cv2.resize(magnitude, (32, 32)).flatten() / 255.0

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    texture_stats = np.array([
        mag.mean(), mag.std(), np.percentile(mag, 25),
        np.percentile(mag, 75), np.percentile(mag, 95)
    ]) / 100.0

    b, g, r = cv2.split(img_np)
    color_stats = np.array([
        r.mean(), g.mean(), b.mean(),
        r.std(), g.std(), b.std()
    ]) / 255.0

    return torch.tensor(np.concatenate([fft_feat, texture_stats, color_stats]), dtype=torch.float32).unsqueeze(0)


def classify_image(image):
    if image is None:
        return {"Real": 0.0, "AI-Generated": 0.0}

    # CLIP Prediction
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        c_logits = clip_model(inputs["pixel_values"].to(device))
        c_prob = torch.softmax(c_logits, dim=1)[0, 1].item()

    # Scratch Prediction
    feats = extract_features(image).to(device)
    with torch.no_grad():
        s_logits = scratch_model(feats)
        s_prob = torch.softmax(s_logits, dim=1)[0, 1].item()

    # Collective Decision
    avg_prob = (c_prob + s_prob) / 2.0

    return {
        "Real": float(1.0 - avg_prob),
        "AI-Generated": float(avg_prob)
    }


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=2, label="Collective Decision"),
    title="Defactify: Collective AI Detector",
    description="Uses a Scratch DIP Model + CLIP Model to make a collective decision.",
    examples=[]
)

if __name__ == "__main__":
    demo.launch()