# app.py
import gradio as gr
import torch
from PIL import Image
from model import CLIPDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CLIPDetector().to(device)
model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
model.eval()

processor = model.processor


def classify_image(image):
    """Predict Real vs AI-Generated."""
    if image is None:
        return {"Real": 0.0, "AI-Generated": 0.0}

    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Predict
    with torch.no_grad():
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1)[0]

    return {
        "Real": float(probs[0]),
        "AI-Generated": float(probs[1])
    }


# Build Gradio interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="Defactify: AI-Generated Image Detector",
    description="Upload an image to check if it's Real or AI-Generated.",
    examples=[
        "Defactify_Images/train_000000.jpg",
        "Defactify_Images/train_000001.jpg"
    ]
)

if __name__ == "__main__":
    demo.launch()