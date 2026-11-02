from datasets import load_dataset
from pathlib import Path

# Pick the split you actually want
dataset = load_dataset("parquet", data_dir="./Defactify", split="train")

# Only take a small preview
N = 200
subset = dataset.select(range(min(N, len(dataset))))

output_dir = Path("./Defactify_Images")
output_dir.mkdir(exist_ok=True)

for i, example in enumerate(subset):
    label_a = str(example["Label_A"]).replace("/", "-").replace(" ", "_")
    label_b = str(example["Label_B"]).replace("/", "-").replace(" ", "_")
    filename = f"{i:05d}_{label_a}_{label_b}.jpg"
    example["Image"].convert("RGB").save(output_dir / filename, quality=90)
    if i % 25 == 0:
        print(f"Saved {i}/{N}...")

print(f"Done. Saved {len(subset)} images to {output_dir}")