# data_prep.py
import glob
from datasets import load_dataset
from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("./Defactify_Images")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_shards(split_name):
    """Load ALL parquet shards for a split (train or test)."""
    pattern = f"./Defactify/data/{split_name}-*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found for split '{split_name}'")
    print(f"  Loading {len(files)} shard(s) for {split_name}")
    return load_dataset("parquet", data_files=files, split="train")


def extract_balanced(split_name, n_per_class, start_offset=0):
    """Extract n_per_class Real + n_per_class AI images from a split."""
    print(f"Loading {split_name} split...")
    ds = load_shards(split_name)
    print(f"  Total examples in {split_name}: {len(ds)}")

    rows = []
    seen_real = 0
    seen_ai = 0
    i = start_offset

    while (seen_real < n_per_class or seen_ai < n_per_class) and i < len(ds):
        ex = ds[i]
        label_a = str(ex["Label_A"])

        if label_a == "0" and seen_real < n_per_class:
            seen_real += 1
        elif label_a == "1" and seen_ai < n_per_class:
            seen_ai += 1
        else:
            i += 1
            continue

        img_id = f"{split_name}_{i:06d}"
        img_path = OUTPUT_DIR / f"{img_id}.jpg"
        if not img_path.exists():
            ex["Image"].convert("RGB").save(img_path, quality=90)

        rows.append({
            "id": img_id,
            "path": str(img_path),
            "label_a": label_a,
            "label_b": str(ex["Label_B"]),
            "caption": str(ex["Caption"]),
        })

        if len(rows) % 200 == 0:
            print(f"  Collected {len(rows)} ({seen_real} real, {seen_ai} ai)")

        i += 1

    print(f"  Done: {seen_real} real, {seen_ai} ai")
    return rows


print("=== Extracting balanced data ===")

train_rows = extract_balanced("train", n_per_class=5000, start_offset=0)
val_rows   = extract_balanced("train", n_per_class=500,  start_offset=10000)
test_rows  = extract_balanced("test",  n_per_class=500,  start_offset=0)

Path("data/manifests").mkdir(parents=True, exist_ok=True)
pd.DataFrame(train_rows).to_csv("data/manifests/train_manifest.csv", index=False)
pd.DataFrame(val_rows).to_csv("data/manifests/val_manifest.csv", index=False)
pd.DataFrame(test_rows).to_csv("data/manifests/test_manifest.csv", index=False)

print(f"\nDone. Train: {len(train_rows)}, Val: {len(val_rows)}, Test: {len(test_rows)}")