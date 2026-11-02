import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(f"data/manifests/{split}_manifest.csv")
    counts = df["label_a"].value_counts().to_dict()
    print(f"{split}: {len(df)} rows | label_a counts: {counts}")