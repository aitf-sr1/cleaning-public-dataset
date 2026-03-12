from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

dataset_dir = Path(__file__).parent / "wacv2016" / "dataset"

rows = []
for label_dir in sorted(dataset_dir.iterdir()):
    if not label_dir.is_dir():
        continue
    label = label_dir.name
    for img_file in sorted(label_dir.iterdir()):
        if img_file.is_file():
            rows.append({"img_file_name": img_file.name, "label": label})

df = pd.DataFrame(rows)

train_df, temp_df = (
    cast(pd.DataFrame, x)
    for x in train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
)
val_df, test_df = (
    cast(pd.DataFrame, x)
    for x in train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)

splits = {"train": train_df, "val": val_df, "test": test_df}
for split_name, split_df in splits.items():
    output_path = output_dir / f"wacv2016_{split_name}.csv"
    split_df.to_csv(output_path, index=False)
    print(f"Saved {len(split_df)} rows to {output_path}")
