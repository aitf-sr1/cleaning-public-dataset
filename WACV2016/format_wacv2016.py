import shutil
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
            rows.append(
                {"img_file_name": img_file.name, "label": label, "img_path": img_file}
            )

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

splits = {"Train": train_df, "Val": val_df, "Test": test_df}
for split_name, split_df in splits.items():
    split_img_dir = output_dir / split_name
    split_img_dir.mkdir(parents=True, exist_ok=True)

    for img_path in split_df["img_path"]:
        shutil.copy(img_path, split_img_dir / Path(img_path).name)

    csv_df = split_df.drop(columns=["img_path"])
    output_path = output_dir / f"wacv2016_{split_name.lower()}.csv"
    csv_df.to_csv(output_path, index=False)
    print(f"Saved {len(csv_df)} rows to {output_path}")
    print(f"Copied {len(split_df)} images to {split_img_dir}")
