from pathlib import Path

import pandas as pd

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
output_path = Path(__file__).parent / "output" / "wacv2016.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
