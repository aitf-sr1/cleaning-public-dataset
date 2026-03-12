from pathlib import Path

import pandas as pd

INPUT_DIR = Path(__file__).parent / "output" / "Cleaning_daisee_normal"
OUTPUT_DIR = Path(__file__).parent / "output" / "engagement_labels"

SPLITS = ["Train", "Validation", "Test"]
LABEL_FILES = {
    "Train": "Cleaned_TrainLabels.csv",
    "Validation": "Cleaned_ValidationLabels.csv",
    "Test": "Cleaned_TestLabels.csv",
}


def extract_engagement(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # change 0 into 1 so it can be merged with WACV2016
    label = df["Engagement"].replace(0, 1)

    return pd.DataFrame(
        {
            "img_file_name": df["Image_Name"],
            "label": label,
        }
    )


def export_split(split_name: str) -> None:
    csv_path = INPUT_DIR / LABEL_FILES[split_name]
    df = extract_engagement(csv_path)

    output_path = OUTPUT_DIR / f"daisee_{split_name.lower()}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in SPLITS:
        export_split(split_name)


if __name__ == "__main__":
    main()
