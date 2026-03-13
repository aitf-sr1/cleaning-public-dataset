import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output" / "engagement"


@dataclass
class DatasetSplit:
    csv_path: Path
    img_dir: Path


@dataclass
class DatasetConfig:
    name: str
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


DATASETS: list[DatasetConfig] = [
    DatasetConfig(
        name="wacv2016",
        train=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/WACV2016/output/wacv2016_train.csv",
            img_dir=Path(__file__).parent.parent / "datasets/WACV2016/output/Train",
        ),
        val=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/WACV2016/output/wacv2016_val.csv",
            img_dir=Path(__file__).parent.parent / "datasets/WACV2016/output/Val",
        ),
        test=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/WACV2016/output/wacv2016_test.csv",
            img_dir=Path(__file__).parent.parent / "datasets/WACV2016/output/Test",
        ),
    ),
    DatasetConfig(
        name="daisee",
        train=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/engagement_labels/daisee_train.csv",
            img_dir=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/Cleaning_daisee_normal/Train",
        ),
        val=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/engagement_labels/daisee_validation.csv",
            img_dir=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/Cleaning_daisee_normal/Validation",
        ),
        test=DatasetSplit(
            csv_path=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/engagement_labels/daisee_test.csv",
            img_dir=Path(__file__).parent.parent
            / "datasets/DAiSEE/output/Cleaning_daisee_normal/Test",
        ),
    ),
]


def collect_split(datasets: list[DatasetConfig], split: str) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        split_config: DatasetSplit = getattr(dataset, split)
        df = pd.read_csv(split_config.csv_path)
        df["source"] = dataset.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def move_images(datasets: list[DatasetConfig], split: str, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dataset in datasets:
        split_config: DatasetSplit = getattr(dataset, split)
        for img_path in sorted(split_config.img_dir.iterdir()):
            if img_path.is_file():
                shutil.move(str(img_path), dest_dir / img_path.name)
                count += 1
    return count


def merge_split(datasets: list[DatasetConfig], split: str, output_dir: Path) -> None:
    print(f"\n[{split.upper()}]")

    img_dest = output_dir / split.capitalize()
    n_images = move_images(datasets, split, img_dest)
    print(f"  Moved {n_images} images to {img_dest}")

    df = collect_split(datasets, split)
    csv_path = output_dir / f"merged_{split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} rows to {csv_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        merge_split(DATASETS, split, OUTPUT_DIR)


if __name__ == "__main__":
    main()
