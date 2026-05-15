"""
============================================================
 STEP 5: Prepare data.yaml for YOLO Training
============================================================
After auto-labeling and review, generates the data.yaml file
that YOLO needs for training, with absolute paths (Windows-safe).

Run:
    python 5_prepare_dataset.py
============================================================
"""

import sys
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "toy_dataset"
DATA_YAML = PROJECT_ROOT / "data" / "data.yaml"


def main():
    print("=" * 60)
    print(" STEP 5: Generate data.yaml")
    print("=" * 60)

    train_images = DATASET_DIR / "train" / "images"
    val_images = DATASET_DIR / "valid" / "images"
    train_labels = DATASET_DIR / "train" / "labels"
    val_labels = DATASET_DIR / "valid" / "labels"

    if not train_images.exists():
        print(f"[ERROR] {train_images} does not exist.")
        print("        Run 3_auto_label.py first.")
        sys.exit(1)

    n_train_img = len(list(train_images.glob("*.jpg")))
    n_val_img = len(list(val_images.glob("*.jpg")))
    n_train_lbl = len([p for p in train_labels.glob("*.txt")
                       if p.name != "classes.txt" and p.stat().st_size > 0])
    n_val_lbl = len([p for p in val_labels.glob("*.txt")
                     if p.name != "classes.txt" and p.stat().st_size > 0])

    print(f"\n[INFO] Train images : {n_train_img}   (with labels: {n_train_lbl})")
    print(f"[INFO] Val   images : {n_val_img}   (with labels: {n_val_lbl})")

    if n_train_lbl == 0:
        print("\n[ERROR] No labeled images found in train split.")
        print("        Did you run 4_review_labels.py and save?")
        sys.exit(1)

    yaml_data = {
        "path": str(DATASET_DIR.resolve()).replace("\\", "/"),
        "train": str(train_images.resolve()).replace("\\", "/"),
        "val": str(val_images.resolve()).replace("\\", "/"),
        "nc": 1,
        "names": ["vehicle"],
    }

    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print(f"\n[OK] Wrote: {DATA_YAML}")
    print("-" * 60)
    print(DATA_YAML.read_text())
    print("-" * 60)
    print("\nNext step:  python 6_finetune_model.py")


if __name__ == "__main__":
    main()
