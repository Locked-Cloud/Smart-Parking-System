"""
============================================================
 STEP 3: Auto-Pre-Label Toy Car Images
============================================================
Uses a STRONG YOLOv8s model (trained on COCO) to automatically
draw bounding boxes around toy cars. Toy cars look enough like
real cars that the COCO model picks most of them up - this gives
you 80%+ of your annotations for FREE.

Output: YOLO-format .txt files next to every image.

YOU MUST review and correct these labels in step 4 using
labelImg before training. The auto-labels are a starting point.

We detect any of: car, truck, bus, motorcycle  -> remap to class 0 (vehicle)

Run:
    python 3_auto_label.py
============================================================
"""

import sys
import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_images"
DATASET_DIR = PROJECT_ROOT / "data" / "toy_dataset"

# Splits
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "valid"
VAL_RATIO = 0.15

# Auto-label settings
AUTOLABEL_CONF = 0.20   # low confidence on purpose - catch more, you'll correct
COCO_VEHICLE_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck


def main():
    print("=" * 60)
    print(" STEP 3: Auto-Pre-Label Toy Car Images with YOLOv8s")
    print("=" * 60)

    images = sorted(RAW_DIR.glob("*.jpg"))
    if not images:
        print(f"[ERROR] No images found in {RAW_DIR}")
        print("        Run 2_capture_training_data.py first.")
        sys.exit(1)
    print(f"[INFO] Found {len(images)} images.")

    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        sys.exit(1)

    # Use YOLOv8s (small) - more accurate auto-labeler than nano
    print("[INFO] Loading YOLOv8s (auto-downloads on first run)...")
    model = YOLO("yolov8s.pt")

    # Prepare output dirs
    for d in [TRAIN_DIR / "images", TRAIN_DIR / "labels",
              VAL_DIR / "images", VAL_DIR / "labels"]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.seed(42)
    shuffled = images.copy()
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_RATIO))
    val_set = set(p.name for p in shuffled[:n_val])
    print(f"[INFO] Train: {len(shuffled) - n_val} images   Val: {n_val} images")

    n_with_labels = 0
    n_total_boxes = 0

    for i, img_path in enumerate(images, 1):
        is_val = img_path.name in val_set
        out_img_dir = (VAL_DIR if is_val else TRAIN_DIR) / "images"
        out_lbl_dir = (VAL_DIR if is_val else TRAIN_DIR) / "labels"

        # Copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)

        # Run model
        results = model(str(img_path), conf=AUTOLABEL_CONF, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            # Empty label file - tells YOLO this image has no objects
            (out_lbl_dir / (img_path.stem + ".txt")).write_text("")
            continue

        h, w = results.orig_shape
        lines = []
        for b in results.boxes:
            cls_id = int(b.cls[0])
            if cls_id not in COCO_VEHICLE_IDS:
                continue
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            # Remap all vehicle classes -> 0 ("vehicle")
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines))
        if lines:
            n_with_labels += 1
            n_total_boxes += len(lines)

        if i % 25 == 0:
            print(f"   ... {i} / {len(images)} images processed")

    print("\n" + "=" * 60)
    print(f" Auto-labeling complete.")
    print(f"   Images with at least 1 detection : {n_with_labels} / {len(images)}")
    print(f"   Total bounding boxes drawn       : {n_total_boxes}")
    print("=" * 60)
    print("""
NEXT (IMPORTANT):
   The auto-labels are NOT perfect. You MUST review them.

   Run:    python 4_review_labels.py
   This opens labelImg so you can quickly correct boxes.

   Then:   python 5_prepare_dataset.py
""")


if __name__ == "__main__":
    main()
