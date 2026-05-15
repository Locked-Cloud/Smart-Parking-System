"""
============================================================
 STEP 4: Review and Correct Auto-Labels with labelme
============================================================
Opens labelme pointed at your dataset so you can:
   - Add boxes the auto-labeler missed
   - Delete wrong boxes (false positives)
   - Tighten loose boxes
   - Make sure every visible toy car is labeled "vehicle"

CONTROLS in labelme:
   Ctrl + R     = create rectangle bbox
   D            = next image
   A            = previous image
   Del          = delete selected bbox
   Ctrl + S     = save

After reviewing, this script auto-converts labelme JSON → YOLO .txt

Run:
    python 4_review_labels.py
============================================================
"""

import subprocess
import sys
import json
import shutil
from pathlib import Path

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "data" / "toy_dataset"

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

CLASSES = ["vehicle"]          # add more if needed


# ── Install helpers ──────────────────────────────────────

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])


def ensure_labelme():
    try:
        import labelme  # noqa
        print("[OK] labelme already installed.")
    except ImportError:
        print("[INFO] Installing labelme …")
        pip_install("labelme")
        print("[OK] labelme installed.")


def find_exe(name):
    exe = shutil.which(name)
    if exe:
        return exe
    # Windows Scripts folder fallback
    fallback = Path(sys.executable).parent / "Scripts" / f"{name}.exe"
    if fallback.exists():
        return str(fallback)
    print(f"[ERROR] Could not find '{name}' executable.")
    print(f"  Try:  pip install {name}")
    sys.exit(1)


# ── YOLO pre-annotations → labelme JSON ─────────────────

def yolo_to_labelme_json(image_path: Path, label_path: Path, classes: list) -> dict:
    """
    Convert one YOLO .txt file to a labelme-compatible JSON dict
    so existing boxes show up when you open the image.
    """
    from PIL import Image as PILImage
    img = PILImage.open(image_path)
    W, H = img.size

    shapes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            label = classes[cls_id] if cls_id < len(classes) else str(cls_id)
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            shapes.append({
                "label": label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": H,
        "imageWidth": W,
    }


def pre_annotate(images_dir: Path, labels_dir: Path, classes: list):
    """Write labelme JSON files next to each image so existing YOLO boxes are visible."""
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"[INFO] Pre-annotating {len(images)} images …")
    for img_path in images:
        json_path = images_dir / (img_path.stem + ".json")
        if json_path.exists():
            continue                          # don't overwrite manual edits
        label_path = labels_dir / (img_path.stem + ".txt")
        data = yolo_to_labelme_json(img_path, label_path, classes)
        json_path.write_text(json.dumps(data, indent=2))
    print("[OK] Pre-annotation done.")


# ── labelme JSON → YOLO .txt ─────────────────────────────

def labelme_to_yolo(images_dir: Path, labels_dir: Path, classes: list):
    """
    After you close labelme, convert its JSON files back to YOLO .txt
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    jsons = list(images_dir.glob("*.json"))
    print(f"[INFO] Converting {len(jsons)} JSON files → YOLO …")

    for json_path in jsons:
        data = json.loads(json_path.read_text())
        W = data["imageWidth"]
        H = data["imageHeight"]
        lines = []
        for shape in data["shapes"]:
            if shape["shape_type"] != "rectangle":
                continue
            label = shape["label"]
            if label not in classes:
                print(f"  [WARN] Unknown label '{label}' in {json_path.name} — skipping")
                continue
            cls_id = classes.index(label)
            (x1, y1), (x2, y2) = shape["points"]
            cx = ((x1 + x2) / 2) / W
            cy = ((y1 + y2) / 2) / H
            bw = abs(x2 - x1) / W
            bh = abs(y2 - y1) / H
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt_path = labels_dir / (json_path.stem + ".txt")
        txt_path.write_text("\n".join(lines))

    # Write classes.txt
    (labels_dir / "classes.txt").write_text("\n".join(classes) + "\n")
    print("[OK] YOLO labels written.")


# ── Launch labelme ───────────────────────────────────────

def launch_labelme(images_dir: Path, labels_dir: Path, classes: list):
    # Pre-annotate so existing boxes are visible
    try:
        pre_annotate(images_dir, labels_dir, classes)
    except Exception as e:
        print(f"[WARN] Pre-annotation skipped ({e}). Pillow not installed?")
        pip_install("Pillow")
        pre_annotate(images_dir, labels_dir, classes)

    labelme_exe = find_exe("labelme")

    # Build a temporary labels file for labelme
    labels_file = labels_dir / "labelme_classes.txt"
    labels_file.write_text("\n".join(classes) + "\n")

    cmd = [
        labelme_exe,
        str(images_dir),
        "--labels",    str(labels_file),
        "--autosave",
        "--nodata",    # don't embed image bytes in JSON (keeps files small)
    ]

    print("\n[INFO] Launching labelme …")
    print("  • Use Ctrl+R to draw a rectangle")
    print("  • Use D / A to navigate images")
    print("  • Boxes are auto-saved on image change\n")
    print("Command:", " ".join(cmd), "\n")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] labelme exited with code {e.returncode}")
        sys.exit(1)

    # Convert back to YOLO after labelme closes
    print("\n[INFO] Converting annotations back to YOLO format …")
    labelme_to_yolo(images_dir, labels_dir, classes)


# ── Main ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" STEP 4: Review labels with labelme")
    print("=" * 60)

    if not TRAIN_IMAGES.exists():
        print(f"[ERROR] Missing folder:\n  {TRAIN_IMAGES}")
        print("Run 3_auto_label.py first.")
        sys.exit(1)

    ensure_labelme()

    # ── TRAIN ──
    print("\n" + "=" * 60)
    print(" REVIEWING TRAIN SPLIT")
    print("=" * 60)
    launch_labelme(TRAIN_IMAGES, TRAIN_LABELS, CLASSES)

    # ── VALID ──
    print("\n" + "=" * 60)
    print(" REVIEWING VALIDATION SPLIT")
    print("=" * 60)
    launch_labelme(VALID_IMAGES, VALID_LABELS, CLASSES)

    print("\n" + "=" * 60)
    print(" Review finished.")
    print("=" * 60)
    print("\nNext step:")
    print("    python 5_prepare_dataset.py")


if __name__ == "__main__":
    main()