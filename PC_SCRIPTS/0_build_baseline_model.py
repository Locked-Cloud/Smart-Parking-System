"""
============================================================
 STEP 0 (OPTIONAL): Build a Baseline Model RIGHT NOW
============================================================
Lets you skip ahead to a working RPi4 system before you've
captured your own toy car photos. We download a pretrained
YOLOv8n model (COCO) and export it to NCNN at 256x256 - the
same input size your fine-tuned model will use later.

Toy cars look enough like real cars that this model will
detect them well enough to test the whole pipeline today.
You can replace it later by running the full 1->7 workflow.

Run:
    py -3.13 0_build_baseline_model.py
============================================================
"""

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RPI4_MODELS = PROJECT_ROOT / "RPI4_SCRIPTS" / "models"
RPI4_MODELS.mkdir(parents=True, exist_ok=True)
IMG_SIZE = 256


def main():
    print("=" * 60)
    print(" STEP 0: Build Baseline NCNN Model (YOLOv8n COCO)")
    print("=" * 60)

    from ultralytics import YOLO
    print("\n[INFO] Downloading YOLOv8n weights (auto-downloads if needed)...")
    model = YOLO("yolov8n.pt")

    print(f"[INFO] Exporting to NCNN at {IMG_SIZE}x{IMG_SIZE}...")
    ncnn_path = model.export(format="ncnn", imgsz=IMG_SIZE)
    print(f"[OK] NCNN export written to: {ncnn_path}")

    src = Path(ncnn_path)
    if src.is_file():
        src = src.parent

    target = RPI4_MODELS / "yolo11n_ncnn_model"   # keep filename the RPi side expects
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    print(f"\n[OK] Baseline model copied to: {target}")

    print("\n" + "=" * 60)
    print(" Baseline model ready!")
    print("=" * 60)
    print("\nYou can now flash RPI4_SCRIPTS/ to your RPi4 and run the system.")
    print("Toy cars will be detected as 'car' class via COCO weights.")
    print("\nIMPORTANT: parking_system.py already has class 0 + COCO ids in its")
    print("'vehicle_classes' set, so this baseline works without changes.")
    print("\nTo improve accuracy on your specific toy cars, run the full")
    print("workflow:  1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7")


if __name__ == "__main__":
    main()
