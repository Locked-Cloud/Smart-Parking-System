"""
============================================================
 STEP 7: Export Fine-Tuned Model to NCNN (RPi4 format)
============================================================
NCNN = fastest runtime for ARM CPU on RPi4.

Output is auto-copied to RPI4_SCRIPTS/models/yolo11n_ncnn_model/

Run:
    python 7_export_model.py
============================================================
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BEST = PROJECT_ROOT / "runs" / "parking" / "stage2_full" / "weights" / "best.pt"
RPI4_MODELS = PROJECT_ROOT / "RPI4_SCRIPTS" / "models"
RPI4_MODELS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 256   # MUST match training input size


def main():
    print("=" * 60)
    print(" STEP 7: Export model -> NCNN")
    print("=" * 60)

    if not BEST.exists():
        print(f"[ERROR] Trained weights not found at: {BEST}")
        print("        Run 6_finetune_model.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed.")
        sys.exit(1)

    print(f"\n[INFO] Source weights : {BEST}")
    print(f"[INFO] Image size     : {IMG_SIZE}")

    model = YOLO(str(BEST))

    # NCNN export
    print("\n[INFO] Exporting to NCNN ...")
    ncnn_path = model.export(format="ncnn", imgsz=IMG_SIZE)
    print(f"[OK] NCNN export written to: {ncnn_path}")

    # ONNX as fallback
    print("\n[INFO] Exporting to ONNX (fallback) ...")
    try:
        onnx_path = model.export(format="onnx", imgsz=IMG_SIZE, opset=12)
        print(f"[OK] ONNX export: {onnx_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed (non-fatal): {e}")

    # Copy NCNN folder into RPI4_SCRIPTS/models/
    ncnn_src = Path(ncnn_path)
    if ncnn_src.is_file():
        ncnn_src = ncnn_src.parent
    target = RPI4_MODELS / "yolo11n_ncnn_model"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(ncnn_src, target)
    print(f"\n[OK] NCNN model copied to: {target}")

    # Also copy .pt
    shutil.copy2(BEST, RPI4_MODELS / "best.pt")
    print(f"[OK] best.pt copied to:    {RPI4_MODELS / 'best.pt'}")

    print("\n" + "=" * 60)
    print(" Export finished!")
    print("=" * 60)
    print("\n[READY TO COPY TO RPi4]")
    print(f"   Folder to flash to USB: {PROJECT_ROOT / 'RPI4_SCRIPTS'}")
    print("\nOptional: python 8_test_model_pc.py    (test on PC webcam first)")


if __name__ == "__main__":
    main()
