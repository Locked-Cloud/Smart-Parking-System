"""
============================================================
 STEP 6: Fine-Tune YOLOv11n on Your Toy-Car Dataset
============================================================
Run:
    python 6_finetune_model.py
============================================================
"""

import os
import sys
import multiprocessing
from pathlib import Path

# ── FIX: Force matplotlib to use non-interactive backend ──
# Without this, ultralytics' plot_training_labels() triggers the Qt
# backend on Windows, which causes a fatal "access violation" crash.
import matplotlib
matplotlib.use("Agg")

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # makes CUDA errors visible

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML    = PROJECT_ROOT / "data" / "data.yaml"
RUNS_DIR     = PROJECT_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────
IMG_SIZE      = 256
BATCH_SIZE    = 16          # AMP enabled → FP16 training fits 4 GB VRAM
STAGE1_EPOCHS = 20
STAGE2_EPOCHS = 80
PATIENCE      = 25
BASE_MODEL = "yolov8n.pt"   # was "yolo11n.pt"

def get_device():
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"\n[INFO] PyTorch        : {torch.__version__}")
    print(f"[INFO] CUDA available : {has_cuda}")
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory // 1024**2
        free_mb  = torch.cuda.mem_get_info(0)[0] // 1024**2
        print(f"[INFO] GPU            : {gpu_name}")
        print(f"[INFO] VRAM           : {free_mb} MB free / {total_mb} MB total")
    print(f"[INFO] Image size     : {IMG_SIZE}")
    print(f"[INFO] Batch size     : {BATCH_SIZE}")
    return 0 if has_cuda else "cpu"


def train_stage1(device):
    from ultralytics import YOLO
    print("\n" + "=" * 60)
    print(" STAGE 1 / 2 : Freeze backbone, train head")
    print("=" * 60)

    model = YOLO(BASE_MODEL)
    model.train(
        data          = str(DATA_YAML),
        imgsz         = IMG_SIZE,
        epochs        = STAGE1_EPOCHS,
        batch         = BATCH_SIZE,
        freeze        = 10,
        lr0           = 0.001,
        optimizer     = "AdamW",
        device        = device,
        project       = str(RUNS_DIR / "parking"),
        name          = "stage1_frozen",
        exist_ok      = True,
        verbose       = True,
        patience      = PATIENCE,
        cache         = False,        # FIX: disable cache entirely — most reliable on Windows
        workers       = 0,            # FIX: must be 0 on Windows
        deterministic = False,        # FIX: True causes silent hang on some Windows setups
        # Augmentations
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.05,
        amp=True,            # FIX: enable mixed precision — halves VRAM usage
    )


def train_stage2(device):
    from ultralytics import YOLO
    print("\n" + "=" * 60)
    print(" STAGE 2 / 2 : Full fine-tune (unfreeze all)")
    print("=" * 60)

    stage1_weights = RUNS_DIR / "parking" / "stage1_frozen" / "weights" / "last.pt"
    if not stage1_weights.exists():
        print(f"[ERROR] Stage 1 weights not found:\n  {stage1_weights}")
        sys.exit(1)

    model = YOLO(str(stage1_weights))
    model.train(
        data          = str(DATA_YAML),
        imgsz         = IMG_SIZE,
        epochs        = STAGE2_EPOCHS,
        batch         = BATCH_SIZE,
        freeze        = 0,
        lr0           = 0.0005,
        lrf           = 0.01,
        cos_lr        = True,
        optimizer     = "AdamW",
        weight_decay  = 0.0005,
        device        = device,
        project       = str(RUNS_DIR / "parking"),
        name          = "stage2_full",
        exist_ok      = True,
        verbose       = True,
        patience      = PATIENCE,
        cache         = False,        # FIX: same
        workers       = 0,            # FIX: same
        deterministic = False,        # FIX: same
        # Strong augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        shear=3.0,
        perspective=0.0005,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        close_mosaic=15,
        amp=True,            # FIX: enable mixed precision — halves VRAM usage
    )


def validate(device):
    from ultralytics import YOLO
    print("\n" + "=" * 60)
    print(" Validating final model")
    print("=" * 60)

    best = RUNS_DIR / "parking" / "stage2_full" / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt not found at:\n  {best}")
        sys.exit(1)

    metrics = YOLO(str(best)).val(
        data    = str(DATA_YAML),
        imgsz   = IMG_SIZE,
        device  = device,
        workers = 0,
        verbose = True,
    )
    try:
        print(f"\n[RESULT] mAP50    = {metrics.box.map50:.4f}")
        print(f"[RESULT] mAP50-95 = {metrics.box.map:.4f}")
    except Exception:
        pass

    best_abs = (RUNS_DIR / "parking" / "stage2_full" / "weights" / "best.pt").resolve()
    print("\n" + "=" * 60)
    print(" Fine-tuning finished!")
    print(f" Best weights: {best_abs}")
    print("=" * 60)
    print("\nNext step:  python 7_export_model.py")


def main():
    print("=" * 60)
    print(" STEP 6: Fine-Tune YOLOv11n on Toy Car Dataset")
    print("=" * 60)

    if not DATA_YAML.exists():
        print(f"[ERROR] {DATA_YAML} not found. Run 5_prepare_dataset.py first.")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        print("[ERROR] torch not installed.")
        sys.exit(1)

    device = get_device()
    train_stage1(device)
    train_stage2(device)
    validate(device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()