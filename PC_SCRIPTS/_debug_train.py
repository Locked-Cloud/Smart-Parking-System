"""Minimal debug script - hooks into the training loop to find the crash."""
import multiprocessing
multiprocessing.freeze_support()

import os, sys, faulthandler, traceback
faulthandler.enable()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

print("=== DEBUG TRAIN v2 ===")

import torch
print(f"torch {torch.__version__}, CUDA {torch.cuda.is_available()}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info(0)
    print(f"VRAM: {free//1024**2} MB free / {total//1024**2} MB total")

# Patch the actual _do_train to add step-by-step debug prints
from ultralytics.engine.trainer import BaseTrainer
import types

original_setup_train = BaseTrainer._setup_train

def debug_setup_train(self, world_size):
    print("[DEBUG] _setup_train START")
    try:
        result = original_setup_train(self, world_size)
        print("[DEBUG] _setup_train DONE")
        print(f"[DEBUG] train_loader: {type(self.train_loader)}")
        print(f"[DEBUG] About to iterate first batch...")
        sys.stdout.flush()
        try:
            it = iter(self.train_loader)
            print("[DEBUG] iter() OK")
            sys.stdout.flush()
            b = next(it)
            print(f"[DEBUG] First batch OK, type={type(b)}")
            sys.stdout.flush()
            del b, it
        except Exception as e:
            print(f"[DEBUG] Batch iteration failed: {e}")
            traceback.print_exc()
            sys.stdout.flush()
        return result
    except Exception as e:
        print(f"[DEBUG] _setup_train FAILED: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        raise

BaseTrainer._setup_train = debug_setup_train

from ultralytics import YOLO
print("Starting GPU training with batch=8, amp=True...")
sys.stdout.flush()

try:
    model = YOLO("yolov8n.pt")
    model.train(
        data=os.path.join(os.path.dirname(__file__), "..", "data", "data.yaml"),
        imgsz=256,
        epochs=1,
        batch=8,
        freeze=10,
        lr0=0.001,
        optimizer="AdamW",
        device=0,
        project=os.path.join(os.path.dirname(__file__), "..", "runs", "parking"),
        name="debug_run2",
        exist_ok=True,
        verbose=True,
        cache=False,
        workers=0,
        deterministic=False,
        amp=True,
    )
    print("TRAINING COMPLETED SUCCESSFULLY!")
except Exception as e:
    print(f"\nEXCEPTION: {type(e).__name__}: {e}")
    traceback.print_exc()
sys.stdout.flush()
