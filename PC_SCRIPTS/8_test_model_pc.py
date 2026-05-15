"""
============================================================
 STEP 8: Test the Exported Model on Your PC Webcam
============================================================
Sanity check before flashing to RPi4. Place a few toy cars in
front of the camera and verify they get green boxes.

Press 'q' in the window to quit.

Run:
    python 8_test_model_pc.py
============================================================
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NCNN_MODEL = PROJECT_ROOT / "RPI4_SCRIPTS" / "models" / "yolo11n_ncnn_model"
PT_MODEL = PROJECT_ROOT / "RPI4_SCRIPTS" / "models" / "best.pt"

CAMERA_INDEX = 0
IMG_SIZE = 256
CONF = 0.35


def main():
    print("=" * 60)
    print(" STEP 8: Test Model on PC Webcam")
    print("=" * 60)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if NCNN_MODEL.exists():
        print(f"[INFO] Using NCNN model: {NCNN_MODEL}")
        model = YOLO(str(NCNN_MODEL))
    elif PT_MODEL.exists():
        print(f"[INFO] Using .pt model: {PT_MODEL}")
        model = YOLO(str(PT_MODEL))
    else:
        print("[ERROR] No model found. Run 7_export_model.py first.")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Press 'q' to quit.")
    fps_t0 = time.time()
    fps_count = 0
    fps_text = ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)
        annotated = results[0].plot()

        fps_count += 1
        if fps_count >= 10:
            dt = time.time() - fps_t0
            fps_text = f"{fps_count / dt:.1f} FPS  (PC test)"
            fps_t0 = time.time()
            fps_count = 0
        cv2.putText(annotated, fps_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("PC Test - q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
