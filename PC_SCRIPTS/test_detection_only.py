"""
============================================================
 PURE OBJECT-DETECTION TEST  (no parking-zone logic)
============================================================
Use this to quickly check whether the current model can see
your toy cars at all. No polygons, no slot logic, just YOLO
on the webcam feed.

It will use, in priority order:
   1) The fine-tuned NCNN model in RPI4_SCRIPTS/models/yolo11n_ncnn_model
   2) best.pt (your fine-tuned PyTorch weights)
   3) yolov8n.pt  (downloads stock COCO model on first run)

Press 'q' to quit.

Run:
    py -3.13 test_detection_only.py
============================================================
"""

import sys
import time
from pathlib import Path

# ---------- Settings you might tweak ----------
CAMERA_INDEX   = 0       # try 0, 1, 2 if your cam isn't found
IMG_SIZE       = 256     # match what we'll use on RPi4
CONF_THRESHOLD = 0.25    # lower => more (and noisier) detections
SHOW_ALL_CLASSES = False # True = show people / chairs / etc as well
# ----------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NCNN_MODEL = PROJECT_ROOT / "RPI4_SCRIPTS" / "models" / "yolo11n_ncnn_model"
PT_MODEL   = PROJECT_ROOT / "RPI4_SCRIPTS" / "models" / "best.pt"

# COCO ids that map to "vehicle"
COCO_VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def pick_model():
    if NCNN_MODEL.exists():
        print(f"[INFO] Using NCNN model: {NCNN_MODEL}")
        return str(NCNN_MODEL)
    if PT_MODEL.exists():
        print(f"[INFO] Using fine-tuned .pt model: {PT_MODEL}")
        return str(PT_MODEL)
    print("[INFO] No custom model found - falling back to YOLOv8n COCO weights.")
    print("       (will be auto-downloaded on first run)")
    return "yolov8n.pt"


def main():
    print("=" * 60)
    print(" PURE DETECTION TEST")
    print("=" * 60)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("        Run 1_install_dependencies.bat first.")
        sys.exit(1)

    model_path = pick_model()
    model = YOLO(model_path)

    # Figure out class names of this model so we can print them nicely
    try:
        names = model.names if hasattr(model, "names") else {}
    except Exception:
        names = {}
    print(f"[INFO] Model classes: {names}")

    print(f"[INFO] Opening camera index {CAMERA_INDEX} ...")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}.")
        print("        Try changing CAMERA_INDEX at the top of this script.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[INFO] Press 'q' to quit, 's' to save a snapshot,")
    print("       '+'/'-' to raise/lower confidence threshold.\n")

    fps_t0, fps_count, fps_text = time.time(), 0, ""
    conf = CONF_THRESHOLD

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame.")
            break

        results = model(frame, imgsz=IMG_SIZE, conf=conf, verbose=False)[0]
        annotated = frame.copy()

        n_vehicles = 0
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls[0])
                score = float(b.conf[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                # Decide whether to draw this detection
                is_vehicle = (cls_id == 0) or (cls_id in COCO_VEHICLE_IDS)
                if not SHOW_ALL_CLASSES and not is_vehicle:
                    continue

                # Color: green = vehicle, gray = other (when SHOW_ALL_CLASSES)
                color = (0, 200, 0) if is_vehicle else (180, 180, 180)
                if is_vehicle:
                    n_vehicles += 1

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_name = names.get(cls_id, f"id{cls_id}")
                label = f"{label_name} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated,
                              (x1, max(0, y1 - th - 6)),
                              (x1 + tw + 4, y1),
                              color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, cv2.LINE_AA)

        # FPS
        fps_count += 1
        if fps_count >= 10:
            dt = time.time() - fps_t0
            fps_text = f"{fps_count / dt:.1f} FPS"
            fps_t0 = time.time()
            fps_count = 0

        header = (f"vehicles seen: {n_vehicles}   "
                  f"conf: {conf:.2f}   {fps_text}   "
                  f"imgsz: {IMG_SIZE}")
        cv2.putText(annotated, header, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated,
                    "[q] quit   [s] save   [+/-] conf",
                    (15, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Detection Test - q to quit", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[OK] saved {fname}")
        elif key in (ord('+'), ord('=')):
            conf = min(0.95, conf + 0.05)
            print(f"[INFO] conf = {conf:.2f}")
        elif key in (ord('-'), ord('_')):
            conf = max(0.05, conf - 0.05)
            print(f"[INFO] conf = {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
