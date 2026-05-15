"""
============================================================
 PURE OBJECT-DETECTION TEST  (RPi4 version)
============================================================
Tests the NCNN model on the RPi4 with the USB webcam.
NO parking-zone logic. NO threading. Just YOLO -> screen.
Ideal for measuring raw FPS / accuracy on the Pi.

Press 'q' in the window to quit.
'+' / '-' adjust confidence threshold live.
's'       saves a snapshot.
'i'       cycles input size  (192 -> 256 -> 320 -> 416 -> 192)

Run on RPi4:
    source parking_env/bin/activate
    python3 test_detection_only.py
============================================================
"""

import sys
import time
from pathlib import Path

# ---------- Settings (also adjustable live) ----------
CAMERA_INDEX     = 0       # try 0, 1, 2 if cam not found ("/dev/video0" also works)
CAM_WIDTH        = 640
CAM_HEIGHT       = 480
START_IMG_SIZE   = 256     # matches the deployed parking_system.py
START_CONF       = 0.25
SHOW_ALL_CLASSES = False   # True = also draw non-vehicle classes (debug)
PRINT_EVERY_N    = 30      # console FPS log frequency
# ----------------------------------------------------

HERE = Path(__file__).resolve().parent
NCNN_MODEL = HERE / "models" / "yolo11n_ncnn_model"
PT_MODEL   = HERE / "models" / "best.pt"

# COCO ids that mean "vehicle" - kept for the baseline / fallback model
COCO_VEHICLE_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck


def pick_model_path():
    if NCNN_MODEL.exists():
        print(f"[INFO] Using NCNN model: {NCNN_MODEL}")
        return str(NCNN_MODEL)
    if PT_MODEL.exists():
        print(f"[INFO] NCNN missing - using .pt model: {PT_MODEL}")
        return str(PT_MODEL)
    print("[ERROR] No model found.")
    print(f"        Expected: {NCNN_MODEL}")
    print("        Make sure you copied the models/ folder from your PC.")
    sys.exit(1)


def open_camera(idx):
    import cv2
    print(f"[INFO] Opening camera {idx} ...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        # On Linux you can also pass a /dev path
        cap = cv2.VideoCapture(f"/dev/video{idx}" if isinstance(idx, int) else idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {idx}.")
        print("        Try changing CAMERA_INDEX (0, 1, 2, /dev/video0 ...)")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    return cap


def main():
    print("=" * 60)
    print(" RPi4 DETECTION TEST  (no parking logic)")
    print("=" * 60)

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        print("        Run ./install_rpi4.sh first.")
        sys.exit(1)

    model_path = pick_model_path()

    print("[INFO] Loading model ... (first load takes a few seconds)")
    t0 = time.time()
    model = YOLO(model_path)
    print(f"[OK] Model loaded in {time.time() - t0:.1f}s")
    try:
        names = model.names if hasattr(model, "names") else {}
    except Exception:
        names = {}
    print(f"[INFO] Model classes: {names}")

    cap = open_camera(CAMERA_INDEX)

    # Live-adjustable state
    img_size_choices = [192, 256, 320, 416]
    img_size_idx = img_size_choices.index(START_IMG_SIZE) \
        if START_IMG_SIZE in img_size_choices else 1
    conf = START_CONF

    print("\n[CONTROLS]")
    print("   q   = quit")
    print("   s   = save snapshot")
    print("   +/- = increase / decrease confidence threshold")
    print("   i   = cycle input size (192 / 256 / 320 / 416)")
    print()

    fps_t0 = time.time()
    fps_count = 0
    fps_text = "..."
    inf_ms_avg = 0.0
    frame_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame.")
            time.sleep(0.05)
            continue

        img_size = img_size_choices[img_size_idx]

        t_inf = time.time()
        results = model(frame, imgsz=img_size, conf=conf, verbose=False)[0]
        dt_inf = (time.time() - t_inf) * 1000.0   # ms
        # Exponential moving average for the inference time
        inf_ms_avg = dt_inf if inf_ms_avg == 0 else (0.9 * inf_ms_avg + 0.1 * dt_inf)

        annotated = frame.copy()

        n_vehicles = 0
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls[0])
                score = float(b.conf[0])
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]

                is_vehicle = (cls_id == 0) or (cls_id in COCO_VEHICLE_IDS)
                if not SHOW_ALL_CLASSES and not is_vehicle:
                    continue

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

        fps_count += 1
        frame_total += 1
        if fps_count >= 10:
            dt = max(time.time() - fps_t0, 1e-6)
            fps_text = f"{fps_count / dt:.1f} FPS"
            fps_t0 = time.time()
            fps_count = 0

        header = (f"vehicles: {n_vehicles}  conf: {conf:.2f}  "
                  f"imgsz: {img_size}  {fps_text}  "
                  f"inf: {inf_ms_avg:.0f} ms")
        cv2.putText(annotated, header, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(annotated,
                    "[q]quit [s]save [+/-]conf [i]imgsz",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("RPi4 Detection Test - q to quit", annotated)
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
        elif key == ord('i'):
            img_size_idx = (img_size_idx + 1) % len(img_size_choices)
            print(f"[INFO] imgsz = {img_size_choices[img_size_idx]}")

        # Periodic console log so you can watch FPS in a headless ssh session too
        if frame_total % PRINT_EVERY_N == 0:
            print(f"[STATUS] {fps_text}   inf {inf_ms_avg:.0f} ms   "
                  f"imgsz {img_size}   conf {conf:.2f}   "
                  f"vehicles {n_vehicles}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print(" Test finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
