"""
============================================================
 STEP 2: Capture Training Images of Your Toy Cars
============================================================
Since you're using TOY CARS (not real cars), the best dataset
is one you capture yourself in your actual parking setup.

This script captures images from your webcam:
  - Auto mode: takes one photo every N seconds.
  - Manual mode: press SPACE to capture, Q to quit.

GOAL: capture 200-400 images covering:
   * different toy cars  (color, model, scale)
   * different positions in / out of parking spots
   * different lighting (room light on/off, daytime, evening)
   * different angles  (re-aim camera occasionally)
   * empty parking and full parking (and partial)

These images will then be auto-pre-labeled by a strong model
in step 3, so you only need to CORRECT labels rather than
draw them from scratch.

Run:
    python 2_capture_training_data.py
============================================================
"""

import cv2
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_images"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
AUTO_INTERVAL_SEC = 1.5      # auto-capture every N seconds when in auto mode
TARGET_IMAGES = 300          # suggested target


def main():
    print("=" * 60)
    print(" STEP 2: Capture Training Images")
    print("=" * 60)

    print(f"\n[INFO] Saving images to: {RAW_DIR}")
    existing = list(RAW_DIR.glob("*.jpg"))
    print(f"[INFO] Existing images: {len(existing)}")
    print(f"[INFO] Target          : ~{TARGET_IMAGES} images")

    print("\n[CONTROLS]")
    print("   SPACE  = capture single image")
    print("   A      = toggle AUTO mode (1 image / 1.5 sec)")
    print("   D      = delete last captured image")
    print("   Q      = quit\n")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    auto_mode = False
    last_auto_t = 0
    last_saved_path = None
    counter_start = len(existing)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Build display
        display = frame.copy()
        n_now = len(list(RAW_DIR.glob("*.jpg")))
        progress = min(100, int(100 * n_now / TARGET_IMAGES))

        cv2.putText(display, f"Captured: {n_now} / {TARGET_IMAGES}  ({progress}%)",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255), 2)
        cv2.putText(display,
                    f"Mode: {'AUTO' if auto_mode else 'MANUAL'}   "
                    f"[SPACE]=capture  [A]=auto  [D]=del last  [Q]=quit",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)

        if auto_mode:
            cv2.circle(display, (display.shape[1] - 30, 30), 12, (0, 0, 255), -1)

        cv2.imshow("Toy Car Capture", display)
        key = cv2.waitKey(1) & 0xFF

        # Auto-capture
        if auto_mode and (time.time() - last_auto_t) >= AUTO_INTERVAL_SEC:
            ts = int(time.time() * 1000)
            path = RAW_DIR / f"img_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            last_saved_path = path
            last_auto_t = time.time()
            print(f"[AUTO] Saved {path.name}")

        if key == ord(' '):
            ts = int(time.time() * 1000)
            path = RAW_DIR / f"img_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            last_saved_path = path
            print(f"[OK] Saved {path.name}")

        elif key == ord('a'):
            auto_mode = not auto_mode
            last_auto_t = time.time()
            print(f"[INFO] AUTO mode: {auto_mode}")

        elif key == ord('d'):
            if last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                print(f"[OK] Deleted {last_saved_path.name}")
                last_saved_path = None

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    final_count = len(list(RAW_DIR.glob("*.jpg")))
    print("\n" + "=" * 60)
    print(f" Captured {final_count - counter_start} new images this session.")
    print(f" Total in folder: {final_count}")
    print("=" * 60)
    if final_count < 100:
        print("\n[WARN] Less than 100 images. You should aim for 200-400")
        print("       for a robust toy-car detector.")
    print("\nNext step:  python 3_auto_label.py")


if __name__ == "__main__":
    main()
