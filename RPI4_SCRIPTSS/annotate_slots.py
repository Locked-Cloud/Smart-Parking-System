"""
============================================================
 PARKING SLOT ANNOTATOR  (run once on the RPi4)
============================================================
Click the four (or more) corners of each parking space, then
press 'N' to save that slot. Repeat for every spot, then press
'S' to write the JSON file. Press 'Q' to quit.

Controls:
    Left click  - add a corner point
    N           - save current polygon as a new slot
    U           - undo last point
    R           - reset current polygon
    D           - delete LAST saved slot
    S           - save all slots to parking_slots.json
    L           - reload reference frame from camera
    Q           - quit (auto-saves first)
============================================================
"""

import cv2
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_JSON = HERE / "parking_slots.json"
REFERENCE_IMG = HERE / "parking_reference.jpg"

CAMERA_INDEX = 0   # change if your camera is on a different index
WINDOW = "Parking Slot Annotator"


def grab_reference_frame():
    """Capture one frame from the webcam to use as the annotation canvas."""
    print("[INFO] Capturing reference frame from camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        return None
    # Match parking_system.py resolution (RPi4 2GB friendly)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Throw away first few frames (camera warm-up)
    for _ in range(10):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("[ERROR] Failed to capture frame.")
        return None
    cv2.imwrite(str(REFERENCE_IMG), frame)
    print(f"[OK] Reference saved to {REFERENCE_IMG}")
    return frame


def load_existing():
    if OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON) as f:
                data = json.load(f)
            return data.get("slots", [])
        except Exception:
            return []
    return []


def save_slots(slots):
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"slots": slots}, f, indent=2)
    print(f"[OK] Saved {len(slots)} slots to {OUTPUT_JSON}")


def main():
    print("=" * 60)
    print(" PARKING SLOT ANNOTATOR")
    print("=" * 60)

    # Get a reference frame (from camera, or reuse saved one)
    if REFERENCE_IMG.exists():
        print(f"[INFO] Reusing existing reference: {REFERENCE_IMG}")
        base = cv2.imread(str(REFERENCE_IMG))
    else:
        base = grab_reference_frame()
    if base is None:
        return

    slots = load_existing()
    next_id = max([s["id"] for s in slots], default=0) + 1
    current_points = []

    def redraw():
        canvas = base.copy()
        # Saved slots
        for slot in slots:
            pts = np.array(slot["points"], dtype=np.int32)
            cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
            c = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, f"#{slot['id']}", tuple(c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # In-progress polygon
        for p in current_points:
            cv2.circle(canvas, tuple(p), 5, (0, 0, 255), -1)
        if len(current_points) >= 2:
            for i in range(len(current_points) - 1):
                cv2.line(canvas, tuple(current_points[i]),
                         tuple(current_points[i + 1]), (0, 0, 255), 2)
        # Help text
        help_lines = [
            "Click corners | N=save slot | U=undo | R=reset",
            "D=delete last | S=save JSON | L=new ref frame | Q=quit"
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(canvas, line, (10, 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, f"Saved slots: {len(slots)}",
                    (10, canvas.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(WINDOW, canvas)

    def on_mouse(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append([x, y])
            redraw()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord('u'):
            if current_points:
                current_points.pop()
                redraw()

        elif key == ord('r'):
            current_points = []
            redraw()

        elif key == ord('n'):
            if len(current_points) >= 3:
                slots.append({"id": next_id, "points": current_points.copy()})
                print(f"[OK] Saved slot #{next_id} "
                      f"({len(current_points)} corners). Total: {len(slots)}")
                next_id += 1
                current_points = []
                redraw()
            else:
                print("[WARN] Need at least 3 corners.")

        elif key == ord('d'):
            if slots:
                removed = slots.pop()
                print(f"[OK] Removed slot #{removed['id']}. Total: {len(slots)}")
                redraw()

        elif key == ord('s'):
            save_slots(slots)

        elif key == ord('l'):
            new_base = grab_reference_frame()
            if new_base is not None:
                base = new_base
                redraw()

        elif key == ord('q'):
            if slots:
                save_slots(slots)
            break

    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
