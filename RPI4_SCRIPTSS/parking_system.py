"""
============================================================
 PARKING DETECTION SYSTEM  -  Raspberry Pi 4
============================================================
Pipeline:
   Camera capture (thread)
       -> Frame queue (max 2)
       -> NCNN YOLO inference (thread)
       -> Polygon zone occupancy (Shapely)
       -> Temporal smoothing (5-frame state machine)
       -> Annotated display (thread)

Press 'q' in the window to quit.
Press 's' to save the current frame as a debug image.

Run:
    python3 parking_system.py
============================================================
"""

import cv2
import json
import time
import queue
import threading
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
HERE = Path(__file__).resolve().parent

CONFIG = {
    # --- Model ---
    "model_path":        str(HERE / "models" / "yolo11n_ncnn_model"),
    "input_size":        256,       # tuned for RPi4 2GB + toy cars
    "conf_threshold":    0.35,

    # --- Camera ---
    # RPi4 2GB has limited RAM, so we capture at lower resolution
    "camera_src":        0,         # USB cam index (or "/dev/video0")
    "cam_width":         640,
    "cam_height":        480,

    # --- Slots ---
    "slots_json":        str(HERE / "parking_slots.json"),

    # --- Occupancy logic ---
    # Slot is considered occupied if a vehicle bbox overlaps the slot
    # polygon by at least this fraction of the slot's area.
    "overlap_threshold": 0.20,
    # Frames of consistent detection required before changing state.
    # Prevents single-frame flicker.
    "confirm_frames":    5,

    # --- Performance ---
    "skip_rate":         2,         # run inference every Nth frame
    # After fine-tuning class 0 = vehicle. The COCO vehicle ids are
    # also accepted in case you ever swap to a stock model for testing.
    "vehicle_classes":   {0, 2, 3, 5, 7},

    # --- Display ---
    "show_window":       True,
    "print_status_every": 30,       # frames; how often to log to stdout
}


# ============================================================
# Parking slot state machine
# ============================================================
class ParkingSlot:
    def __init__(self, slot_id, points, confirm_frames=5):
        from shapely.geometry import Polygon
        self.id = slot_id
        self.pts = np.array(points, dtype=np.int32)
        self.polygon = Polygon(points)
        self.area = max(self.polygon.area, 1e-6)
        self.state = "free"
        self.occ_counter = 0
        self.free_counter = 0
        self.confirm_frames = confirm_frames

    def update(self, vehicle_boxes, overlap_threshold):
        from shapely.geometry import box as shapely_box
        detected = False
        for x1, y1, x2, y2 in vehicle_boxes:
            vp = shapely_box(x1, y1, x2, y2)
            inter = self.polygon.intersection(vp).area
            if inter / self.area >= overlap_threshold:
                detected = True
                break

        if detected:
            self.occ_counter = min(self.occ_counter + 1, self.confirm_frames + 1)
            self.free_counter = 0
        else:
            self.free_counter = min(self.free_counter + 1, self.confirm_frames + 1)
            self.occ_counter = 0

        if self.occ_counter >= self.confirm_frames:
            self.state = "occupied"
        elif self.free_counter >= self.confirm_frames:
            self.state = "free"
        return self.state


# ============================================================
# Helpers
# ============================================================
def push_latest(q, item):
    """Put item into queue, dropping oldest if full (latest-only buffer)."""
    if q.full():
        try:
            q.get_nowait()
        except queue.Empty:
            pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def load_slots():
    p = Path(CONFIG["slots_json"])
    if not p.exists():
        print(f"[ERROR] {p} not found. Run annotate_slots.py first.")
        return []
    with open(p) as f:
        data = json.load(f)
    return [
        ParkingSlot(s["id"], s["points"], CONFIG["confirm_frames"])
        for s in data.get("slots", [])
    ]


# ============================================================
# Workers
# ============================================================
def capture_worker(frame_q, stop_event):
    cap = cv2.VideoCapture(CONFIG["camera_src"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["cam_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["cam_height"])
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CONFIG['camera_src']}")
        stop_event.set()
        return

    print("[CAPTURE] Started.")
    while not stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        push_latest(frame_q, frame)
    cap.release()
    print("[CAPTURE] Stopped.")


def inference_worker(frame_q, result_q, slots, stop_event):
    from ultralytics import YOLO
    print("[INFERENCE] Loading model ...")
    model = YOLO(CONFIG["model_path"])
    print("[INFERENCE] Model loaded.")

    frame_count = 0
    last_annotated = None
    last_status_print = 0
    fps_t0 = time.time()
    fps_count = 0
    fps_text = ""

    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_count += 1

        # Skip-rate: keep showing last annotated frame for skipped ones
        if frame_count % CONFIG["skip_rate"] != 0:
            if last_annotated is not None:
                push_latest(result_q, last_annotated)
            continue

        # ---- Run detection ----
        results = model(
            frame,
            imgsz=CONFIG["input_size"],
            conf=CONFIG["conf_threshold"],
            verbose=False,
        )[0]

        vehicle_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls[0])
                if cls_id in CONFIG["vehicle_classes"]:
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    vehicle_boxes.append(xyxy)

        # ---- Update slot states ----
        for slot in slots:
            slot.update(vehicle_boxes, CONFIG["overlap_threshold"])

        # ---- FPS ----
        fps_count += 1
        if fps_count >= 10:
            dt = max(time.time() - fps_t0, 1e-6)
            fps_text = f"{fps_count / dt:.1f} FPS"
            fps_t0 = time.time()
            fps_count = 0

        # ---- Annotate ----
        annotated = frame.copy()

        # Draw detected vehicles (subtle blue)
        for x1, y1, x2, y2 in vehicle_boxes:
            cv2.rectangle(annotated, (int(x1), int(y1)),
                          (int(x2), int(y2)), (255, 100, 0), 1)

        # Draw slots
        for slot in slots:
            if slot.state == "occupied":
                color = (0, 0, 255)   # red
                label = "OCC"
            else:
                color = (0, 255, 0)   # green
                label = "FREE"
            cv2.polylines(annotated, [slot.pts], True, color, 2)
            c = slot.pts.mean(axis=0).astype(int)
            cv2.putText(annotated, f"#{slot.id} {label}", tuple(c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        occupied = sum(1 for s in slots if s.state == "occupied")
        free = len(slots) - occupied
        header = f"FREE: {free}/{len(slots)}   OCCUPIED: {occupied}   {fps_text}"
        cv2.putText(annotated, header, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        last_annotated = annotated
        push_latest(result_q, annotated)

        # Periodic console log
        if frame_count - last_status_print >= CONFIG["print_status_every"]:
            states = "".join("O" if s.state == "occupied" else "."
                             for s in slots)
            print(f"[STATUS] frame={frame_count}  {fps_text}  "
                  f"free={free}/{len(slots)}  states=[{states}]")
            last_status_print = frame_count

    print("[INFERENCE] Stopped.")


def display_worker(result_q, stop_event):
    last = None
    print("[DISPLAY] Started. Press 'q' to quit, 's' to snapshot.")
    while not stop_event.is_set():
        try:
            last = result_q.get(timeout=0.05)
        except queue.Empty:
            pass

        if CONFIG["show_window"] and last is not None:
            cv2.imshow("Parking Monitor", last)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break
            elif key == ord('s') and last is not None:
                fname = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, last)
                print(f"[OK] Saved snapshot: {fname}")
    if CONFIG["show_window"]:
        cv2.destroyAllWindows()
    print("[DISPLAY] Stopped.")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print(" RPi4 PARKING DETECTION SYSTEM")
    print("=" * 60)

    if not Path(CONFIG["model_path"]).exists():
        print(f"[ERROR] Model not found at {CONFIG['model_path']}")
        print("        Copy the yolo11n_ncnn_model folder into models/.")
        return

    slots = load_slots()
    if not slots:
        print("[ERROR] No parking slots defined. Run annotate_slots.py first.")
        return
    print(f"[INFO] Loaded {len(slots)} parking slots.")

    frame_q = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    threads = [
        threading.Thread(target=capture_worker,
                         args=(frame_q, stop_event),
                         daemon=True, name="capture"),
        threading.Thread(target=inference_worker,
                         args=(frame_q, result_q, slots, stop_event),
                         daemon=True, name="inference"),
        threading.Thread(target=display_worker,
                         args=(result_q, stop_event),
                         daemon=True, name="display"),
    ]
    for t in threads:
        t.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[CTRL-C] Shutting down ...")
        stop_event.set()

    for t in threads:
        t.join(timeout=3.0)

    # Final summary
    occupied = sum(1 for s in slots if s.state == "occupied")
    free = len(slots) - occupied
    print("\n" + "=" * 60)
    print(f" FINAL: free={free}/{len(slots)}   occupied={occupied}")
    print("=" * 60)


if __name__ == "__main__":
    main()
