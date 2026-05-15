# 🚗 Toy Car Parking Detection Dataset

A custom-built object detection dataset of **toy cars** in a miniature parking lot, designed for training lightweight YOLO models deployable on **edge devices** like the Raspberry Pi 4.

---

## 📊 Dataset Statistics

| Metric | Value |
|---|---|
| **Total Images** | 500 |
| **Training Images** | 425 |
| **Validation Images** | 75 |
| **Classes** | 1 (`vehicle`) |
| **Annotation Format** | YOLO (`.txt` — `class_id cx cy w h`, normalized) |
| **Image Format** | JPEG |
| **Train/Val Split** | 85% / 15% |

---

## 🎯 Why This Dataset?

Public datasets like COCO are trained on **real-world vehicles** — they don't generalize well to **toy cars**, which differ significantly in:

- **Scale** — toy cars are 5–10 cm long vs. real cars at 4+ meters
- **Shape** — simplified, rounded body shapes
- **Texture** — glossy plastic surfaces, uniform colors
- **Context** — miniature parking lots with different backgrounds

This dataset bridges that gap by providing real annotated images of toy cars captured from a **fixed overhead camera** — the exact setup used in an embedded parking detection system.

---

## 📁 Dataset Structure

```
toy-car-parking-detection/
├── data.yaml              # YOLOv8 dataset config (relative paths)
├── train/
│   ├── images/            # 425 training images (.jpg)
│   └── labels/            # 425 YOLO annotation files (.txt)
└── valid/
    ├── images/            # 75 validation images (.jpg)
    └── labels/            # 75 YOLO annotation files (.txt)
```

---

## 📝 Annotation Format

Labels follow the standard **YOLO format** — one `.txt` file per image with the same filename:

```
<class_id> <center_x> <center_y> <width> <height>
```

All values are **normalized** (0.0 – 1.0) relative to image dimensions.

**Example** (`img_001.txt`):
```
0 0.629022 0.256091 0.126981 0.222987
0 0.374464 0.393817 0.149567 0.290018
0 0.174946 0.537069 0.139493 0.320633
```

Each line represents one toy car bounding box. Class `0` = `vehicle`.

---

## 🚀 Quick Start (Kaggle Notebook)

### Train YOLOv8n on this dataset:

```python
!pip install ultralytics

from ultralytics import YOLO

# Load a pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train on the toy car dataset
model.train(
    data="/kaggle/input/toy-car-parking-detection/data.yaml",
    imgsz=256,
    epochs=100,
    batch=16,
    name="toy_car_detector"
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
```

### Export for Raspberry Pi (NCNN):

```python
# Export the best model to NCNN format for ARM CPUs
model = YOLO("runs/detect/toy_car_detector/weights/best.pt")
model.export(format="ncnn", imgsz=256)
```

---

## 📸 Data Collection Details

| Parameter | Value |
|---|---|
| **Camera** | USB webcam (640×480 capture resolution) |
| **Mounting** | Fixed overhead position (tripod) |
| **Subjects** | Various toy car models (different colors, shapes, sizes) |
| **Scenarios** | Empty lot, full lot, partial occupancy, varied lighting |
| **Labeling** | Auto-pre-labeled with YOLOv8s (COCO), then manually reviewed & corrected |

---

## 🔗 Related Project

This dataset is part of the **Smart Parking Detection System** — a complete end-to-end pipeline for real-time parking occupancy detection on Raspberry Pi 4 using NCNN inference.

**GitHub:** [Smart Parking Detection System](https://github.com/INSERT_YOUR_USERNAME/smart-parking-detection)

---

## 📄 License

This dataset is released under the **CC BY 4.0** license. You are free to share and adapt the material for any purpose, as long as appropriate credit is given.

---

## 📌 Citation

If you use this dataset in your research or project, please cite:

```
@misc{toy_car_parking_detection_2026,
  title   = {Toy Car Parking Detection Dataset},
  author  = {Ibrahim Ahmed},
  year    = {2026},
  url     = {https://www.kaggle.com/datasets/INSERT_YOUR_KAGGLE_USERNAME/toy-car-parking-detection}
}
```
