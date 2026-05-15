<p align="center">
  <img src="https://img.shields.io/badge/Platform-Raspberry%20Pi%204-C51A4A?style=for-the-badge&logo=raspberrypi&logoColor=white" alt="Raspberry Pi 4"/>
  <img src="https://img.shields.io/badge/Model-YOLOv8n-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv8"/>
  <img src="https://img.shields.io/badge/Runtime-NCNN-FF6F00?style=for-the-badge" alt="NCNN"/>
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🚗 Smart Parking Detection System</h1>
<h3 align="center">Real-Time Toy-Car Parking Occupancy Detection on Raspberry Pi 4</h3>

<p align="center">
  An end-to-end embedded computer vision pipeline that detects <b>toy cars</b> in a miniature parking lot using a <b>Raspberry Pi 4 (2 GB)</b> with a USB webcam — from data collection and model training on a PC, to real-time inference on edge hardware.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start — Deploy Baseline Model](#-quick-start--deploy-baseline-model)
- [PC Pipeline — Train a Custom Model](#-pc-pipeline--train-a-custom-model)
  - [Step 1: Install Dependencies](#step-1-install-dependencies)
  - [Step 2: Capture Training Data](#step-2-capture-training-data)
  - [Step 3: Auto-Label Images](#step-3-auto-label-images)
  - [Step 4: Review & Correct Labels](#step-4-review--correct-labels)
  - [Step 5: Prepare Dataset](#step-5-prepare-dataset)
  - [Step 6: Fine-Tune Model](#step-6-fine-tune-model)
  - [Step 7: Export to NCNN](#step-7-export-to-ncnn)
  - [Step 8: Test on PC](#step-8-test-on-pc-optional)
- [RPi4 Deployment](#-rpi4-deployment)
  - [Step A: Transfer Files](#step-a-transfer-files-via-usb)
  - [Step B: Install on RPi4](#step-b-install-one-time)
  - [Step C: Define Parking Slots](#step-c-define-parking-slots-one-time)
  - [Step D: Run the System](#step-d-run-the-live-system)
- [Performance Tuning](#-performance-tuning)
- [Troubleshooting](#-troubleshooting)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🔍 Overview

Public object detection datasets (like COCO) are trained on **real-world vehicles** — they don't generalize well to **toy cars**, which differ significantly in shape, scale, texture, and material. This project solves that problem by providing a complete pipeline to:

1. **Collect** your own training images of your specific toy cars in your specific parking layout.
2. **Auto-label** images using a pretrained COCO model, then manually refine.
3. **Fine-tune** a lightweight YOLOv8n model on your custom dataset.
4. **Export** the model to NCNN format optimized for ARM CPUs.
5. **Deploy** a real-time parking occupancy system on a Raspberry Pi 4.

> **💡 A pre-built baseline model is included** — you can deploy to the RPi4 and have a working system in under 30 minutes, then improve accuracy later with custom training.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Plug & Play Baseline** | Pre-exported NCNN model included — deploy immediately without training |
| **Custom Training Pipeline** | 8-step guided workflow to train on your own toy cars for maximum accuracy |
| **Smart Auto-Labeling** | YOLOv8s pre-labels your images, saving 80%+ of manual annotation work |
| **Two-Stage Fine-Tuning** | Frozen backbone → full unfreeze strategy optimized for small datasets |
| **NCNN Inference** | Fastest runtime for ARM CPUs — achieves ~12–20 FPS on RPi4 2 GB |
| **Multi-Threaded Architecture** | Separate capture, inference, and display threads for maximum throughput |
| **Temporal Smoothing** | 5-frame state machine prevents flickering between FREE/OCCUPIED states |
| **Interactive Slot Annotation** | Click-to-define parking slot polygons with a visual GUI tool |
| **GPU Auto-Detection** | Automatically uses NVIDIA CUDA for training if available, falls back to CPU |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PC SIDE (Windows)                            │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐  │
│  │ Capture  │──▶│  Auto    │──▶│ Review & │──▶│  Fine-Tune     │  │
│  │ Images   │   │  Label   │   │ Correct  │   │  YOLOv8n       │  │
│  │ (Step 2) │   │ (Step 3) │   │ (Step 4) │   │  (Step 6)      │  │
│  └──────────┘   └──────────┘   └──────────┘   └───────┬────────┘  │
│                                                        │           │
│                                                        ▼           │
│                                                ┌──────────────┐    │
│                                                │ Export NCNN  │    │
│                                                │ (Step 7)     │    │
│                                                └──────┬───────┘    │
└───────────────────────────────────────────────────────┼────────────┘
                                                        │ USB Flash
┌───────────────────────────────────────────────────────┼────────────┐
│                     RPi4 SIDE (Linux)                  ▼            │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │  Annotate    │    │         parking_system.py                │  │
│  │  Parking     │    │                                          │  │
│  │  Slots       │    │  ┌─────────┐  ┌──────────┐  ┌────────┐ │  │
│  │  (one-time)  │    │  │ Camera  │─▶│  YOLO    │─▶│Display │ │  │
│  └──────────────┘    │  │ Thread  │  │ Inference│  │ Thread │ │  │
│                      │  └─────────┘  │ + Shapely│  └────────┘ │  │
│                      │               │ Overlap  │              │  │
│                      │               └──────────┘              │  │
│                      └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
smart-parking-detection/
│
├── PC_SCRIPTS/                        # Training pipeline (run on PC)
│   ├── 0_build_baseline_model.py      #   Build baseline NCNN model from COCO weights
│   ├── 1_install_dependencies.bat     #   One-click dependency installer
│   ├── 2_capture_training_data.py     #   Webcam image capture tool
│   ├── 3_auto_label.py                #   Auto-label with YOLOv8s (COCO)
│   ├── 4_review_labels.py             #   Launch labelImg for manual review
│   ├── 5_prepare_dataset.py           #   Generate data.yaml for training
│   ├── 6_finetune_model.py            #   Two-stage YOLOv8n fine-tuning
│   ├── 7_export_model.py              #   Export to NCNN + ONNX formats
│   └── 8_test_model_pc.py             #   Sanity test on PC webcam
│
├── RPI4_SCRIPTS/                      # Deployment package (copy to RPi4)
│   ├── install_rpi4.sh                #   One-time RPi4 setup script
│   ├── annotate_slots.py              #   Interactive parking slot definer
│   ├── parking_system.py              #   Main real-time detection system
│   ├── test_detection_only.py         #   Detection-only test (no slots)
│   └── models/
│       └── yolo11n_ncnn_model/        #   Pre-built NCNN model (ready to use)
│
├── data/
│   ├── raw_images/                    #   Captured training photos
│   ├── toy_dataset/                   #   Train/valid splits with labels
│   │   ├── train/images/ & labels/
│   │   └── valid/images/ & labels/
│   └── data.yaml                      #   Dataset configuration for YOLO
│
├── runs/                              #   Training outputs & metrics
│   └── parking/
│       ├── stage1_frozen/
│       └── stage2_full/
│
├── OPTION_B_GUIDE.txt                 #   Detailed step-by-step training guide
└── README.md                          #   This file
```

---

## 📦 Prerequisites

### PC (Training & Export)

| Requirement | Details |
|---|---|
| **OS** | Windows 10/11 |
| **Python** | 3.13 (PyTorch does not support 3.14 yet) |
| **Disk Space** | ~10 GB free |
| **GPU** *(optional)* | NVIDIA GPU with CUDA support (training: ~30 min vs ~3 hrs on CPU) |
| **Webcam** | USB webcam (ideally the same one used on the RPi4) |

### Raspberry Pi 4

| Requirement | Details |
|---|---|
| **Board** | Raspberry Pi 4 (2 GB RAM or higher) |
| **OS** | Raspberry Pi OS **64-bit** (Bullseye or Bookworm) — verify with `uname -m` → `aarch64` |
| **Camera** | USB webcam |
| **Internet** | Required only for initial setup |

---

## 🚀 Quick Start — Deploy Baseline Model

A pre-built NCNN model (YOLOv8n with COCO weights, 256×256 input) is already included. You can have a working parking system **today** without any training:

```
1. Copy RPI4_SCRIPTS/ folder to a USB drive
2. Plug USB into RPi4 and copy files
3. Run install_rpi4.sh (one-time)
4. Define parking slots with annotate_slots.py
5. Launch parking_system.py
```

> The baseline model detects toy cars using COCO's `car`/`truck` classes. For higher accuracy on your specific toy cars, follow the [custom training pipeline](#-pc-pipeline--train-a-custom-model) below.

---

## 💻 PC Pipeline — Train a Custom Model

> **Time estimate:** ~1.5–2.5 hours total (with GPU)

### Step 1: Install Dependencies

```cmd
cd PC_SCRIPTS
1_install_dependencies.bat
```

Installs: `ultralytics`, `opencv-python`, `shapely`, `supervision`, `labelImg`, `ncnn`, and CUDA-enabled PyTorch (if NVIDIA GPU detected).

---

### Step 2: Capture Training Data

```cmd
py -3.13 2_capture_training_data.py
```

Opens a webcam window for capturing training images of your toy cars.

| Key | Action |
|---|---|
| `SPACE` | Capture one photo |
| `A` | Toggle auto-capture (1 photo / 1.5s) |
| `D` | Delete last captured photo |
| `Q` | Quit |

**Target: 200–400 photos** covering:
- ✅ Every toy car you own (all colors/models)
- ✅ Cars inside, outside, and between parking spots
- ✅ Empty, full, and partially-full lot
- ✅ Different lighting conditions
- ✅ Same camera angle/height as final deployment

> **Tip:** Mount the webcam on a tripod. Use auto-mode and slowly rearrange cars for ~3 minutes to effortlessly capture 120+ diverse images.

---

### Step 3: Auto-Label Images

```cmd
py -3.13 3_auto_label.py
```

- Runs **YOLOv8s** (COCO) on every captured image at low confidence (0.20)
- Maps detected vehicles (`car`, `truck`, `bus`, `motorcycle`) → class `0` (`vehicle`)
- Splits images into `train/` (85%) and `valid/` (15%)
- Generates YOLO-format `.txt` label files

> Auto-labels are ~80% accurate — the next step lets you fix the remaining errors.

---

### Step 4: Review & Correct Labels

```cmd
py -3.13 4_review_labels.py
```

Opens **labelImg** for manual review and correction.

| Key | Action |
|---|---|
| `W` | Draw new bounding box |
| `D` | Next image |
| `A` | Previous image |
| `Del` | Delete selected box |
| `Ctrl+S` | Save |

> **Important:** Ensure the format dropdown shows **YOLO** (not Pascal VOC). Enable **View → Auto Save Mode**.

---

### Step 5: Prepare Dataset

```cmd
py -3.13 5_prepare_dataset.py
```

Generates `data/data.yaml` with absolute paths for YOLO training.

---

### Step 6: Fine-Tune Model

```cmd
py -3.13 6_finetune_model.py
```

Two-stage training optimized for small toy-car datasets:

| Stage | Epochs | Strategy | Purpose |
|---|---|---|---|
| **Stage 1** | 20 | Freeze backbone (10 layers) | Train detection head on your data |
| **Stage 2** | 80 | Full unfreeze + strong augmentation | Adapt entire network to toy cars |

**Augmentations applied:** color jitter, rotation, mosaic, mixup, flipping, scale, shear, and perspective transforms.

| Hardware | Estimated Time |
|---|---|
| NVIDIA GPU (e.g., GTX 1650 Ti) | ~25–45 minutes |
| CPU only | ~2–4 hours |

**Output:** `runs/parking/stage2_full/weights/best.pt`

---

### Step 7: Export to NCNN

```cmd
py -3.13 7_export_model.py
```

- Exports `best.pt` → **NCNN** (primary, fastest on ARM) + **ONNX** (fallback)
- Auto-copies the NCNN model to `RPI4_SCRIPTS/models/yolo11n_ncnn_model/`
- After this step, `RPI4_SCRIPTS/` is ready to flash to USB

---

### Step 8: Test on PC *(Optional)*

```cmd
py -3.13 8_test_model_pc.py
```

Quick sanity check — place toy cars in front of the webcam and verify they get detected with green bounding boxes. Press `Q` to quit.

---

## 🍓 RPi4 Deployment

### Step A: Transfer Files via USB

```bash
# On PC: copy RPI4_SCRIPTS/ to USB drive
# On RPi4:
cp -r /media/$USER/<USB_NAME>/RPI4_SCRIPTS ~/parking
cd ~/parking
```

### Step B: Install (One-Time)

```bash
chmod +x install_rpi4.sh
./install_rpi4.sh
```

Installs OpenCV, Ultralytics, NCNN, Shapely, Supervision, and sets the CPU governor to `performance` mode. Takes ~5–15 minutes.

### Step C: Define Parking Slots (One-Time)

```bash
source parking_env/bin/activate
python3 annotate_slots.py
```

| Key | Action |
|---|---|
| **Left-click** | Add a corner point |
| `N` | Save current polygon as a slot |
| `U` | Undo last point |
| `R` | Reset current polygon |
| `D` | Delete last saved slot |
| `S` | Save all slots to `parking_slots.json` |
| `L` | Reload reference frame from camera |
| `Q` | Quit (auto-saves) |

> Click the 4 corners of each parking spot → press `N` → repeat → press `S`.

### Step D: Run the Live System

```bash
source parking_env/bin/activate
python3 parking_system.py
```

**On-Screen Display:**

| Element | Meaning |
|---|---|
| 🟩 Green polygon | Slot is **FREE** |
| 🟥 Red polygon | Slot is **OCCUPIED** |
| 🟦 Blue thin boxes | Detected toy cars |
| Header bar | Live count (free/occupied) + FPS |

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Save annotated snapshot |

---

## ⚡ Performance Tuning

**Expected:** ~12–20 FPS on RPi4 2 GB at 256×256 input with `skip_rate=2`

All tuning parameters are at the top of `parking_system.py`:

| Parameter | Default | Effect |
|---|---|---|
| `input_size` | `256` | ↓ 192 = faster, ↑ 320 = more accurate |
| `conf_threshold` | `0.35` | ↓ = more detections + more false positives |
| `overlap_threshold` | `0.20` | Fraction of slot area that must be covered |
| `confirm_frames` | `5` | Higher = smoother state, slower reaction |
| `skip_rate` | `2` | Higher = better display FPS, slower detection |

**If FPS is too low:**

1. Lower `input_size` to `192`
2. Increase `skip_rate` to `3`
3. Verify 64-bit OS: `uname -m` → `aarch64`
4. Confirm performance governor: `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`
5. Check thermal throttling: `vcgencmd get_throttled` → `throttled=0x0`
6. Add a heatsink/fan — sustained inference generates significant heat

**RAM tips (2 GB model):**
- Close all other applications (no browser, no extra services)
- Camera resolution is capped at 640×480 to minimize memory usage

---

## 🛠 Troubleshooting

| Problem | Solution |
|---|---|
| `Cannot open camera 0` | Try changing `camera_src` to `1`, `2`, or `"/dev/video0"` |
| Model not found | Ensure `models/yolo11n_ncnn_model/` contains both `.bin` and `.param` files |
| Very low FPS on RPi4 | Confirm 64-bit OS + `performance` governor + heatsink attached |
| Many false positives | Raise `conf_threshold` to `0.45` |
| State flickers FREE ↔ OCC | Increase `confirm_frames` to `8` or `10` |
| Detector misses toy cars | Capture more diverse photos (Step 2) and retrain |
| Out-of-memory crash | Lower `input_size` to `192`, reduce `cam_width`/`cam_height` |
| `py -3.13` not recognized | Ensure Python 3.13 is installed and added to PATH |

---

## 📈 Improving Accuracy

If detection accuracy is unsatisfactory after the first training round:

1. **Add more data** — Run `2_capture_training_data.py` again with different lighting, angles, and car arrangements
2. **Re-run the pipeline** — Steps 3 → 7 (new images are added automatically)
3. **Increase training epochs** — Set `STAGE2_EPOCHS = 120` in `6_finetune_model.py`
4. **Review labels carefully** — Missed or incorrect labels in Step 4 are the #1 cause of poor accuracy

---

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| **Object Detection** | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (nano variant) |
| **Edge Inference** | [NCNN](https://github.com/Tencent/ncnn) — optimized for ARM CPUs |
| **Computer Vision** | [OpenCV](https://opencv.org/) — capture, display, annotation |
| **Geometric Overlap** | [Shapely](https://shapely.readthedocs.io/) — polygon intersection for slot occupancy |
| **Annotation Tool** | [labelImg](https://github.com/HumanSignal/labelImg) — YOLO-format bounding box editor |
| **Training Framework** | [PyTorch](https://pytorch.org/) with CUDA support (auto-detected) |
| **Visualization** | [Supervision](https://github.com/roboflow/supervision) — detection visualization utilities |
| **Target Hardware** | Raspberry Pi 4 (2 GB) + USB Webcam |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built with ❤️ for Embedded Systems</b><br>
  <sub>If this project helped you, consider giving it a ⭐</sub>
</p>
# Smart-Parking-System
