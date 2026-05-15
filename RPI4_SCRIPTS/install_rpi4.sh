#!/bin/bash
# ============================================================
#  RPi4 Installer for Parking Detection System
#  Run on the Raspberry Pi 4 ONCE after copying files from USB
# ============================================================
#
#  Usage:
#      chmod +x install_rpi4.sh
#      ./install_rpi4.sh
#
#  REQUIREMENTS:
#    - Raspberry Pi OS 64-bit (Bullseye or Bookworm)
#    - Internet connection
# ============================================================

set -e  # exit on any error

echo "=========================================================="
echo " RPi4 Parking System - Installer"
echo "=========================================================="

# 0. Verify we're on 64-bit OS
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "[WARN] You are running a $ARCH OS."
    echo "       64-bit Raspberry Pi OS (aarch64) is strongly recommended"
    echo "       for best performance."
    read -p "Continue anyway? [y/N] " yn
    if [[ "$yn" != "y" && "$yn" != "Y" ]]; then exit 1; fi
fi

# 1. System packages
echo ""
echo "[STEP 1/4] Installing system packages..."
sudo apt update
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libopencv-dev python3-opencv \
    libatlas-base-dev \
    libjpeg-dev libtiff-dev \
    cpufrequtils \
    git

# 2. Set CPU to performance mode (faster inference, no throttling)
echo ""
echo "[STEP 2/4] Setting CPU governor to 'performance'..."
sudo cpufreq-set -g performance || true
echo "Current governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"

# 3. Python virtual environment
echo ""
echo "[STEP 3/4] Creating Python virtual environment..."
if [ ! -d "parking_env" ]; then
    python3 -m venv parking_env --system-site-packages
fi
source parking_env/bin/activate

python3 -m pip install --upgrade pip wheel setuptools

# 4. Python packages
echo ""
echo "[STEP 4/4] Installing Python packages (this can take 5-15 minutes)..."
pip install numpy
pip install opencv-python
pip install shapely
pip install supervision
pip install ultralytics      # brings in NCNN inference support
pip install ncnn

echo ""
echo "=========================================================="
echo " INSTALLATION COMPLETE!"
echo "=========================================================="
echo ""
echo "Activate the environment in every new shell with:"
echo "    source parking_env/bin/activate"
echo ""
echo "Step A: Define parking slots (one-time setup)"
echo "    python3 annotate_slots.py"
echo ""
echo "Step B: Run the full parking system"
echo "    python3 parking_system.py"
echo ""
