@echo off
REM ============================================================
REM  PC Dependency Installer for Toy-Car Parking Detection
REM  Run this ONCE on your PC.
REM
REM  Uses Python 3.13 (Ultralytics / PyTorch don't support 3.14 yet)
REM ============================================================

set PY=py -3.13

echo.
echo ==========================================================
echo  Installing Python packages (Python 3.13)
echo ==========================================================
echo.

%PY% -m pip install --upgrade pip

REM --- Core ML / detection ---
%PY% -m pip install ultralytics
%PY% -m pip install opencv-python
%PY% -m pip install numpy
%PY% -m pip install pillow
%PY% -m pip install pyyaml
%PY% -m pip install tqdm
%PY% -m pip install matplotlib

REM --- Geometry / parking zone logic ---
%PY% -m pip install shapely
%PY% -m pip install supervision

REM --- Annotation tool ---
%PY% -m pip install labelImg

REM --- Export helpers ---
%PY% -m pip install onnx
%PY% -m pip install onnxruntime
%PY% -m pip install ncnn

REM --- CUDA-enabled PyTorch (for NVIDIA GPUs)
echo.
echo ==========================================================
echo  Installing CUDA-enabled PyTorch (skip if no NVIDIA GPU)
echo ==========================================================
%PY% -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo ==========================================================
echo  All dependencies installed!
echo ==========================================================
echo.
echo Next step: python 2_capture_training_data.py
echo.
pause
