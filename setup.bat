@echo off
chcp 65001 >nul
echo ============================================================
echo  Real-time Screen Object Detection System - Setup
echo  Hardware target: Lenovo Y7000 2024 (RTX 4060)
echo ============================================================
echo.

:: Check for Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3.11 from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version 2>&1
echo.

:: Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] PyTorch not installed yet or CUDA not available.
    echo Continuing with pip install...
)

echo.
echo [1/4] Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Some packages failed to install. Check errors above.
)

echo.
echo [2/4] Downloading YOLOv8n model...
if not exist "engines\yolov8n.pt" (
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    if exist "yolov8n.pt" (
        move yolov8n.pt engines\yolov8n.pt
        echo   Downloaded engines\yolov8n.pt
    )
) else (
    echo   engines\yolov8n.pt already exists
)

echo.
echo [3/4] Exporting ONNX model...
if not exist "engines\yolov8n.onnx" (
    python -c "from ultralytics import YOLO; m=YOLO('engines/yolov8n.pt'); m.export(format='onnx', imgsz=640)"
    if exist "yolov8n.onnx" (
        move yolov8n.onnx engines\yolov8n.onnx
        echo   Exported engines\yolov8n.onnx
    )
) else (
    echo   engines\yolov8n.onnx already exists
)

echo.
echo [4/4] Setup complete!
echo.
echo ------------------------------------------------------------
echo  Usage:
echo    python main.py                Start detection
echo    python main.py --no-control   Disable cursor movement
echo    python main.py --conf 0.4     Lower confidence threshold
echo    python main.py --debug        Debug logging
echo    python main.py --region 0 0 1920 1080  Capture region
echo ------------------------------------------------------------
echo.
echo  Press any key to run the system now...
pause >nul
python main.py