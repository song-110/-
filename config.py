"""
Configuration constants for the real-time screen detection system.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import os


@dataclass
class CaptureConfig:
    """Screen capture configuration."""
    # Target region: None = full primary monitor, (x, y, w, h) for window region
    region: Optional[Tuple[int, int, int, int]] = None
    # Resize dimensions for model input
    target_width: int = 640
    target_height: int = 640
    # Capture FPS cap (0 = unlimited)
    max_fps: int = 0
    # DXGI output adapter index for dxcam
    output_idx: int = 0


@dataclass
class PerceptionConfig:
    """YOLO perception configuration."""
    # Model paths
    model_pt: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "engines", "yolov8n.pt"
    ))
    model_onnx: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "engines", "yolov8n.onnx"
    ))
    model_trt: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "engines", "yolov8n.trt"
    ))
    # Inference backend priority: "tensorrt", "onnx", "torch"
    backend: str = ""  # auto-detect if empty
    # Confidence threshold
    confidence: float = 0.45
    # IOU threshold for NMS
    iou_threshold: float = 0.5
    # Target COCO class IDs to track (0 = person)
    target_classes: List[int] = field(default_factory=lambda: [0])
    # Maximum number of detections
    max_detections: int = 50
    # CUDA stream for async inference
    use_cuda_stream: bool = True
    # Warm-up iterations before real inference
    warmup_iterations: int = 10
    # Processing resolution (model input size)
    input_size: Tuple[int, int] = (640, 640)
    # FP16 precision for TensorRT
    use_fp16: bool = True


@dataclass
class OverlayConfig:
    """Overlay visualization configuration."""
    # Box color (RGBA)
    box_color: Tuple[int, int, int, int] = (0, 255, 0, 200)
    # Box line width
    line_width: int = 2
    # Label font size
    font_size: int = 12
    # Label color
    label_color: Tuple[int, int, int, int] = (255, 255, 255, 255)
    # Label background color
    label_bg: Tuple[int, int, int, int] = (0, 0, 0, 150)
    # Whether to show confidence score
    show_confidence: bool = True
    # Whether to show distance info
    show_distance: bool = True
    # Overlay window title
    window_title: str = "DetectionOverlay"
    # Refresh interval (ms)
    refresh_interval: int = 16  # ~60 FPS


@dataclass
class ControlConfig:
    """Cursor control configuration."""
    # Number of smoothing steps for cursor movement
    smoothing_steps: int = 20
    # Dead zone radius in pixels (cursor locked if within this range)
    dead_zone: int = 2
    # Maximum angular speed in degrees/second
    max_angular_speed: float = 360.0
    # Movement delay between steps (seconds)
    step_delay: float = 0.001
    # Enable cursor control
    enabled: bool = True


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    # Whether to run the overlay visualization
    enable_overlay: bool = True
    # Whether to run cursor control
    enable_control: bool = True
    # Log level
    log_level: str = "INFO"
    # Pipeline frame queue size
    frame_queue_size: int = 2


def get_config() -> SystemConfig:
    """Return the default system configuration."""
    return SystemConfig()
