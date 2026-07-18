"""
Real-time Screen Object Detection and Visualization System
==========================================================

Three-layer pipeline architecture:
  1. Capture Layer    - DXGI/DWM desktop screengrab (dxcam)
  2. Perception Layer - YOLOv8n object detection (ONNX/TensorRT/torch)
  3. Visualization    - Win32 transparent overlay + cursor control

Hardware target: Lenovo Y7000 2024 (RTX 4060)

Usage:
    python main.py                # Default: full screen, person detection
    python main.py --region x y w h  # Capture specific region
    python main.py --no-overlay       # Disable overlay
    python main.py --no-control       # Disable cursor movement
    python main.py --conf 0.5         # Set confidence threshold
"""

import argparse
import logging
import signal
import sys
import time
from collections import deque
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time Screen Object Detection System"
    )
    parser.add_argument(
        "--region", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
        help="Capture region (x y w h), default: full screen",
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Detection confidence threshold (default: 0.45)",
    )
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Disable the overlay visualization window",
    )
    parser.add_argument(
        "--no-control", action="store_true",
        help="Disable cursor movement control",
    )
    parser.add_argument(
        "--backend", choices=["tensorrt", "onnx", "torch"], default="",
        help="Inference backend override (default: auto-detect)",
    )
    parser.add_argument(
        "--target-size", type=int, default=640,
        help="Model input size (default: 640)",
    )
    parser.add_argument(
        "--smoothing", type=int, default=20,
        help="Cursor smoothing steps (default: 20)",
    )
    parser.add_argument(
        "--dead-zone", type=int, default=2,
        help="Cursor dead zone in pixels (default: 2)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging",
    )
    parser.add_argument(
        "--generate-onnx", action="store_true",
        help="Generate ONNX model from PyTorch and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Real-time Screen Object Detection System")
    logger.info("Hardware: Lenovo Y7000 2024 (RTX 4060)")
    logger.info("=" * 60)

    # ---- 1. Initialize Capture Layer ----
    from capture.dxgi_capture import ScreenCapture

    region = None
    if args.region:
        region = tuple(args.region)
        logger.info(f"Capture region: {region}")

    capture = ScreenCapture(
        region=region,
        target_width=args.target_size,
        target_height=args.target_size,
        output_idx=0,
        max_fps=0,
    )

    # ---- 2. Initialize Perception Layer ----
    from perception.yolo_detector import YOLODetector
    from utils.config import PerceptionConfig

    pcfg = PerceptionConfig()
    pcfg.confidence = args.conf
    pcfg.input_size = (args.target_size, args.target_size)

    detector = YOLODetector(
        confidence=pcfg.confidence,
        iou_threshold=pcfg.iou_threshold,
        target_classes=pcfg.target_classes,
        input_size=pcfg.input_size,
        backend=args.backend or pcfg.backend,
        warmup_iterations=pcfg.warmup_iterations,
    )

    # Generate ONNX and exit if requested
    if args.generate_onnx:
        logger.info("Exporting ONNX model...")
        success = detector.generate_onnx(pcfg.model_pt, pcfg.model_onnx)
        logger.info(f"ONNX export {'succeeded' if success else 'failed'}")
        capture.stop()
        return

    logger.info(f"Active backend: {detector.backend}")

    # ---- 3. Initialize Overlay Layer ----
    overlay = None
    if not args.no_overlay:
        from overlay.overlay import DetectionOverlay
        overlay = DetectionOverlay()
        if not overlay.start():
            logger.warning("Overlay initialization failed; continuing without overlay")
            overlay = None

    # ---- 4. Initialize Cursor Controller ----
    cursor_ctrl = None
    if not args.no_control:
        from control.cursor import CursorController
        cursor_ctrl = CursorController(
            smoothing_steps=args.smoothing,
            dead_zone=args.dead_zone,
            max_angular_speed=360.0,
            step_delay=0.001,
        )
        logger.info(
            f"Cursor control: smoothing={args.smoothing}, "
            f"deadzone={args.dead_zone}px"
        )

    # ---- Frame ring buffer for pipeline throughput measurement ----
    frame_times = deque(maxlen=100)
    detect_times = deque(maxlen=100)
    overlay_times = deque(maxlen=100)
    detection_count = 0
    frame_count = 0

    # ---- Graceful shutdown ----
    shutdown_flag = False

    def on_shutdown(sig, frame):
        nonlocal shutdown_flag
        logger.info("Shutdown signal received...")
        shutdown_flag = True

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    logger.info("Pipeline started. Press Ctrl+C to stop.")
    logger.info("-" * 60)

    # ---- Main Pipeline Loop ----
    try:
        while not shutdown_flag:
            loop_start = time.perf_counter()

            # Layer 1: Capture
            t0 = time.perf_counter()
            frame = capture.capture()
            t_cap = (time.perf_counter() - t0) * 1000
            frame_times.append(t_cap)

            if frame is None:
                time.sleep(0.001)
                continue
            frame_count += 1

            # Layer 2: Detect
            t0 = time.perf_counter()
            detections = detector.detect(frame)
            t_det = (time.perf_counter() - t0) * 1000
            detect_times.append(t_det)

            if detections:
                detection_count += 1

            # Layer 3a: Visualization (overlay)
            if overlay and overlay._running:
                t0 = time.perf_counter()
                overlay.update_detections(detections)
                overlay.draw()
                t_draw = (time.perf_counter() - t0) * 1000
                overlay_times.append(t_draw)

            # Layer 3b: Cursor control
            if cursor_ctrl and detections:
                # Select best detection (highest confidence * area)
                best = max(
                    detections,
                    key=lambda d: d[4] * ((d[2] - d[0]) * (d[3] - d[1])),
                )
                target_center = cursor_ctrl.get_target_center(best)
                screen_center = cursor_ctrl.get_screen_center()
                cursor_ctrl.move_to_target(target_center, screen_center)

            # Status logging every 60 frames
            if frame_count % 60 == 0 and frame_count > 0:
                avg_cap = sum(frame_times) / max(len(frame_times), 1)
                avg_det = sum(detect_times) / max(len(detect_times), 1)
                avg_ov = sum(overlay_times) / max(len(overlay_times), 1) if overlay_times else 0
                total_lat = avg_cap + avg_det + avg_ov
                logger.info(
                    f"Frame #{frame_count:4d} | "
                    f"Capture: {avg_cap:5.1f}ms | "
                    f"Detect: {avg_det:5.1f}ms | "
                    f"Overlay: {avg_ov:4.1f}ms | "
                    f"Total: {total_lat:5.1f}ms | "
                    f"Detections: {detection_count}"
                )

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down...")
        if overlay:
            overlay.stop()
        capture.stop()
        logger.info("Shutdown complete.")
        logger.info(f"Stats: {frame_count} frames, {detection_count} detections")
        if frame_times:
            logger.info(
                f"Avg latency: capture={sum(frame_times)/len(frame_times):.2f}ms, "
                f"detect={sum(detect_times)/len(detect_times):.2f}ms"
            )


if __name__ == "__main__":
    main()