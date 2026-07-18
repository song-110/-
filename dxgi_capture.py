"""
Low-latency screen capture using DXGI Desktop Duplication API via dxcam.
"""
import logging
import time
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ScreenCapture:
    """High-performance screen capture using DXGI Desktop Duplication API."""

    def __init__(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_width: int = 640,
        target_height: int = 640,
        output_idx: int = 0,
        max_fps: int = 0,
    ):
        self.region = region
        self.target_width = target_width
        self.target_height = target_height
        self.max_fps = max_fps
        self._camera = None
        self._backend = None
        self._prev_frame_time = 0
        self._min_frame_interval = 1.0 / max_fps if max_fps > 0 else 0
        self._init_backend(output_idx)

    def _init_backend(self, output_idx: int) -> None:
        try:
            import dxcam
            self._camera = dxcam.create(output_idx=output_idx, output_color="BGR")
            self._backend = "dxcam"
            logger.info("Screen capture backend: dxcam (DXGI Desktop Duplication)")
            self._camera.start(target_fps=0, video_mode=True)
            return
        except ImportError:
            logger.warning("dxcam not available; falling back to mss")
        except Exception as e:
            logger.warning(f"dxcam init failed ({e}); falling back to mss")
        try:
            import mss
            self._mss = mss.mss()
            if self.region:
                self._monitor = {
                    "left": self.region[0], "top": self.region[1],
                    "width": self.region[2], "height": self.region[3],
                }
            else:
                self._monitor = self._mss.monitors[1]
            self._backend = "mss"
            logger.info("Screen capture backend: mss")
        except ImportError:
            raise ImportError(
                "No screen capture backend available. Install dxcam or mss."
            )

    def capture(self) -> Optional[np.ndarray]:
        if self._backend == "dxcam":
            return self._capture_dxcam()
        elif self._backend == "mss":
            return self._capture_mss()
        return None

    def capture_tensor(self) -> Optional[np.ndarray]:
        frame = self.capture()
        if frame is None:
            return None
        frame_rgb = frame[..., ::-1]
        try:
            import cv2
            frame_resized = cv2.resize(
                frame_rgb, (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR,
            )
        except ImportError:
            h, w = frame_rgb.shape[:2]
            h_step = h / self.target_height
            w_step = w / self.target_width
            h_idx = (np.arange(self.target_height) * h_step).astype(int)
            w_idx = (np.arange(self.target_width) * w_step).astype(int)
            frame_resized = frame_rgb[h_idx[:, None], w_idx]
        tensor = np.ascontiguousarray(
            frame_resized.transpose(2, 0, 1)[None, ...]
        ).astype(np.float32) / 255.0
        return tensor

    def _capture_dxcam(self) -> Optional[np.ndarray]:
        if self._camera is None:
            return None
        now = time.perf_counter()
        if now - self._prev_frame_time < self._min_frame_interval:
            return None
        self._prev_frame_time = now
        frame = self._camera.get_latest_frame()
        if frame is None:
            return None
        if self.region is not None:
            x, y, w, h = self.region
            frame = frame[y:y + h, x:x + w]
        return frame

    def _capture_mss(self) -> Optional[np.ndarray]:
        now = time.perf_counter()
        if now - self._prev_frame_time < self._min_frame_interval:
            return None
        self._prev_frame_time = now
        sct = self._mss.grab(self._monitor)
        frame = np.asarray(sct)[:, :, :3]
        return frame

    def stop(self) -> None:
        if self._backend == "dxcam" and self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
        self._camera = None
        logger.info("Screen capture stopped.")

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()