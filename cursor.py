"""
Cursor control with smooth movement interpolation, deadzone,
and angular speed limiting to simulate natural hand motion.
"""
import logging
import math
import time
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CursorController:
    """Smooth cursor movement controller with deadzone and speed limits."""

    def __init__(
        self,
        smoothing_steps: int = 20,
        dead_zone: int = 2,
        max_angular_speed: float = 360.0,
        step_delay: float = 0.001,
    ):
        self.smoothing_steps = smoothing_steps
        self.dead_zone = dead_zone
        self.max_angular_speed = max_angular_speed  # degrees/second
        self.step_delay = step_delay
        self._prev_target: Optional[Tuple[int, int]] = None

    def move_to_target(
        self,
        target_center: Tuple[int, int],
        screen_center: Tuple[int, int],
    ) -> bool:
        """Move cursor smoothly toward the target center.

        Returns True if cursor reached the target (within deadzone).
        """
        dx = target_center[0] - screen_center[0]
        dy = target_center[1] - screen_center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # Deadzone: lock cursor if within deadzone
        if distance <= self.dead_zone:
            return True

        # Calculate angle for speed limiting
        angle = math.atan2(dy, dx)
        steps = min(self.smoothing_steps, max(1, int(distance)))
        step_dx = dx / steps
        step_dy = dy / steps

        # Apply angular speed limiting
        max_pixels_per_step = self._max_pixel_step()
        step_distance = math.sqrt(step_dx ** 2 + step_dy ** 2)
        if step_distance > max_pixels_per_step:
            scale = max_pixels_per_step / step_distance
            step_dx *= scale
            step_dy *= scale

        try:
            import ctypes
            user32 = ctypes.windll.user32

            for i in range(steps):
                current_step_dx = int(round(step_dx * (steps - i) / steps
                                           + step_dx * i / steps))
                current_step_dy = int(round(step_dy * (steps - i) / steps
                                           + step_dy * i / steps))

                # Get current cursor position
                pt = type("POINT", (ctypes.Structure,), {
                    "_fields_": [("x", ctypes.c_long), ("y", ctypes.c_long)]
                })()
                user32.GetCursorPos(ctypes.byref(pt))

                new_x = pt.x + int(round(step_dx))
                new_y = pt.y + int(round(step_dy))

                # Clamp to screen bounds
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
                new_x = max(0, min(screen_w, new_x))
                new_y = max(0, min(screen_h, new_y))

                user32.SetCursorPos(new_x, new_y)

                if self.step_delay > 0:
                    time.sleep(self.step_delay)

            self._prev_target = target_center
            return True

        except ImportError:
            logger.warning("ctypes unavailable for cursor control")
            return False

    def _max_pixel_step(self) -> float:
        """Maximum pixel movement per step based on angular speed limit.

        Assumes ~360 pixels per full rotation at a reference distance,
        scaled by the step delay to match degrees/second.
        """
        # ~360 pixels/full-rotation as a reference
        degrees_per_step = self.max_angular_speed * self.step_delay
        pixels_per_step = (degrees_per_step / 360.0) * 360.0
        return pixels_per_step

    def calculate_offset(
        self, target_center: Tuple[int, int], screen_center: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Calculate pixel offset from screen center to target."""
        return (
            target_center[0] - screen_center[0],
            target_center[1] - screen_center[1],
        )

    @staticmethod
    def get_target_center(
        detection: Tuple[int, int, int, int, float, int],
    ) -> Tuple[int, int]:
        """Get the aim point from a detection box (top 1/4 center)."""
        x1, y1, x2, y2 = detection[:4]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 4  # Upper quarter for head/upper body
        return (int(cx), int(cy))

    @staticmethod
    def get_screen_center() -> Tuple[int, int]:
        """Get the current screen dimensions and center."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            return (w // 2, h // 2)
        except ImportError:
            return (960, 540)  # fallback

    @staticmethod
    def get_cursor_pos() -> Tuple[int, int]:
        """Get current cursor position."""
        try:
            import ctypes
            user32 = ctypes.windll.user32
            pt = type("POINT", (ctypes.Structure,), {
                "_fields_": [("x", ctypes.c_long), ("y", ctypes.c_long)]
            })()
            user32.GetCursorPos(ctypes.byref(pt))
            return (pt.x, pt.y)
        except ImportError:
            return (0, 0)