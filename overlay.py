"""Transparent overlay window for detection visualization."""
import logging
import ctypes
from typing import List, Tuple

logger = logging.getLogger(__name__)

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_POPUP = 0x80000000
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01


class DetectionOverlay:
    BOX_COLOR = (0, 255, 0, 200)
    LINE_WIDTH = 2
    FONT_SIZE = 14
    LABEL_COLOR = (255, 255, 255, 255)
    LABEL_BG = (0, 0, 0, 150)

    def __init__(self):
        self._detections: List[Tuple] = []
        self._screen_w = 1920
        self._screen_h = 1080
        self._window = None
        self._running = False
        self._wndproc = None

    def start(self) -> bool:
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32
            kernel32 = ctypes.windll.kernel32

            self._screen_w = user32.GetSystemMetrics(0)
            self._screen_h = user32.GetSystemMetrics(1)
            logger.info(f"Overlay screen: {self._screen_w}x{self._screen_h}")

            hinst = kernel32.GetModuleHandleW(None)
            class_name = "DetectionOverlayClass"

            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_longlong, ctypes.c_void_p,
                ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64,
            )

            def wnd_proc(hwnd, msg, wparam, lparam):
                return 0
            self._wndproc = wnd_proc

            WNDCLASSW = type("WNDCLASSW", (ctypes.Structure,), {
                "_fields_": [
                    ("style", ctypes.c_uint),
                    ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", ctypes.c_void_p),
                    ("hIcon", ctypes.c_void_p),
                    ("hCursor", ctypes.c_void_p),
                    ("hbrBackground", ctypes.c_void_p),
                    ("lpszMenuName", ctypes.c_wchar_p),
                    ("lpszClassName", ctypes.c_wchar_p),
                ]
            })

            wc = WNDCLASSW()
            wc.lpfnWndProc = WNDPROC(wnd_proc)
            wc.hInstance = hinst
            wc.lpszClassName = class_name

            user32.RegisterClassW(ctypes.byref(wc))

            ex_style = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST
            self._window = user32.CreateWindowExW(
                ex_style, class_name, "Detection Overlay", WS_POPUP,
                0, 0, self._screen_w, self._screen_h,
                0, 0, hinst, 0,
            )

            if not self._window:
                logger.error("CreateWindowEx failed")
                return False

            user32.SetLayeredWindowAttributes(self._window, 0, 255, ULW_ALPHA)
            user32.ShowWindow(self._window, 1)
            self._running = True
            logger.info("Overlay window created")
            return True

        except Exception as e:
            logger.error(f"Overlay init failed: {e}")
            return False

    def update_detections(self, detections: List[Tuple]) -> None:
        self._detections = detections

    def draw(self) -> None:
        if not self._window or not self._running:
            return
        try:
            user32 = ctypes.windll.user32
            gdi32 = ctypes.windll.gdi32

            hdc_screen = user32.GetDC(0)
            hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
            hbm = gdi32.CreateCompatibleBitmap(hdc_screen, self._screen_w, self._screen_h)
            old_bmp = gdi32.SelectObject(hdc_mem, hbm)

            brush = gdi32.CreateSolidBrush(0x00000000)
            rect_t = type("RECT", (ctypes.Structure,), {
                "_fields_": [
                    ("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long),
                ]
            })
            rect = rect_t(0, 0, self._screen_w, self._screen_h)
            user32.FillRect(hdc_mem, ctypes.byref(rect), brush)
            gdi32.DeleteObject(brush)

            font = gdi32.CreateFontW(
                -self.FONT_SIZE, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "Consolas"
            )
            old_font = gdi32.SelectObject(hdc_mem, font)

            for det in self._detections:
                x1, y1, x2, y2, conf, cls_id = det
                self._draw_box(gdi32, hdc_mem, x1, y1, x2, y2)
                self._draw_label(gdi32, hdc_mem, x1, y1, x2, y2, conf)

            gdi32.SelectObject(hdc_mem, old_font)
            gdi32.DeleteObject(font)
            gdi32.SelectObject(hdc_mem, old_bmp)

            blend = type("BLENDFUNCTION", (ctypes.Structure,), {
                "_fields_": [
                    ("BlendOp", ctypes.c_ubyte),
                    ("BlendFlags", ctypes.c_ubyte),
                    ("SourceConstantAlpha", ctypes.c_ubyte),
                    ("AlphaFormat", ctypes.c_ubyte),
                ]
            })()
            blend.BlendOp = AC_SRC_OVER
            blend.SourceConstantAlpha = 255
            blend.AlphaFormat = AC_SRC_ALPHA

            sz = type("SIZE", (ctypes.Structure,), {
                "_fields_": [("cx", ctypes.c_long), ("cy", ctypes.c_long)]
            })()
            sz.cx, sz.cy = self._screen_w, self._screen_h

            pt = type("POINT", (ctypes.Structure,), {
                "_fields_": [("x", ctypes.c_long), ("y", ctypes.c_long)]
            })(0, 0)

            user32.UpdateLayeredWindow(
                self._window, hdc_screen, None, ctypes.byref(sz),
                hdc_mem, ctypes.byref(pt), 0, ctypes.byref(blend), ULW_ALPHA,
            )

            user32.ReleaseDC(0, hdc_screen)
            gdi32.DeleteDC(hdc_mem)
            gdi32.DeleteObject(hbm)

            msg = type("MSG", (ctypes.Structure,), {
                "_fields_": [
                    ("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                    ("wParam", ctypes.c_uint64), ("lParam", ctypes.c_longlong),
                    ("time", ctypes.c_uint), ("pt_x", ctypes.c_long), ("pt_y", ctypes.c_long),
                ]
            })()
            while user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 1):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

        except Exception as e:
            logger.error(f"Draw failed: {e}")

    def _draw_box(self, gdi32, hdc, x1, y1, x2, y2):
        r, g, b, a = self.BOX_COLOR
        color = r | (g << 8) | (b << 16)
        pen = gdi32.CreatePen(0, self.LINE_WIDTH, color)
        old_pen = gdi32.SelectObject(hdc, pen)
        nb = gdi32.GetStockObject(5)
        old_br = gdi32.SelectObject(hdc, nb)
        gdi32.Rectangle(hdc, x1, y1, x2, y2)
        gdi32.SelectObject(hdc, old_pen)
        gdi32.SelectObject(hdc, old_br)
        gdi32.DeleteObject(pen)

    def _draw_label(self, gdi32, hdc, x1, y1, x2, y2, conf):
        label = f"person {conf:.0%}"
        r_bg, g_bg, b_bg, _ = self.LABEL_BG
        bg = r_bg | (g_bg << 8) | (b_bg << 16)
        brush = gdi32.CreateSolidBrush(bg)
        lh = self.FONT_SIZE + 4
        old = gdi32.SelectObject(hdc, brush)
        gdi32.Rectangle(hdc, x1, y1 - lh, x2, y1)
        gdi32.SelectObject(hdc, old)
        gdi32.DeleteObject(brush)
        r_t, g_t, b_t, _ = self.LABEL_COLOR
        gdi32.SetTextColor(hdc, r_t | (g_t << 8) | (b_t << 16))
        gdi32.SetBkMode(hdc, 1)
        buf = ctypes.create_unicode_buffer(label)
        gdi32.TextOutW(hdc, x1 + 2, y1 - lh + 2, buf, len(label))

    def stop(self) -> None:
        self._running = False
        if self._window:
            try:
                ctypes.windll.user32.DestroyWindow(self._window)
            except Exception:
                pass
            self._window = None
        logger.info("Overlay stopped.")