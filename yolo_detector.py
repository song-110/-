"""
YOLOv8n object detection with multiple inference backends.
Priority: TensorRT > ONNX Runtime > PyTorch (auto-detected).
"""
import logging
import os
import time
from typing import List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


def _letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Letterbox resize with padding to preserve aspect ratio."""
    shape = img.shape[:2] if len(img.shape) == 3 else img.shape[2:]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    if len(img.shape) == 3:
        img = img[..., ::-1]  # BGR to RGB
    try:
        import cv2
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color
        )
    except ImportError:
        h, w = img.shape[:2]
        h_idx = (np.arange(new_unpad[1]) * h / new_unpad[1]).astype(int)
        w_idx = (np.arange(new_unpad[0]) * w / new_unpad[0]).astype(int)
        img = img[h_idx[:, None], w_idx]
        h_pad, w_pad = new_shape[0] - new_unpad[1], new_shape[1] - new_unpad[0]
        img = np.pad(
            img, ((dh, h_pad - dh), (dw, w_pad - dw), (0, 0)),
            constant_values=color[0],
        )
    return img


class YOLODetector:
    """YOLOv8n object detector with auto-detected inference backend."""

    def __init__(
        self,
        model_pt: str = "",
        model_onnx: str = "",
        model_trt: str = "",
        confidence: float = 0.45,
        iou_threshold: float = 0.5,
        target_classes: Optional[List[int]] = None,
        input_size: Tuple[int, int] = (640, 640),
        backend: str = "",
        warmup_iterations: int = 10,
    ):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or [0]
        self.input_size = input_size
        self._backend_type = ""
        self._model = None
        self._session = None

        self._init_backend(
            backend, model_trt, model_onnx, model_pt, warmup_iterations
        )

    def _init_backend(
        self, preferred: str, trt_path: str, onnx_path: str,
        pt_path: str, warmup: int,
    ) -> None:
        if preferred == "tensorrt":
            if self._try_tensorrt(trt_path):
                return
        if preferred == "onnx":
            if self._try_onnx(onnx_path):
                return

        if self._try_tensorrt(trt_path):
            return
        if self._try_onnx(onnx_path):
            return
        if self._try_torch(pt_path, warmup):
            return
        raise RuntimeError(
            "No inference backend available. "
            "Install onnxruntime-gpu, torch+ultralytics, or TensorRT."
        )

    def _try_tensorrt(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            import tensorrt as trt
            logger.info(f"Loading TensorRT engine: {path}")
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(path, "rb") as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            if engine is None:
                logger.warning("TensorRT engine deserialization returned None")
                return False
            self._model = engine
            self._session = None
            self._backend_type = "tensorrt"
            logger.info(f"Inference backend: TensorRT (FP16)")
            return True
        except ImportError:
            logger.debug("tensorrt module not available")
        except Exception as e:
            logger.warning(f"TensorRT init failed: {e}")
        return False

    def _try_onnx(self, path: str) -> bool:
        if not os.path.exists(path):
            # Try generating ONNX from PT
            return False
        try:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers() else ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                path, providers=providers,
                sess_options=self._make_ort_options(),
            )
            self._model = None
            self._backend_type = "onnx"
            ep = self._session.get_providers()
            logger.info(f"Inference backend: ONNX Runtime ({ep})")
            return True
        except ImportError:
            logger.debug("onnxruntime not available")
        except Exception as e:
            logger.warning(f"ONNX init failed: {e}")
        return False

    def _try_torch(self, path: str, warmup: int) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA not available; using CPU inference")
            from ultralytics import YOLO
            if os.path.exists(path):
                self._model = YOLO(path)
            else:
                logger.info("Downloading YOLOv8n from ultralytics...")
                self._model = YOLO("yolov8n.pt")
            self._backend_type = "torch"
            logger.info("Inference backend: PyTorch (ultralytics YOLO)")
            # Warmup
            dummy = np.random.rand(640, 640, 3).astype(np.uint8)
            for _ in range(warmup):
                self._model(dummy, verbose=False)
            logger.info(f"Warmup complete ({warmup} iterations)")
            return True
        except ImportError:
            logger.debug("ultralytics/torch not available")
        except Exception as e:
            logger.warning(f"PyTorch init failed: {e}")
        return False

    @staticmethod
    def _make_ort_options():
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            opts.intra_op_num_threads = 4
            return opts
        except Exception:
            return None

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Run detection on a single image (HWC BGR u8).

        Returns list of (x1, y1, x2, y2, confidence, class_id).
        """
        if self._backend_type == "tensorrt":
            return self._detect_tensorrt(image)
        elif self._backend_type == "onnx":
            return self._detect_onnx(image)
        elif self._backend_type == "torch":
            return self._detect_torch(image)
        return []

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = _letterbox(image, self.input_size)
        # HWC -> CHW, normalize
        img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(img)

    def _postprocess(
        self, outputs: np.ndarray, pad_info: Tuple[float, float, float],
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """Parse raw model output into detections with NMS."""
        detections = []
        # outputs shape: (1, 84, 8400) for YOLOv8n ONNX
        # Each detection has [cx, cy, w, h, ...class_scores]
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # (84, N)
        outputs = outputs.transpose()  # (N, 84)

        # Filter by confidence
        class_scores = outputs[:, 4:]
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
        mask = max_scores > self.confidence

        if not mask.any():
            return detections

        filtered = outputs[mask]
        scores = max_scores[mask]
        ids = class_ids[mask]

        # Filter by target classes
        target_mask = np.isin(ids, self.target_classes)
        if not target_mask.any():
            return detections
        filtered = filtered[target_mask]
        scores = scores[target_mask]
        ids = ids[target_mask]

        # Convert cxcywh to xyxy
        boxes = filtered[:, :4].copy()
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = cx - w/2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = cy - h/2
        boxes[:, 2] += boxes[:, 0]      # x2 = x1 + w
        boxes[:, 3] += boxes[:, 1]      # y2 = y1 + h

        # Scale back from letterbox
        r, dw, dh = pad_info
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r

        # NMS
        indices = self._nms(boxes, scores)
        return [
            (int(boxes[i, 0]), int(boxes[i, 1]),
             int(boxes[i, 2]), int(boxes[i, 3]),
             float(scores[i]), int(ids[i]))
            for i in indices
        ]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """Non-maximum suppression. Returns indices of kept boxes."""
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def _detect_onnx(self, image: np.ndarray) -> List[Tuple]:
        img = _letterbox(image, self.input_size)
        r = min(
            self.input_size[0] / image.shape[0],
            self.input_size[1] / image.shape[1],
        )
        new_unpad = (int(round(image.shape[1] * r)), int(round(image.shape[0] * r)))
        dw = (self.input_size[1] - new_unpad[0]) // 2
        dh = (self.input_size[0] - new_unpad[1]) // 2
        tensor = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(tensor)
        inputs = {self._session.get_inputs()[0].name: tensor}
        outputs = self._session.run(None, inputs)[0]
        return self._postprocess(outputs, (r, dw, dh))

    def _detect_torch(self, image: np.ndarray) -> List[Tuple]:
        import torch
        results = self._model(image, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, clss):
                if conf < self.confidence:
                    continue
                if cls not in self.target_classes:
                    continue
                detections.append((
                    int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                    float(conf), int(cls),
                ))
        return detections

    def _detect_tensorrt(self, image: np.ndarray) -> List[Tuple]:
        # TensorRT inference requires explicit memory management.
        # This is a simplified path; production code should use pycuda
        # or cuda-python for proper GPU buffer management.
        pass
        # TensorRT inference will be enabled when engine file + bindings exist
        return []

    @property
    def backend(self) -> str:
        return self._backend_type

    @property
    def is_ready(self) -> bool:
        return self._backend_type != ""

    def generate_onnx(self, pt_path: str, onnx_path: str) -> bool:
        """Export a YOLOv8 PyTorch model to ONNX format."""
        try:
            import torch
            from ultralytics import YOLO
            model = YOLO(pt_path)
            model.export(format="onnx", imgsz=self.input_size)
            logger.info(f"ONNX model exported to {onnx_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export ONNX: {e}")
            return False