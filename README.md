
实时屏幕目标检测与可视化系统。 功能：通过 DXGI 桌面采集实时截取屏幕画面，YOLOv8n/s 目标检测模型识别画面中的人物目标，Win32 透明覆盖层实时绘制检测边界框，支持光标平滑追踪。 YOLOv8 + ONNX Runtime + dxcam (DXGI) + Win32 GDI + PyTorch。三层流水线并行——采集层、感知层、可视化层互不阻塞。推理后端支持 ONNX Runtime / TensorRT / PyTorch 自动切换。覆盖层使用 WS_EX_LAYERED 透明窗口 + DIB Section 32位位图实现 per-pixel alpha 渲染。光标追踪采用指数衰减平滑算法。 TensorRT FP16 推理延迟约 5-7ms，端到端总延迟约 12ms。
