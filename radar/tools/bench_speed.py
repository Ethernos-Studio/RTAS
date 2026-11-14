"""推理速度矩阵：模型 / 精度 / 输入尺寸，找出能跑到 30FPS 的配置。

用法：
    uv run python tools/bench_speed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

FRAMES = sorted((ROOT / "frames").glob("v001_1-8_*.jpg"))[:24]


def bench(model, *, imgsz: int, half: bool = False, device: str = "0", label: str = "") -> float:
    images = [cv2.imread(str(path)) for path in FRAMES]
    model.predict(images[0], imgsz=imgsz, half=half, device=device, verbose=False)   # 预热
    start = time.perf_counter()
    for image in images:
        model.predict(image, imgsz=imgsz, half=half, device=device, conf=0.25, verbose=False)
    elapsed = time.perf_counter() - start
    fps = len(images) / elapsed
    print(f"{label:<34} {fps:6.1f} FPS   {elapsed / len(images) * 1000:5.1f} ms/帧")
    return fps


def main() -> int:
    from ultralytics import YOLO

    weights_s = ROOT / "runs/arknights/v1/weights/best.pt"
    model_s = YOLO(str(weights_s))
    print(f"GPU: {model_s.predictor and '' or ''}", end="")

    bench(model_s, imgsz=1280, label="v1(yolov8s) fp32 @1280")
    bench(model_s, imgsz=1280, half=True, label="v1(yolov8s) fp16 @1280")
    bench(model_s, imgsz=960, label="v1(yolov8s) fp32 @960")
    bench(model_s, imgsz=960, half=True, label="v1(yolov8s) fp16 @960")
    bench(model_s, imgsz=768, half=True, label="v1(yolov8s) fp16 @768")

    nano = YOLO("yolov8n.pt")
    bench(nano, imgsz=1280, half=True, label="yolov8n fp16 @1280")
    bench(nano, imgsz=960, half=True, label="yolov8n fp16 @960")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
