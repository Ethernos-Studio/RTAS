"""拆开量：解码 / 检测 / 画框 各占多少时间（持续负载，接近真实使用）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.frames import open_video  # noqa: E402

SRC = ROOT / "raw" / "1-8.mp4"
N = 240


def decode_only() -> tuple[list[np.ndarray], float]:
    stream = open_video(SRC, sample_fps=30.0, start=31.0)
    frames = []
    start = time.perf_counter()
    for _t, frame in stream.frames:
        frames.append(frame)
        if len(frames) >= N:
            break
    elapsed = time.perf_counter() - start
    stream.close()
    print(f"解码: {len(frames)} 帧 / {elapsed:.2f}s = {len(frames)/elapsed:5.1f} FPS")
    return frames, elapsed


def detect(frames, imgsz: int, batch: int, device: str | None) -> float:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    model.predict(frames[:batch], imgsz=imgsz, verbose=False, device=device)
    start = time.perf_counter()
    done = 0
    for i in range(0, len(frames) - batch + 1, batch):
        model.predict(frames[i:i + batch], imgsz=imgsz, conf=0.25, verbose=False, device=device)
        done += batch
    elapsed = time.perf_counter() - start
    print(f"检测 imgsz={imgsz} batch={batch} device={device}: {done/elapsed:5.1f} FPS")
    return elapsed


def draw(frames) -> float:
    start = time.perf_counter()
    for frame in frames:
        canvas = cv2.resize(frame, (1280, 720))
        for _ in range(6):
            cv2.rectangle(canvas, (100, 100), (160, 150), (90, 220, 90), 2)
            cv2.putText(canvas, "B2 0.93", (100, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 90), 1)
        cv2.putText(canvas, f"frame 123 enemies 6 imgsz 960 batch 4", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    elapsed = time.perf_counter() - start
    print(f"画框+缩放: {len(frames)} 帧 / {elapsed:.2f}s = {len(frames)/elapsed:5.1f} FPS")
    return elapsed


def identity(frames, imgsz: int = 960, batch: int = 4) -> float:
    from eyesofpriestess.identity import Encoder, Gallery, crop_box
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    gallery = Gallery.load(ROOT / "weights/gallery.npz")
    encoder = Encoder("resnet18", device="cuda")
    encoder.embed([np.zeros((32, 32, 3), np.uint8)])          # 预热
    start = time.perf_counter()
    done = 0
    for i in range(0, len(frames) - batch + 1, batch):
        chunk = frames[i:i + batch]
        results = model.predict(chunk, imgsz=imgsz, conf=0.25, verbose=False, device="0")
        all_crops = []
        for frame, result in zip(chunk, results):
            for box in result.boxes or []:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                all_crops.append(crop_box(frame, [x1, y1, x2 - x1, y2 - y1], pad=0.0))
        if all_crops:
            matches = gallery.identify(encoder.embed(all_crops))
            done += len(matches)
    elapsed = time.perf_counter() - start
    print(f"检测+身份: {done} 个框 / {elapsed:.2f}s（含检测）")
    return elapsed


if __name__ == "__main__":
    frames, _ = decode_only()
    detect(frames, 1280, 1, "0")
    detect(frames, 960, 4, "0")
    detect(frames, 768, 4, "0")
    draw(frames)
    identity(frames)
