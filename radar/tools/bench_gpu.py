"""看一下瓶颈到底在 GPU 还是 CPU：批量大小 vs FPS，同时采样 GPU 利用率。"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

FRAMES = [cv2.imread(str(p)) for p in sorted((ROOT / "frames").glob("v001_1-8_*.jpg"))[:32]]


def gpu_util_samples(stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        try:
            text = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if text:
                util, mem = [part.strip() for part in text.split(",")[:2]]
                out.append(float(util))
        except Exception:
            pass
        time.sleep(0.15)


def bench(model, imgsz: int, batch: int) -> None:
    model.predict(FRAMES[:batch], imgsz=imgsz, verbose=False, device="0")
    util: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=gpu_util_samples, args=(stop, util), daemon=True)
    sampler.start()
    start = time.perf_counter()
    done = 0
    for i in range(0, len(FRAMES) - batch + 1, batch):
        model.predict(FRAMES[i:i + batch], imgsz=imgsz, conf=0.25, verbose=False, device="0")
        done += batch
    elapsed = time.perf_counter() - start
    stop.set()
    sampler.join(timeout=1)
    peak = max(util) if util else 0.0
    avg = sum(util) / len(util) if util else 0.0
    print(f"imgsz={imgsz:4d} batch={batch:2d}: {done/elapsed:5.1f} FPS   "
          f"GPU util 平均 {avg:4.0f}% / 峰值 {peak:3.0f}%")


if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    for imgsz in (1280, 960):
        for batch in (1, 4, 8, 16):
            if batch > len(FRAMES):
                continue
            bench(model, imgsz, batch)
