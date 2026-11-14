"""决定性实验：图库用"检测框裁图"重建（而不是 SAM 紧框），身份能到多少？

之前 98.7% 是用**点击中心**裁的图测的，而线上推理拿到的是 YOLO 检测框
（更松、会偏移、有时合并）——分布不一致正是线上掉到 ~50% 的主因。
这里直接用检测框 + 抖动来建库，并在**留出视频**上测。
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.identity import Encoder, Gallery, crop_box  # noqa: E402
from eyesofpriestess.postprocess import MergeSuppressor  # noqa: E402
from eyesofpriestess.schema import load_clicks  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def jitter(box, dx, dy, scale):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return [cx - w / 2, cy - h / 2, w, h]


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    encoder = Encoder("resnet18", device="cuda", checkpoint=ROOT / "weights/embedder.pt")
    clicks = load_clicks(ROOT / "clicks")

    train_videos = ("v001_1-8", "v000_1-7")
    test_video = "v002_1-9"

    # 1) 从训练视频的检测框建库
    bank: dict[str, list[np.ndarray]] = defaultdict(list)
    frames = [f for f in clicks if f.startswith(train_videos)]
    for frame in frames:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        points = [(c.cls, c.x, c.y) for c in clicks[frame] if not c.is_ignore and c.cls]
        if not points:
            continue
        result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(boxes)
        for code, x, y in points:
            half_w, half_h = max(8.0, 1920 * 0.015), max(8.0, 1080 * 0.03)
            gt = [x - half_w, y - half_h, x + half_w, y + half_h]
            best, best_iou = -1, 0.3
            for index, box in enumerate(boxes):
                value = iou(gt, box)
                if value > best_iou:
                    best, best_iou = index, value
            if best < 0:
                continue
            box = boxes[best]
            for dx, dy, scale in ((0, 0, 1.0), (3, 0, 0.95), (-3, 0, 1.05), (0, 4, 1.0)):
                crop = crop_box(image, jitter(box, dx, dy, scale), pad=0.0)
                if crop.size:
                    bank[code].append(encoder.embed([crop])[0])
    print("建库:", {code: len(items) for code, items in bank.items()})

    prototypes = {code: np.mean(np.stack(items), axis=0) for code, items in bank.items()}
    codes = list(prototypes)
    matrix = np.stack([prototypes[code] for code in codes])
    gallery = Gallery(codes=codes, prototypes=matrix, samples={c: len(bank[c]) for c in codes})

    # 2) 在留出视频上测
    correct = total = 0
    confusion: dict[str, int] = defaultdict(int)
    for frame in [f for f in clicks if f.startswith(test_video)]:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        points = [(c.cls, c.x, c.y) for c in clicks[frame] if not c.is_ignore and c.cls]
        if not points:
            continue
        result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(boxes)
        if not boxes:
            total += len(points)
            continue
        matches = gallery.identify(encoder.embed(
            [crop_box(image, [b[0], b[1], b[2] - b[0], b[3] - b[1]], pad=0.0) for b in boxes]))
        for code, x, y in points:
            half_w, half_h = max(8.0, 1920 * 0.015), max(8.0, 1080 * 0.03)
            gt = [x - half_w, y - half_h, x + half_w, y + half_h]
            best, best_iou = -1, 0.3
            for index, box in enumerate(boxes):
                value = iou(gt, box)
                if value > best_iou:
                    best, best_iou = index, value
            total += 1
            if best >= 0:
                if matches[best].code == code:
                    correct += 1
                else:
                    confusion[f"{code}->{matches[best].code}"] += 1
    print(f"留出 {test_video}: {correct}/{total} = {correct/max(1,total):.1%}")
    print("混淆:", sorted(confusion.items(), key=lambda kv: -kv[1])[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
