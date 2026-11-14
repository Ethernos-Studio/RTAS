"""扫去重阈值：重复框 vs 误合并，找平衡点。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.postprocess import MergeSuppressor, dedupe  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    data = ROOT / "dataset"
    images = sorted((data / "images/val").glob("*.jpg"))

    cache = []
    for image_path in images:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt = []
        for line in (data / "labels/val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = (float(v) for v in line.split())
            if int(cls) == 0:
                gt.append([(cx - w / 2) * width, (cy - h / 2) * height,
                           (cx + w / 2) * width, (cy + h / 2) * height])
        result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
        raw = [([float(v) for v in box.xyxy[0].tolist()], float(box.conf))
               for box in (result.boxes or []) if int(box.cls[0]) == 0]
        cache.append((gt, raw))

    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.70):
        merged = missed = dupes = 0
        for gt, raw in cache:
            boxes = MergeSuppressor()([box for box, _ in raw])
            kept = [box for box in boxes]
            deduped = [box for box, _ in dedupe([(box, 1.0) for box in kept],
                                               iou_threshold=threshold)]
            dupes += len(kept) - len(deduped)
            # 合并框：一个预测框里塞了 >=2 个 GT
            for box in deduped:
                inside = 0
                for g in gt:
                    x1, y1 = max(box[0], g[0]), max(box[1], g[1])
                    x2, y2 = min(box[2], g[2]), min(box[3], g[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if inter / max((g[2] - g[0]) * (g[3] - g[1]), 1e-6) >= 0.7:
                        inside += 1
                if inside >= 2:
                    merged += 1
            for g in gt:
                if not any(iou(box, g) >= 0.5 for box in deduped):
                    missed += 1
        print(f"去重阈值 {threshold:.2f}: 去掉重复 {dupes:2d} 个，合并框 {merged:2d} 个，漏检 {missed:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
