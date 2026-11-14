"""看看误报到底落在哪：把 val 上"没匹配到 GT"的框截出来，并统计框底血条颜色。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.bars import find_bars  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def bar_color_near(image: np.ndarray, box, bars) -> str:
    """看框内/框下缘附近有没有条，返回它的颜色。"""
    x1, y1, x2, y2 = box
    for bar in bars:
        cx, cy = (bar.x1 + bar.x2) / 2, (bar.y1 + bar.y2) / 2
        if x1 - 4 <= cx <= x2 + 4 and y1 <= cy <= y2 + 24:
            return bar.kind
    return "none"


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    data = ROOT / "dataset"
    tiles = []
    stats: dict[str, int] = {}

    for image_path in sorted((data / "images/val").glob("*.jpg"))[:6]:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        label_path = data / "labels/val" / f"{image_path.stem}.txt"
        gt = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = (float(v) for v in line.split())
            gt.append([(cx - w / 2) * width, (cy - h / 2) * height,
                       (cx + w / 2) * width, (cy + h / 2) * height])

        result = model.predict(image, imgsz=960, conf=0.25, iou=0.8, verbose=False)[0]
        bars = find_bars(image)
        for box in result.boxes or []:
            if int(box.cls[0]) != 0:
                continue
            coords = list(map(float, box.xyxy[0].tolist()))
            if any(iou(coords, g) >= 0.5 for g in gt):
                continue
            color = bar_color_near(image, coords, bars)
            stats[color] = stats.get(color, 0) + 1
            x1, y1, x2, y2 = (int(v) for v in coords)
            crop = image[max(0, y1 - 6):y2 + 26, max(0, x1 - 6):x2 + 6]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (96, 96))
            cv2.putText(crop, f"{color[:4]} {float(box.conf[0]):.2f}", (3, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            tiles.append(crop)

    print("误报框内/下方血条颜色统计:", stats, "（red=敌人蓝/绿=友军或技能条）")
    if tiles:
        rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
        cv2.imwrite(str(ROOT / "out" / "_fp_crops.jpg"), np.vstack(rows))
        print(f"误报裁剪已存 out/_fp_crops.jpg，共 {len(tiles)} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
