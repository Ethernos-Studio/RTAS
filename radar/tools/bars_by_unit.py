"""条只可能在单位附近：用检测框当 ROI，再看这一小条带里有没有条。

全图找条会捞到一堆地板花纹（实测 43 根里 30+ 是花纹/线缆/UI）。
真条的位置是确定的：敌人血条在**头顶**，干员血条在**脚下**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.bars import find_bars  # noqa: E402

FRAME = ROOT / "out" / "_battle_frame.jpg"


def bars_above(image, box, pad: int = 18) -> list:
    x1, y1, x2, _y2 = [int(v) for v in box]
    roi = image[max(0, y1 - pad):y1 + 4, max(0, x1 - 6):x2 + 6]
    if roi.size == 0:
        return []
    return find_bars(roi, min_width=20)


def bars_below(image, box, pad: int = 14) -> list:
    x1, _y1, x2, y2 = [int(v) for v in box]
    roi = image[y2 - 2:y2 + pad, max(0, x1 - 6):x2 + 6]
    if roi.size == 0:
        return []
    return find_bars(roi, min_width=20)


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    image = cv2.imread(str(FRAME))
    result = model.predict(image, imgsz=960, conf=0.25, iou=0.8, verbose=False)[0]

    print(f"{'#':>2} {'cls':<6} {'位置':<18} 头顶条      脚下条")
    for index, box in enumerate(result.boxes or [], 1):
        coords = [float(v) for v in box.xyxy[0].tolist()]
        above = bars_above(image, coords)
        below = bars_below(image, coords)
        name = result.names[int(box.cls[0])]
        print(f"{index:2d} {name:<6} ({int(coords[0]):4d},{int(coords[1]):4d})-({int(coords[2]):4d},{int(coords[3]):4d})  "
              f"{[b.kind for b in above]!s:<12} {[b.kind for b in below]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
