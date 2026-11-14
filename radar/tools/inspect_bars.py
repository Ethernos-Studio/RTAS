"""诊断：真实血条/技能条到底是什么颜色和尺寸（别靠猜阈值）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

FRAME = ROOT / "frames" / "..\\out\\_battle_frame.jpg"


def main() -> int:
    image = cv2.imread(str(FRAME))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    print(f"帧 {FRAME.name}  {image.shape[1]}x{image.shape[0]}")

    # 宽松的"彩色细条"候选：任何明显彩色、比较亮的像素
    mask = ((sat > 60) & (val > 60)).astype(np.uint8)
    # 只保留横向细长条：先横向闭运算，再按尺寸筛
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)

    rows = []
    for index in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[index])
        if w < 12 or h < 2 or h > 14 or w / max(h, 1) < 3:
            continue
        patch = hsv[y:y + h, x:x + w]
        mean_h = float(np.median(patch[:, :, 0]))
        mean_s = float(patch[:, :, 1].mean())
        mean_v = float(patch[:, :, 2].mean())
        # 条上方/下方是不是暗底（半透明黑）
        above = val[max(0, y - 3):y, x:x + w]
        below = val[y + h:min(val.shape[0], y + h + 3), x:x + w]
        rows.append((w, h, x, y, mean_h, mean_s, mean_v,
                     float(above.mean()) if above.size else -1,
                     float(below.mean()) if below.size else -1))

    rows.sort(key=lambda item: -item[0])
    print(f"横向细条候选: {len(rows)} 个（宽, 高, x, y, 中位色相, 饱和度, 亮度, 上方亮度, 下方亮度）")
    for row in rows[:24]:
        print("  " + "  ".join(f"{value:7.1f}" if isinstance(value, float) else f"{value:5d}" for value in row))

    # 把候选画出来
    vis = image.copy()
    for w, h, x, y, *_ in rows:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imwrite(str(ROOT / "out" / "_bars_candidates.jpg"), cv2.resize(vis, (1280, 720)))
    print("wrote out/_bars_candidates.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
