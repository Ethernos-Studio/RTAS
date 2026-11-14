"""可行性探测：能不能自动检出地块网格（黄白虚线格线）的交点。

如果能稳定拿到格点，就可以做"3D 相机拟合"：
已知世界坐标是 (i,j,0) 的正方形网格，解出单应 H = K[r1 r2 t]，
再分解出相机的俯仰/滚转——部署预览的镜头俯仰就是这个量的变化。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

FRAMES = ["v000_1-7_f000979.jpg", "v000_1-7_f001069.jpg", "v000_1-7_f001089.jpg"]


def grid_mask(image: np.ndarray) -> np.ndarray:
    """格线是亮黄/白的虚线：高亮度 + 低到中饱和。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    yellowish = (hue >= 15) & (hue <= 45) & (sat >= 40) & (val >= 170)
    whitish = (sat < 45) & (val >= 205)
    mask = (yellowish | whitish).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def main() -> int:
    for name in FRAMES:
        image = cv2.imread(str(ROOT / "frames" / name))
        if image is None:
            continue
        mask = grid_mask(image)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=40, minLineLength=60, maxLineGap=25)
        count = 0 if lines is None else len(lines.reshape(-1, 4))
        vis = image.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        out = ROOT / "out" / f"_grid_{name[-9:-4]}.jpg"
        cv2.imwrite(str(out), vis)
        print(f"{name}: 候选线段 {count} 条 -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
