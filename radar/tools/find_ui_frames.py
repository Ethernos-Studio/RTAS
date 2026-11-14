"""找出"关卡标题卡 / 编队预览"这类不该标的帧（给标注前剔除用）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def stats(image: np.ndarray) -> tuple[float, float]:
    """返回 (中心高亮文字占比, 全图亮度)。"""
    height, width = image.shape[:2]
    center = image[int(height * 0.28):int(height * 0.72), int(width * 0.25):int(width * 0.75)]
    gray = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
    white = float((gray > 225).mean())          # 标题文字是接近纯白的大字
    return white, float(gray.mean())


def main() -> int:
    video = sys.argv[1] if len(sys.argv) > 1 else "v000_1-10"
    frames = sorted((ROOT / "frames").glob(f"{video}_*.jpg"))
    print(f"{video}: {len(frames)} 帧")
    suspicious = []
    for path in frames:
        image = cv2.imread(str(path))
        if image is None:
            continue
        white, brightness = stats(image)
        if white > 0.012:                      # 标题卡中心的白色文字面积远大于战斗画面
            suspicious.append((path.name, white, brightness))
    for name, white, brightness in suspicious:
        print(f"  可疑 UI 帧: {name}  白字占比 {white:.3f}  亮度 {brightness:.0f}")
    print(f"共 {len(suspicious)} 帧建议 Skip（在 LS 里点 Skip/cancel，不要标点）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
