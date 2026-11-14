"""用血条颜色区分敌我：干员血条是绿/蓝，敌人是红。

用户说得对——干员唯一稳定的共性是"脚下圆环 + 血条"，400 多种干员光看长相没法覆盖。
那就别靠长相：**检测框正上方的血条颜色**才是稳定信号，
W、维什戴尔、塔露拉、支援机械全都吃这一条（她们的血条也是队友色）。

实现：取框上方一条窄带，找横向细长、颜色饱和的区域，按色相分类：
  red   -> 敌人（保留）
  green -> 干员/友方（丢弃）
  blue  -> 干员/友方（丢弃，部分干员是蓝条）
  None  -> 看不见血条（保留，宁可多留也别漏）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")


def bar_color_at(image: np.ndarray, box, *, pad: int = 6, band: int = 26) -> str | None:
    """看框正上方还有没有血条，返回 'red' / 'green' / 'blue' / None。"""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)

    top = max(0, y1 - band)
    bottom = min(height, y1 + pad)
    left = max(0, x1 - 4)
    right = min(width, x2 + 4)
    if bottom - top < 3 or right - left < 8:
        return None

    patch = image[top:bottom, left:right]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    masks = {
        "red": (((hue <= 25) | (hue >= 170)) & (sat > 60) & (val > 60)),
        "green": ((hue >= 35) & (hue <= 89) & (sat > 60) & (val > 60)),
        "blue": ((hue >= 90) & (hue <= 140) & (sat > 60) & (val > 60)),
    }

    best_name, best_score = None, 0.0
    for name, mask in masks.items():
        if not mask.any():
            continue
        rows = mask.sum(axis=1)
        # 血条的特征：某一行里有一整段连续的同色像素（横条），不是零散噪点
        row_max = int(rows.max())
        ratio = row_max / max(1, patch.shape[1])
        score = ratio * (row_max / max(1, patch.shape[0]))
        if ratio > 0.35 and score > best_score:
            best_name, best_score = name, score
    return best_name


def keep_as_enemy(image: np.ndarray, box) -> tuple[bool, str | None]:
    """返回 (是否算敌人, 血条颜色)。绿/蓝条 = 干员/友方 -> 丢弃。"""
    color = bar_color_at(image, box)
    if color in ("green", "blue"):
        return False, color
    return True, color
