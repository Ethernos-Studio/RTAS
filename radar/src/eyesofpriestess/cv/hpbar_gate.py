"""干员挡住敌人时，敌人的血条和干员的蓝条会挨在一起 —— 门控要保守。

规则改成：
1. 在框上方窗口里同时找两种条；
2. **只要找到敌方橙条就保留**（哪怕同时有蓝条 —— 说明有敌人被挡在干员那一格里）；
3. 只有"只看到蓝条、没看到橙条"时才判为干员/友方丢丢弃；
4. 什么都找不到 -> 保留（宁可多留，不能漏）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def find_hp_bars(image: np.ndarray, box, *, above: int = 56, below: int = 14) -> dict[str, int]:
    """返回 {归属: y} —— 该框附近找到的血条（同种取离框最近的一条）。"""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, x1), min(width, x2)
    if x2 - x1 < 12:
        return {}
    box_width = x2 - x1
    top = max(0, y1 - above)
    bottom = min(height, y2 + below)
    if bottom - top < 6:
        return {}

    strip = image[top:bottom, x1:x2]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    colorful = (sat > 70) & (val > 90)
    masks = {
        "ally": colorful & (hue >= 92) & (hue <= 118),      # 友方浅蓝
        "enemy": colorful & (hue >= 4) & (hue <= 25),      # 敌方橙红
    }

    found: dict[str, int] = {}
    min_run = int(box_width * 0.45)
    for kind, mask in masks.items():
        best_distance = 10 ** 9
        for row in range(strip.shape[0]):
            line = mask[row]
            if line.sum() < min_run:
                continue
            run = longest = 0
            for value in line:
                run = run + 1 if value else 0
                longest = max(longest, run)
            if longest < min_run:
                continue
            y = top + row
            # 血条通常就在框顶上方 2~12px；离框越近越可能是这个目标的
            distance = abs(y1 - y)
            if distance < best_distance:
                best_distance = distance
                found[kind] = y
    return found


def keep_as_enemy(image: np.ndarray, box) -> tuple[bool, str | None]:
    bars = find_hp_bars(image, box)
    if "enemy" in bars:
        # 看到敌方血条就保留：哪怕同一格里有干员的蓝条（敌人被阻挡就是这种情况）
        return True, "enemy"
    if "ally" in bars:
        return False, "ally"
    return True, None
