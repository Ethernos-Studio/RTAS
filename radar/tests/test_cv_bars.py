# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""血条 CV：颜色 + 几何找条（不依赖模型）。"""

from __future__ import annotations

import cv2
import numpy as np

from eyesofpriestess.cv.bars import draw, find_bars, summarize


def _canvas(width: int = 320, height: int = 180) -> np.ndarray:
    """一块中灰底图（模拟被半透明黑底压住的画面）。"""
    return np.full((height, width, 3), (90, 90, 90), dtype=np.uint8)


def _draw_bar(image: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
    """画一根条：半透明黑底 + 纯色填充（和游戏里一致，没有白色填充）。

    黑底要给到 3px：真实 trough 就是比填充宽一圈，这也是 find_bars 判断
    "这是条，不是地板花纹"的依据。
    """
    cv2.rectangle(image, (x - 3, y - 3), (x + w + 2, y + h + 2), (25, 25, 25), -1)   # 黑底
    cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)                         # 填充


def test_finds_colored_bars():
    image = _canvas()
    _draw_bar(image, 40, 60, 60, 5, (40, 40, 210))     # 红（血条·敌）
    _draw_bar(image, 140, 60, 55, 5, (220, 120, 40))   # 蓝（血条·友）
    _draw_bar(image, 60, 100, 50, 3, (40, 200, 60))    # 绿（技能条）

    bars = find_bars(image)
    kinds = summarize(bars)

    assert kinds.get("red") == 1, kinds
    assert kinds.get("blue") == 1, kinds
    assert kinds.get("green") == 1, kinds
    for bar in bars:
        assert bar.width >= 40 and bar.height <= 6


def test_ignores_square_blobs_and_background():
    """方的色块、大片背景都不是条。"""
    image = _canvas()
    cv2.rectangle(image, (40, 40), (80, 80), (40, 40, 210), -1)      # 40x40 方块
    cv2.rectangle(image, (150, 30), (300, 120), (220, 120, 40), -1)  # 大块蓝色

    assert find_bars(image) == []


def test_ignores_bright_background_stripes():
    """亮底上的细条纹不算血条（条必须压在暗底上）。"""
    image = np.full((180, 320, 3), (240, 240, 240), dtype=np.uint8)
    cv2.rectangle(image, (40, 60), (110, 65), (40, 40, 210), -1)

    assert find_bars(image) == []


def test_draw_marks_bars():
    image = _canvas()
    _draw_bar(image, 40, 60, 60, 5, (40, 40, 210))
    bars = find_bars(image)

    marked = draw(image, bars)

    assert marked.shape == image.shape
    assert np.any(marked != image)      # 画上去了


def test_empty_image_is_safe():
    assert find_bars(np.zeros((0, 0, 3), dtype=np.uint8)) == []
