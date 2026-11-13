# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""导航按钮判定测试（模板匹配，不需要 OCR）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rtas.vision.nav_detector import (
    DEFAULT_TEMPLATE,
    NavMenuButtonDetector,
    NavigationState,
    detect_navigation,
)


def _canvas_with_home(scale: float = 0.9, top_left: tuple[int, int] = (200, 25)):
    """在深色画布左上角贴入导航按钮模板。"""
    canvas = np.full((759, 1264, 3), 38, dtype=np.uint8)
    template = cv2.imread(str(DEFAULT_TEMPLATE), cv2.IMREAD_UNCHANGED)
    assert template is not None
    if template.shape[2] == 4:
        template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
    resized = cv2.resize(
        template,
        (int(template.shape[1] * scale), int(template.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    x, y = top_left
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    expected = (x + resized.shape[1] // 2, y + resized.shape[0] // 2)
    return canvas, expected


def test_template_asset_exists():
    assert DEFAULT_TEMPLATE.is_file()


def test_detects_home_button_in_top_left():
    image, expected = _canvas_with_home()
    state, match = detect_navigation(image)

    assert state is NavigationState.SUB_MENU
    assert match.found
    assert match.confidence >= 0.85
    assert match.center is not None
    assert abs(match.center[0] - expected[0]) <= 3
    assert abs(match.center[1] - expected[1]) <= 3
    assert match.as_dict()["state"] == "sub_menu"


def test_missing_home_button_means_main_menu():
    canvas = np.full((759, 1264, 3), 38, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (500, 90), (60, 60, 60), -1)

    detector = NavMenuButtonDetector()
    state, match = detector.detect_state(canvas)

    assert not match.found
    assert state is NavigationState.MAIN_MENU
    assert match.confidence < 0.85


def test_threshold_controls_detection():
    image, _ = _canvas_with_home()
    strict = NavMenuButtonDetector(threshold=0.999)
    assert not strict.detect(image).found

    loose = NavMenuButtonDetector(threshold=0.5)
    assert loose.detect(image).found


def test_detector_visualize_returns_copy():
    image, _ = _canvas_with_home()
    detector = NavMenuButtonDetector()
    match = detector.detect(image)
    vis = detector.visualize(image, match)

    assert vis.shape == image.shape
    assert not np.array_equal(vis, image)


def test_detector_handles_empty_image():
    detector = NavMenuButtonDetector()
    match = detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not match.found
