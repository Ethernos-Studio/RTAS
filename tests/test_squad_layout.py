# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""编队识别布局与名称清洗测试（不依赖 EasyOCR / 截图）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from rtas.vision.squad_recognizer import EliteLevel, SquadConfig, SquadRecognizer


def test_card_grid_matches_measured_layout():
    """1264x711 实测：首卡 (152,106)，尺寸 113x248，列距 140、行距 254。"""
    image = np.zeros((711, 1264, 3), dtype=np.uint8)
    rects = SquadRecognizer(SquadConfig()).card_rects(image)

    assert len(rects) == 12
    first = rects[0]
    assert first[0] == 0 and first[1] == 0
    assert first[2] == (152, 106, 113, 248)

    # 第二列 / 第二行
    assert rects[1][2][0] == 292
    assert rects[6][2][1] == 360


def test_card_grid_scales_with_resolution():
    """同一套比例在 1920x1080 下应等比放大。"""
    small = SquadRecognizer(SquadConfig()).card_rects(np.zeros((711, 1264, 3), dtype=np.uint8))
    large = SquadRecognizer(SquadConfig()).card_rects(np.zeros((1080, 1920, 3), dtype=np.uint8))

    scale = 1920 / 1264
    assert large[0][2][0] == pytest.approx(small[0][2][0] * scale, abs=2)
    assert large[0][2][2] == pytest.approx(small[0][2][2] * scale, abs=2)
    assert len(large) == 12


def test_empty_slot_is_rejected():
    recognizer = SquadRecognizer(SquadConfig())
    blank = np.full((248, 113, 3), 30, dtype=np.uint8)
    assert recognizer._is_valid_card(blank) is False

    noisy = np.random.RandomState(0).randint(0, 255, (248, 113, 3), dtype=np.uint8)
    assert recognizer._is_valid_card(noisy) is True


def test_name_cleaning_and_noise_filter():
    recognizer = SquadRecognizer(SquadConfig())
    assert recognizer._clean_name("伊内丝 E2") == "伊内丝E2"
    assert recognizer._clean_name("  ") == ""
    # 图标文字 / 纯数字判为噪声
    assert recognizer._is_noise_name("ORIG") is True
    assert recognizer._is_noise_name("CRC") is True
    assert recognizer._is_noise_name("70") is True
    # 正常名称保留
    assert recognizer._is_noise_name("Mon3tr") is False
    assert recognizer._is_noise_name("维什戴尔") is False


TEMPLATE_DIR = Path("templates/icons/elitism")

#: 真实截图（仅本地存在时跑定位/阈值联调；仓库里不带大图）
SCREENSHOTS = {
    "__temp__/arknights_20261113_153138.png": [[2, 2, 2, 2, 0, 2], [2, 2, 2, 1, 2, 2]],
    "__temp__/arknights_20261113_165455.png": [[2, 1, 0, 1, 0], [2, 1, 0, 0, 0]],
}


def _badge_card(bright_upper: bool, bright_below: bool) -> np.ndarray:
    """造一张 113x248 的卡：徽章带里画「明暗相间」的上半 + 可选亮的下半。

    locate 被固定成 (scale=0.73, 偏移 0)，所以窗口位置可以精确推算：
    带原点 (2,134)，窗口宽 29，上半 22 行、下半 6 行。
    """
    card = np.full((248, 113, 3), 40, dtype=np.uint8)
    left, top, width = 2, 134, 29
    if bright_upper:
        chess = np.indices((22, width)).sum(axis=0) % 2 == 0   # 一半白一半黑，像交叉双刀
        card[top:top + 22, left:left + width][chess] = 255
    if bright_below:
        card[top + 22:top + 28, left:left + width] = 255
    return card


@pytest.fixture
def fixed_locator(monkeypatch):
    recognizer = SquadRecognizer(SquadConfig())
    monkeypatch.setattr(
        recognizer, "_locate_badge",
        lambda roi, templates, card_width: (0.73, (0, 0)))
    return recognizer


def test_elite_decision_branches(fixed_locator):
    """没有徽章 / 只有一层 / 多一层，以及纯白卡面不算徽章。"""
    recognizer = fixed_locator
    assert recognizer._recognize_elite_level(_badge_card(False, False)) is EliteLevel.E0
    assert recognizer._recognize_elite_level(_badge_card(True, False)) is EliteLevel.E1
    assert recognizer._recognize_elite_level(_badge_card(True, True)) is EliteLevel.E2

    blank = np.full((248, 113, 3), 235, dtype=np.uint8)   # 阿消/罗德岛那种白立绘
    assert recognizer._recognize_elite_level(blank) is EliteLevel.E0


def test_elite_templates_come_from_packed_icons():
    """模板必须是解包出来的 40x30 / 40x38 渲染图，且只保留实体像素。"""
    recognizer = SquadRecognizer(SquadConfig())
    templates = recognizer.elite_templates()
    assert set(templates) == {1, 2}
    assert templates[1].shape[1] == templates[2].shape[1] == 40
    assert templates[1].shape[0] < templates[2].shape[0]      # E2 多叠一份，更高
    assert templates[1].sum() > 0 and templates[2].sum() > templates[1].sum()


@pytest.mark.parametrize("path,truth", list(SCREENSHOTS.items()))
def test_elite_level_on_real_screenshots(path, truth):
    """真实截图上逐卡断言精英化（截图不在仓库里时跳过）。"""
    if not Path(path).exists():
        pytest.skip(f"缺少截图 {path}")

    recognizer = SquadRecognizer(SquadConfig())
    image = cv2.imread(path)
    assert image is not None
    checked = 0
    for row, col, (x, y, w, h) in recognizer.card_rects(image):
        if col >= len(truth[row]):
            continue
        got = recognizer._recognize_elite_level(image[y:y + h, x:x + w])
        assert got.value == truth[row][col], f"{path} 第 ({row},{col}) 张应为 E{truth[row][col]}，实得 {got}"
        checked += 1
    assert checked == sum(len(r) for r in truth)


def test_elite_region_covers_measured_box():
    """精英化区域（卡内比例）要覆盖实测 (152,245)~(200,290)，并多留出 E2 第二层刀。"""
    config = SquadConfig()
    x0, y0, w, h = config.elite_region_ratio
    top, bottom = y0 * 248, (y0 + h) * 248
    right = (x0 + w) * 113
    assert top <= 139 and bottom >= 190   # 245-106=139, 290-106=184，再含下沿
    assert right >= 48
