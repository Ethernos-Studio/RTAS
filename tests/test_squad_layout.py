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


def _slot(filled: bool) -> np.ndarray:
    """造一张编队格：空位是斜纹 + 「+」，真卡是等级圆圈 + 发白名称条。"""
    card = np.full((248, 113, 3), (40, 50, 62), dtype=np.uint8)
    for offset in range(-248, 113, 18):
        cv2.line(card, (offset, 0), (offset + 248, 248), (70, 78, 90), 2)
    if not filled:
        cv2.line(card, (46, 150), (66, 150), (150, 155, 160), 4)
        cv2.line(card, (56, 140), (56, 160), (150, 155, 160), 4)
        return card
    cv2.circle(card, (22, 196), 16, (240, 240, 240), 3)
    cv2.rectangle(card, (10, 220), (70, 236), (245, 245, 245), -1)
    return card


def test_empty_slot_with_stripes_is_not_an_operator():
    """实机截图里空位有斜纹和「+」，边缘密度够高，必须靠等级圈/名称条挡掉。"""
    recognizer = SquadRecognizer(SquadConfig())
    assert recognizer._is_valid_card(_slot(False)) is False
    assert recognizer._is_valid_card(_slot(True)) is True


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


def _blank_card() -> np.ndarray:
    return np.full((248, 113, 3), 40, dtype=np.uint8)


def _solid_art_card() -> np.ndarray:
    """卡面亮部：一整块实心白，凸包占比接近 1。"""
    card = _blank_card()
    card[150:185, 8:45] = 255
    return card


def _crossed_blades(second_layer: bool = False) -> np.ndarray:
    """画一个「交叉双刀」：两条细白线交叉，可选再叠一层（E2 是错位重叠）。"""
    card = _blank_card()
    dy = 27
    cv2.line(card, (10, 146), (44, 146 + dy), (255, 255, 255), 2)
    cv2.line(card, (44, 146), (10, 146 + dy), (255, 255, 255), 2)
    if second_layer:
        cv2.line(card, (10, 154), (44, 154 + dy), (255, 255, 255), 2)
        cv2.line(card, (44, 154), (10, 154 + dy), (255, 255, 255), 2)
    return card


def test_badge_blob_separates_blades_from_bright_art():
    """交叉双刀「瘦」（凸包占比低），卡面亮部「实心」（接近 1）。"""
    recognizer = SquadRecognizer(SquadConfig())

    assert recognizer._badge_blob(_blank_card()) is None

    _, solid = recognizer._badge_blob(_solid_art_card())
    assert solid >= recognizer.ELITE_BLOB_CONVEXITY

    area, convexity = recognizer._badge_blob(_crossed_blades())
    assert area >= recognizer.ELITE_BLOB_AREA
    assert convexity < recognizer.ELITE_BLOB_CONVEXITY


def test_second_layer_makes_the_blob_bigger():
    """E2 多一层刀：面积更大、凸包更高，但依然是「瘦」的。"""
    recognizer = SquadRecognizer(SquadConfig())
    single_area, single_convexity = recognizer._badge_blob(_crossed_blades())
    double_area, double_convexity = recognizer._badge_blob(_crossed_blades(second_layer=True))

    assert double_area > single_area
    assert double_convexity < recognizer.ELITE_BLOB_CONVEXITY


def test_recognize_elite_returns_e0_without_blades():
    """没有刀（空卡 / 实心白立绘）一律 E0。"""
    recognizer = SquadRecognizer(SquadConfig())
    assert recognizer._recognize_elite_level(_blank_card()) is EliteLevel.E0
    assert recognizer._recognize_elite_level(_solid_art_card()) is EliteLevel.E0


def test_ncc_shape_similarity():
    """形状相关性：相同为 1，反相为 -1，常量块无法比较记 0。"""
    recognizer = SquadRecognizer(SquadConfig())
    a = np.zeros((10, 10), dtype=np.float32)
    a[2:8, 2:8] = 1.0

    assert recognizer._ncc(a, a) == pytest.approx(1.0)
    assert recognizer._ncc(a, 1.0 - a) == pytest.approx(-1.0)
    assert recognizer._ncc(a, np.full((10, 10), 0.5, dtype=np.float32)) == 0.0


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
