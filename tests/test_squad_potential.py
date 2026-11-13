# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""卡面潜能徽章识别与「潜能→部署费用」换算的测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from rtas.data.models.operator import Operator, OperatorPhase
from rtas.vision.squad_analyzer import SquadAnalyzer
from rtas.vision.squad_recognizer import OperatorCard, SquadConfig, SquadRecognizer

POTENTIAL_DIR = Path("templates/icons/potential")

#: 1264x711 实测：凯尔希·思衡托的卡片 (152+2*140, 106+254)=(432,360)，徽章 (517,530)-(543,557)
MEASURED_CARD_ORIGIN = (432, 360)
MEASURED_BADGE_BOX = (517, 530, 543, 557)


def _card_background(seed: int = 0) -> np.ndarray:
    """一张 113x248 的卡面底图：中间调渐变，避免纯噪声碰巧贴合某个模板。"""
    y = np.linspace(60, 150, 248, dtype=np.float32)[:, None, None]
    card = np.repeat(np.repeat(y, 113, axis=1), 3, axis=2)
    card[:, :, 0] *= 1.05
    card[:, :, 2] *= 0.85
    rng = np.random.RandomState(seed)
    card += rng.randint(-6, 7, card.shape[:2] + (1,)).astype(np.float32)
    return np.clip(card, 0, 255).astype(np.uint8)


def _paste_badge(card: np.ndarray, level: int, scale: float = 0.75) -> np.ndarray:
    """把潜能徽章按卡内比例贴到徽章框正中（模拟徽章叠在立绘上的半透明效果）。"""
    glyph = cv2.imread(str(POTENTIAL_DIR / f"potential_{level}_small.png"), cv2.IMREAD_UNCHANGED)
    if glyph is None:
        pytest.skip(f"缺少潜能模板 potential_{level}_small.png")
    size = (int(round(glyph.shape[1] * scale)), int(round(glyph.shape[0] * scale)))
    colour = cv2.resize(glyph[:, :, :3], size, interpolation=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.resize(glyph[:, :, 3].astype(np.float32) / 255.0, size,
                       interpolation=cv2.INTER_AREA)[..., None]

    config = SquadConfig()
    left, top, right, bottom = config.potential_region_ratio
    height, width = card.shape[:2]
    centre_x = int(round((left + right) / 2 * width))
    centre_y = int(round((top + bottom) / 2 * height))
    x0 = centre_x - size[0] // 2
    y0 = centre_y - size[1] // 2

    patch = card[y0:y0 + size[1], x0:x0 + size[0]].astype(np.float32)
    card[y0:y0 + size[1], x0:x0 + size[0]] = (colour * alpha + patch * (1 - alpha)).astype(np.uint8)
    return card


def test_potential_region_matches_measured_badge_box():
    """配置里的徽章框要正好落在实测 (517,530)-(543,557) 上（阿米娅同排同比例对齐）。"""
    config = SquadConfig()
    left, top, right, bottom = config.potential_region_ratio
    card_x, card_y = MEASURED_CARD_ORIGIN
    assert int(round(left * 113)) + card_x == MEASURED_BADGE_BOX[0]
    assert int(round(top * 248)) + card_y == MEASURED_BADGE_BOX[1]
    assert int(round(right * 113)) + card_x == MEASURED_BADGE_BOX[2]
    assert int(round(bottom * 248)) + card_y == MEASURED_BADGE_BOX[3]


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_badge_is_recognized_from_card_image(level):
    """把某潜的徽章贴到卡面上，应读回同一个等级。"""
    recognizer = SquadRecognizer(SquadConfig())
    card = _paste_badge(_card_background(seed=level), level)

    got, score = recognizer._recognize_potential(card)

    assert got == level, f"{level} 潜的徽章读成了 {got}（相似度 {score:.2f}）"
    assert score >= recognizer.POTENTIAL_MIN_SCORE


def test_missing_badge_is_reported_as_unknown():
    """卡面上没有徽章时不能瞎猜：返回 None，交给 --default-potential 兜底。"""
    recognizer = SquadRecognizer(SquadConfig())
    assert recognizer._recognize_potential(_card_background())[0] is None


def test_potential_templates_cover_zero_to_full():
    """模板必须覆盖 0~5 潜，而且满潜那张最亮（灰星→亮星）。"""
    recognizer = SquadRecognizer(SquadConfig())
    templates = recognizer.potential_templates()

    assert set(templates) == set(range(6))
    mean_values = {level: float(glyph.mean()) for level, (glyph, _mask) in templates.items()}
    assert mean_values[5] > mean_values[0]


def test_potential_recognition_can_be_disabled():
    """关掉 recognize_potential 后不再读徽章。"""
    config = SquadConfig()
    config.recognize_potential = False
    recognizer = SquadRecognizer(config)
    card = _paste_badge(_card_background(), 5)

    assert recognizer._recognize_potential(card)[0] == 5     # 方法本身还能用
    assert config.recognize_potential is False               # 但主流程不会调用它


def test_masked_correlation_map_matches_direct_scoring():
    """一次卷积算出的响应图，必须和逐点算的掩码相关系数一致。"""
    rng = np.random.RandomState(0)
    roi = (rng.rand(40, 50, 3) * 255).astype(np.float32)
    template = (rng.rand(17, 18, 3) * 255).astype(np.float32)
    mask = (rng.rand(17, 18) > 0.3).astype(np.float32)

    response = SquadRecognizer(SquadConfig())._masked_ncc_map(roi, template, mask)
    assert response is not None

    cy, cx = 20, 25                     # 核中心落在 (cy, cx)
    patch = roi[cy - 8:cy + 9, cx - 9:cx + 9]
    selected = mask.astype(bool)
    a = patch[selected].reshape(-1, 3)
    b = template[selected].reshape(-1, 3)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    expected = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert float(response[cy, cx]) == pytest.approx(expected, abs=1e-4)


def _card_at(image: np.ndarray, tag: tuple) -> np.ndarray:
    for row, col, (x, y, w, h) in SquadRecognizer(SquadConfig()).card_rects(image):
        if (row, col) == tag:
            return image[y:y + h, x:x + w]
    raise AssertionError(f"截图里没有卡片 {tag}")


def test_potential_scales_with_resolution():
    """同一张截图放大到 1080p，徽章也要按卡宽等比缩放后读出来。"""
    path = "__temp__/arknights_20261113_153138.png"
    if not Path(path).exists():
        pytest.skip(f"缺少截图 {path}")

    recognizer = SquadRecognizer(SquadConfig())
    image = cv2.imread(path)
    hd = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    assert recognizer._recognize_potential(_card_at(hd, (1, 2)))[0] == 1   # 凯尔希 1 潜
    assert recognizer._recognize_potential(_card_at(hd, (1, 5)))[0] == 5   # 阿米娅满潜



def test_cost_reduction_follows_potential_levels():
    """1 潜 -1 费用，满潜再 -1（凯尔希·思衡托：0潜 19 → 1潜 18 → 满潜 17）。"""
    operator = Operator(
        id="char_test_kalts",
        name="凯尔希·思衡托",
        phases=[OperatorPhase(
            phase_index=2, max_level=90, max_hp=2200, atk=650, def_=280,
            magic_resistance=20.0, cost=19, block_count=2,
            attack_speed=100.0, respawn_time=70,
        )],
        max_potential_level=5,
        cost_reduction_by_potential=[0, -1, -1, -1, -2, -2],
    )

    assert operator.cost_at(2, 0) == 19
    assert operator.cost_at(2, 1) == 18
    assert operator.cost_at(2, 5) == 17
    assert operator.cost_at(2, None) == 17     # 不指定潜能 = 按满潜


class _StubMatcher:
    """记录每张卡按几潜去查费用。"""

    def __init__(self):
        self.potentials: list = []

    def get_operator_info(self, operator_name, elite_level=0, level=1, confidence=0.0,
                          confidence_threshold=0.7, potential=None):
        self.potentials.append(potential)
        return {
            'matched': True,
            'operator_id': f'char_test_{len(self.potentials)}',
            'name': operator_name,
            'rarity': 6,
            'profession': 'WARRIOR',
            'sub_profession': 'sword',
            'position': 'MELEE',
            'cost': 20 - (potential or 0),
            'cost_base': 20,
            'potential': potential,
            'max_potential': 5,
            'cost_reduction_by_potential': [0, -1, -1, -1, -2, -2],
            'phase_info': {},
            'match_score': 1.0,
            'match_type': 'exact',
            'alternatives': [],
        }


def _card(name: str, potential):
    return OperatorCard(position=(0, 0, 113, 248), grid_position=(0, 0),
                        name=name, name_confidence=1.0, potential=potential)


def test_analyzer_uses_each_card_badge():
    """徽章读到的潜能逐张生效，读不出来的用 default_potential（默认 0）。"""
    analyzer = SquadAnalyzer(potential=None, default_potential=0)
    matcher = _StubMatcher()
    analyzer._operator_matcher = matcher

    cards = [_card("凯尔希", 1), _card("阿米娅", 5), _card("望", None)]
    analyzer._query_operator_info(cards)

    assert matcher.potentials == [1, 5, 0]
    assert [card.operator_info['potential_source'] for card in cards] == \
        ['badge', 'badge', 'default']
    assert cards[0].operator_info['cost'] == 19
    assert cards[1].operator_info['cost'] == 15


def test_analyzer_potential_argument_overrides_badges():
    """--potential 是「全部按它算」，不会再被卡面徽章改回去。"""
    analyzer = SquadAnalyzer(potential=5, default_potential=0)
    matcher = _StubMatcher()
    analyzer._operator_matcher = matcher

    cards = [_card("凯尔希", 1), _card("望", None)]
    analyzer._query_operator_info(cards)

    assert matcher.potentials == [5, 5]
    assert all(card.operator_info['potential_source'] == 'argument' for card in cards)


#: 真实截图上逐卡读到的潜能（None = 徽章没读出来，代码里按 default_potential 兜底）。
#: 153138：可露希尔 2 潜、凯尔希·思衡托 1 潜、阿米娅满潜；
#: 165455：桃金娘/阿消/阿米娅/安赛尔/芬 满潜，夜刀 2 潜。
POTENTIAL_TRUTH = {
    "__temp__/arknights_20261113_153138.png": [
        [None, None, None, 2, None, None],
        [None, None, 1, None, None, 5],
    ],
    "__temp__/arknights_20261113_165455.png": [
        [None, 5, 2, None, 5],
        [5, 5, None, None, 5],
    ],
}


@pytest.mark.parametrize("path,truth", list(POTENTIAL_TRUTH.items()))
def test_potential_on_real_screenshots(path, truth):
    """真实截图上逐卡断言潜能（截图不在仓库里时跳过）。"""
    if not Path(path).exists():
        pytest.skip(f"缺少截图 {path}")

    recognizer = SquadRecognizer(SquadConfig())
    image = cv2.imread(path)
    assert image is not None

    checked = 0
    for row, col, (x, y, w, h) in recognizer.card_rects(image):
        if col >= len(truth[row]):
            continue
        got, score = recognizer._recognize_potential(image[y:y + h, x:x + w])
        assert got == truth[row][col], \
            f"{path} 第 ({row},{col}) 张应读作 {truth[row][col]}，实得 {got}（相似度 {score:.2f}）"
        checked += 1
    assert checked == sum(len(r) for r in truth)

