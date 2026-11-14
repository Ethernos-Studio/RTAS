# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机模型：用资料里给出的验证点锁死投影实现。"""

from __future__ import annotations

import pytest

from eyesofpriestess.cv.camera import BattleCamera


@pytest.fixture()
def camera() -> BattleCamera:
    """资料验证时用的是 2772x1280（台服手机分辨率）。"""
    return BattleCamera(screen_w=2772, screen_h=1280)


def test_project_center_lands_near_screen_center(camera):
    """世界原点必须落在屏幕中心附近（x 精确居中，y 偏上一点）。"""
    x, y, depth = camera.project((0.0, 0.0, 0.0))

    assert x == pytest.approx(1386.0, abs=1.0)
    assert y == pytest.approx(569.8, abs=2.0)
    assert depth > 0


def test_one_world_unit_horizontal(camera):
    """1 个世界单位 ≈ 185px（2772 宽屏下）。"""
    x, _y, _ = camera.project((1.0, 0.0, 0.0))

    assert x - 1386.0 == pytest.approx(185.0, abs=2.0)


def test_one_world_unit_vertical_is_up(camera):
    """世界 y 向上，屏幕上 y 变小。"""
    _x, y, _ = camera.project((0.0, 1.0, 0.0))

    assert 569.8 - y == pytest.approx(148.7, abs=3.0)


def test_matches_documented_validation_points(camera):
    """作者给的验证点：一个几乎精确，一个落在"脚底/插值"容差内。"""
    x1, y1 = camera.project((0.27, 0.48, 0.0))[:2]
    assert x1 == pytest.approx(1435, abs=3)
    assert y1 == pytest.approx(495, abs=5)

    x2, y2 = camera.project((1.8, 0.012, 0.0))[:2]
    assert abs(x2 - 1732) < 25
    assert abs(y2 - 583) < 25


def test_project_map_uses_per_stage_offset(camera):
    """地图->世界：world_x = a - map_x、world_z = map_y - b。"""
    x, y = camera.project_map(1.2, 3.5, (1.2, 3.5))

    assert x == pytest.approx(1386.0, abs=1.0)


def test_fit_offset_recovers_known_offset(camera):
    """给几组对应点，拟合应能还原出 (a, b)。"""
    truth = (2.5, 4.0)
    correspondences = []
    for map_x, map_y in ((0, 0), (3, 0), (0, 3), (3, 3)):
        x, y = camera.project_map(map_x, map_y, truth)
        correspondences.append((map_x, map_y, x, y))

    a, b = camera.fit_offset(correspondences, guess=(0.0, 0.0))

    assert a == pytest.approx(truth[0], abs=0.2)
    assert b == pytest.approx(truth[1], abs=0.2)


def test_from_pose_matches_default_when_using_same_parameters(camera):
    """from_pose 用同样的 fov/俯仰/高度/偏移构造，应该和默认矩阵一致。"""
    posed = BattleCamera.from_pose(screen_w=2772, screen_h=1280, fov_deg=40, pitch_deg=30,
                                   height=9.5028, y_offset=0.3794)
    x, y, _ = posed.project((1.0, 0.0, 0.0))

    assert x == pytest.approx(1571.0, abs=6.0)
    assert y == pytest.approx(569.8, abs=6.0)
