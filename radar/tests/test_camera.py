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


def test_project_center_matches_documented_value(camera):
    """world(0,0,0) 必须落在屏幕中心偏上（资料实测 1386, 585）。"""
    x, y, depth = camera.project((0.0, 0.0, 0.0))

    assert x == pytest.approx(1386.0, abs=0.5)
    assert y == pytest.approx(585.0, abs=0.5)
    assert depth > 0


def test_one_world_unit_horizontal(camera):
    """1 个世界单位 ≈ 193px（2772 宽屏下）。"""
    x, y, _ = camera.project((1.0, 0.0, 0.0))

    assert x - 1386.0 == pytest.approx(192.7, abs=1.0)
    assert y == pytest.approx(585.0, abs=0.5)


def test_one_world_unit_vertical_is_up(camera):
    """世界 y 向上，屏幕上 y 变小。"""
    x, y, _ = camera.project((0.0, 1.0, 0.0))

    assert x == pytest.approx(1386.0, abs=0.5)
    assert 585.0 - y == pytest.approx(155.4, abs=1.5)


def test_project_map_uses_per_stage_offset(camera):
    """地图->世界：world_x = a - map_x、world_z = map_y - b。"""
    # map 坐标 (1.2, 3.5) + offset (1.2, 3.5) -> 世界原点 -> 屏幕中心
    x, y = camera.project_map(1.2, 3.5, (1.2, 3.5))

    assert x == pytest.approx(1386.0, abs=0.5)
    assert y == pytest.approx(585.0, abs=0.5)


def test_fit_offset_recovers_known_offset(camera):
    """给几组对应点，拟合应能还原出 (a, b)。"""
    truth = (2.5, 4.0)
    correspondences = []
    for map_x, map_y in ((0, 0), (3, 0), (0, 3), (3, 3)):
        x, y = camera.project_map(map_x, map_y, truth)
        correspondences.append((map_x, map_y, x, y))

    a, b = camera.fit_offset(correspondences, guess=(0.0, 0.0))

    assert a == pytest.approx(truth[0], abs=0.15)
    assert b == pytest.approx(truth[1], abs=0.15)


def test_from_pose_matches_default_when_using_same_parameters(camera):
    """from_pose 用资料参数构造出来，应该和默认矩阵基本一致。"""
    posed = BattleCamera.from_pose(screen_w=2772, screen_h=1280, fov_deg=40, pitch_deg=30, height=9.125)
    x, y, _ = posed.project((1.0, 0.0, 0.0))

    assert x == pytest.approx(1578.7, abs=6.0)
    assert y == pytest.approx(585.0, abs=6.0)
