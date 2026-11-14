# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""相机位姿拟合：不信任任何预置参数，从对应点自己解。"""

from __future__ import annotations

import pytest

from eyesofpriestess.cv.camera import BattleCamera


def _correspondences(camera: BattleCamera, offset: tuple[float, float]):
    points = []
    for col in (0, 2, 4, 6):
        for row in (0, 2, 4):
            projected = camera.project_map(col, row, offset)
            if projected is not None:
                points.append((float(col), float(row), projected[0], projected[1]))
    return points


def test_fit_pose_recovers_parameters():
    """给一组"真值相机"生成的对应点，拟合应能还原参数并保持低残差。"""
    truth = dict(fov_deg=50.0, pitch_deg=35.0, height=11.0, roll_deg=3.0)
    offset = (2.0, 4.0)
    camera = BattleCamera.from_pose(screen_w=1920, screen_h=1080, **truth)
    points = _correspondences(camera, offset)

    fitted, report = BattleCamera.fit_pose(points, screen_w=1920, screen_h=1080)

    assert report["residual_mean_px"] < 2.0
    assert report["params"]["fov"] == pytest.approx(50.0, abs=3.0)
    assert report["params"]["pitch"] == pytest.approx(35.0, abs=3.0)
    assert report["params"]["height"] == pytest.approx(11.0, abs=1.5)
    assert report["params"]["a"] == pytest.approx(2.0, abs=0.5)
    assert report["params"]["b"] == pytest.approx(4.0, abs=0.5)


def test_fit_pose_requires_six_points():
    with pytest.raises(ValueError, match="至少需要 6 组"):
        BattleCamera.fit_pose([(0, 0, 100, 100), (1, 0, 200, 100)])


def test_from_pose_roll_changes_projection():
    """横滚不为 0 时投影应水平倾斜（Blender 里对不上时先查这个）。"""
    flat = BattleCamera.from_pose(screen_w=1920, screen_h=1080, fov_deg=45, pitch_deg=30,
                                  height=9, roll_deg=0.0)
    rolled = BattleCamera.from_pose(screen_w=1920, screen_h=1080, fov_deg=45, pitch_deg=30,
                                    height=9, roll_deg=5.0)

    x_flat, y_flat, _ = flat.project((1.0, 0.0, 1.0))
    x_rolled, y_rolled, _ = rolled.project((1.0, 0.0, 1.0))

    assert x_flat != pytest.approx(x_rolled, abs=1.0)
    assert y_flat != pytest.approx(y_rolled, abs=1.0)
