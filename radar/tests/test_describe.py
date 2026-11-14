# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""场景图 -> VLM 可读文字（带坐标）。"""

from __future__ import annotations

from eyesofpriestess.describe import Describer, describe, zone_of


def _unit(track: int, x: int, y: int, w: int = 40, h: int = 60, code: str | None = "B2") -> dict:
    return {"track_id": track, "cls": "enemy", "conf": 0.9,
            "box": [x, y, w, h], "code": code, "code_score": 0.9}


def test_zone_of_corners_and_center():
    assert zone_of(10, 10, 1920, 1080) == "左上"
    assert zone_of(960, 540, 1920, 1080) == "中央"
    assert zone_of(1900, 1070, 1920, 1080) == "右下"


def test_summary_counts_codes_and_zones():
    describer = Describer(width=1920, height=1080)
    summary = describer.summarize([_unit(1, 10, 10), _unit(2, 100, 20, code="B1")])

    assert summary["enemies"] == 2
    assert summary["codes"] == {"B2": 1, "B1": 1}
    assert summary["zones"] == {"左上": 2}
    assert "敌人 2" in describer.line(3.0, summary)


def test_unit_lines_carry_coordinates():
    """VLM 要的是"图 + 带坐标的文字"，所以每个单位都要有中心坐标和尺寸。"""
    describer = Describer(width=1920, height=1080)
    lines = describer.unit_lines([_unit(7, 900, 700)])

    assert len(lines) == 1
    assert "center=(920,730)" in lines[0]
    assert "size=40x60" in lines[0]
    assert "#7" in lines[0]
    assert "B2" in lines[0]


def test_describe_smooths_per_second_buckets():
    """逐帧计数会抖，按秒聚合成一条（取中位数那一帧）。"""
    records = [
        {"t": 0.0, "frame_idx": 1, "units": [_unit(1, 10, 10)]},
        {"t": 0.3, "frame_idx": 2, "units": [_unit(1, 10, 10), _unit(2, 50, 10)]},
        {"t": 0.6, "frame_idx": 3, "units": [_unit(1, 12, 10)]},
        {"t": 1.2, "frame_idx": 4, "units": []},
    ]

    lines = describe(records, every=1.0)

    headers = [line for line in lines if line.startswith("t=")]
    assert len(headers) == 2
    assert "敌人 1" in headers[0]
    assert "没有敌人" in headers[1]
    assert any("center=" in line for line in lines)


def test_unknown_code_shows_label():
    describer = Describer(width=1920, height=1080)
    lines = describer.unit_lines([_unit(3, 100, 100, code=None)])

    assert "未识别" in lines[0]
