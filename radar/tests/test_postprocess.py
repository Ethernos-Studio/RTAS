# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""检测后处理：合并框抑制 / 友军抑制。"""

from __future__ import annotations

import numpy as np

from eyesofpriestess.ally import AllyFilter
from eyesofpriestess.postprocess import MergeSuppressor, containment


def test_containment():
    assert containment([10, 10, 20, 20], [0, 0, 40, 40]) == 1.0
    assert containment([0, 0, 40, 40], [10, 10, 20, 20]) < 0.1


def test_suppresses_merged_box_covering_two_children():
    """一个大框套着两个各自的小框 = 合并框，要丢掉。"""
    suppressor = MergeSuppressor()
    boxes = [
        [100, 100, 140, 130],      # 左
        [150, 100, 190, 130],      # 右
        [100, 100, 190, 130],      # 把两个都框住的合并框
    ]

    kept = suppressor(boxes)

    assert len(kept) == 2
    assert [100, 100, 190, 130] not in kept
    assert suppressor.dropped == 1


def test_keeps_single_large_object():
    """只套住一个子框的大框不算合并框（那是正常的层级/大目标）。"""
    suppressor = MergeSuppressor()
    boxes = [[10, 10, 50, 50], [12, 12, 48, 48]]

    assert len(suppressor(boxes)) == 2
    assert suppressor.dropped == 0


def test_keeps_adjacent_non_overlapping():
    suppressor = MergeSuppressor()
    boxes = [[0, 0, 30, 30], [60, 0, 90, 30], [120, 0, 150, 30]]

    assert len(suppressor(boxes)) == 3


def test_ally_filter_drops_units_inside_known_operator_boxes():
    """RTAS 给了干员框时，落在里面的 enemy detection 要压掉。"""
    filter_ = AllyFilter(use_bars=False)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    units = [([50, 50, 90, 90], 1), ([120, 60, 150, 90], 2)]
    ally_boxes = [[45, 45, 95, 95]]

    kept = filter_.filter(image, units, ally_boxes=ally_boxes)

    assert [track for _box, track in kept] == [2]
    assert filter_.dropped == 1


def test_ally_filter_without_ally_boxes_keeps_everything():
    filter_ = AllyFilter(use_bars=False)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    units = [([10, 10, 30, 30], 1)]

    assert filter_.filter(image, units) == units
