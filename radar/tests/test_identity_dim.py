# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""图库/编码器维度自检：维度不匹配要给清晰报错，而不是 matmul 崩溃。"""

from __future__ import annotations

import numpy as np
import pytest

from eyesofpriestess.identity import Gallery


def test_identify_rejects_dimension_mismatch():
    gallery = Gallery(codes=["A", "B"], prototypes=np.zeros((2, 256), dtype=np.float32),
                      samples={"A": 1, "B": 1})

    with pytest.raises(ValueError, match="特征维度不匹配"):
        gallery.identify(np.zeros((1, 512), dtype=np.float32))


def test_identify_accepts_matching_dimension():
    prototypes = np.eye(2, dtype=np.float32)
    gallery = Gallery(codes=["A", "B"], prototypes=prototypes, samples={"A": 1, "B": 1})

    matches = gallery.identify(np.array([[0.9, 0.1]], dtype=np.float32))

    assert matches[0].code == "A"


def test_identify_always_returns_a_code_even_with_low_score():
    """按用户要求：低置信度不隐藏，永远给最可能的编号，分数自己看。"""
    prototypes = np.eye(2, dtype=np.float32)
    gallery = Gallery(codes=["A", "B"], prototypes=prototypes, samples={"A": 1, "B": 1})

    matches = gallery.identify(np.array([[0.2, 0.1]], dtype=np.float32))

    assert matches[0].code == "A"
    assert matches[0].score == pytest.approx(0.2, abs=1e-6)
