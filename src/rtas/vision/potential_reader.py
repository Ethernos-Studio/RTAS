# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of RTAS (Rhodeisland Tactical Autonomous System).
#
# RTAS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RTAS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with RTAS. If not, see <https://www.gnu.org/licenses/>.

"""干员详情页的潜能识别。

编队卡面只有等级和精英化，看不到潜能；潜能（以及它带来的部署费用减免）
只出现在「干员详情」页：`潜能` 标签下方有一个小徽章，写着 `MAX` 或 `1`~`5`。

这里不依赖具体分辨率：先整页 OCR，找到 `潜能` 标签，再取它正下方最近的
`MAX` / 数字文本块。同时顺手把 `精英化` 的徽章文字也读出来（`MAX` 或 `1`/`2`）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: `MAX` 在 OCR 里常被读成 NAX / MA X / M4X 之类
_MAX_PATTERN = re.compile(r'^[MN][A4][XK]$')
#: 潜能徽章正文（1~5，0 表示没有潜能）
_LEVEL_PATTERN = re.compile(r'^([0-5])$')


@dataclass
class PotentialInfo:
    """详情页上读到的潜能 / 精英化。"""

    potential: Optional[int] = None          # 0~5，满潜时为 max_potential_level
    max_potential: bool = False              # 徽章写的是 MAX
    raw_text: str = ""                       # 徽章原始 OCR 文本
    elite_text: str = ""                     # 精英化徽章的原始文本（MAX / 1 / 2）
    boxes: List[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "potential": self.potential,
            "max_potential": self.max_potential,
            "raw_text": self.raw_text,
            "elite_text": self.elite_text,
        }


def _box_of(item: Any) -> tuple[float, float, float, float]:
    points = getattr(item, "bbox", None) or []
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    return min(xs), min(ys), max(xs), max(ys)


def _normalize(text: str) -> str:
    return "".join(ch for ch in (text or "").upper() if ch.isalnum())


def _is_max(text: str) -> bool:
    return bool(_MAX_PATTERN.match(_normalize(text)))


def _badge_below(items: Sequence[Any], label: str,
                 max_drop: float = 120.0) -> Optional[Any]:
    """在 `label` 正下方找最近的文本块（徽章文字）。"""
    label_box = None
    for item in items:
        if label in (getattr(item, "text", "") or ""):
            label_box = _box_of(item)
            break
    if label_box is None:
        return None

    lx0, ly0, lx1, ly1 = label_box
    center_x = (lx0 + lx1) / 2
    best = None
    best_distance = None
    for item in items:
        if item is None or getattr(item, "text", "") == label:
            continue
        x0, y0, x1, y1 = _box_of(item)
        drop = y0 - ly1
        if drop < -5 or drop > max_drop:
            continue
        # 徽章一般和标签居中对齐（允许一定偏移）
        offset = abs((x0 + x1) / 2 - center_x)
        if offset > (lx1 - lx0):
            continue
        distance = drop + offset * 0.5
        if best_distance is None or distance < best_distance:
            best, best_distance = item, distance
    return best


def read_potential(image: Union[str, Path, np.ndarray], *,
                   backend: Optional[Any] = None,
                   max_potential_level: int = 5) -> PotentialInfo:
    """识别详情页上的潜能与精英化文字。

    Args:
        image: 图片路径或 BGR 数组
        backend: 已初始化的 OCR 后端；不传则用默认引擎现建一个
        max_potential_level: OCR 读到 MAX 时返回的满潜等级（6 星通常为 5）

    Returns:
        ``PotentialInfo``；没读到就是 ``potential=None``
    """
    if isinstance(image, (str, Path)):
        data = cv2.imread(str(image))
        if data is None:
            raise FileNotFoundError(f"无法读取图片: {image}")
    else:
        data = image

    if backend is None:
        from rtas.vision.ocr import create_backend
        from rtas.vision.ocr.deps import cuda_available

        backend = create_backend(gpu=bool(cuda_available()))
        # 各后端的 initialize() 成功时返回 None，只有明确 False 才算失败
        if backend.initialize() is False:
            raise RuntimeError("OCR 后端初始化失败")

    items = backend.recognize(data)
    info = PotentialInfo(boxes=list(items))

    badge = _badge_below(items, "潜能")
    if badge is not None:
        info.raw_text = getattr(badge, "text", "") or ""
        normalized = _normalize(info.raw_text)
        if _is_max(normalized):
            info.potential = max_potential_level
            info.max_potential = True
        else:
            match = _LEVEL_PATTERN.match(normalized)
            if match:
                info.potential = int(match.group(1))
                info.max_potential = info.potential >= max_potential_level

    elite_badge = _badge_below(items, "精英化")
    if elite_badge is not None:
        info.elite_text = getattr(elite_badge, "text", "") or ""

    logger.info(
        f"潜能识别: potential={info.potential} max={info.max_potential} "
        f"raw={info.raw_text!r} elite={info.elite_text!r}"
    )
    return info
