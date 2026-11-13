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

# @author error-0x12
# @brief OCR 后端公共接口与结果模型

"""OCR 后端的公共接口。

RTAS 的视觉模块只需要两种能力：把图里的文字读出来，以及知道每段文字在哪里。
不同 OCR 引擎的返回格式差异很大（EasyOCR 的 ``(bbox, text, conf)``、
PaddleOCR 的嵌套列表、Tesseract 的词级 TSV、Umi-OCR 的 JSON ……），
因此这里统一成 :class:`OCRText`，再由 :meth:`OCRBackend.readtext` 转成
EasyOCR 风格的返回值，让既有调用方几乎不用改动。
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BBox = list[list[int]]

DEFAULT_LANGUAGES = ["ch_sim", "en"]


@dataclass(frozen=True)
class OCRText:
    """一段 OCR 识别结果。"""

    text: str
    confidence: float
    bbox: BBox | None = None

    @property
    def center(self) -> tuple[int, int] | None:
        """边界框中心点（没有 bbox 时返回 None）。"""
        if not self.bbox:
            return None
        xs = [p[0] for p in self.bbox]
        ys = [p[1] for p in self.bbox]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def to_easyocr(self) -> tuple[BBox, str, float]:
        """转换成 EasyOCR ``detail=1`` 的单条格式。"""
        return (self.bbox or []), self.text, float(self.confidence)


@dataclass(frozen=True)
class BackendInfo:
    """OCR 后端的可用性信息，用于 ``rtas test`` / ``rtas ocr --list``。"""

    name: str
    display_name: str
    available: bool
    kind: str = "local"  # local / http / binary
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "available": self.available,
            "kind": self.kind,
            "reason": self.reason,
        }


class OCRBackend(ABC):
    """OCR 后端基类。

    子类只需要实现 :meth:`recognize`，其余适配由基类完成。
    """

    name: str = ""
    display_name: str = ""
    kind: str = "local"
    #: 是否能给出文字位置。DeepSeek-OCR 这类整图转文本的模型为 False。
    supports_boxes: bool = True

    def __init__(
        self,
        languages: Sequence[str] | None = None,
        gpu: bool = False,
        **options: Any,
    ) -> None:
        self.languages = list(languages or DEFAULT_LANGUAGES)
        self.gpu = bool(gpu)
        self.options = options
        self._initialized = False

    # -- 生命周期 ---------------------------------------------------------
    @classmethod
    @abstractmethod
    def probe(cls) -> BackendInfo:
        """不加载模型、不联网地判断后端是否可用。"""

    @abstractmethod
    def recognize(self, image: np.ndarray) -> list[OCRText]:
        """识别图像中的文字。"""

    def initialize(self) -> None:
        """按需初始化。默认什么都不做。"""

    def shutdown(self) -> None:
        """释放资源。默认什么都不做。"""

    # -- EasyOCR 兼容接口 -------------------------------------------------
    def readtext(
        self,
        image: np.ndarray,
        detail: int = 1,
        **kwargs: Any,
    ) -> list[Any]:
        """EasyOCR 兼容的识别入口。

        Args:
            image: BGR 图像
            detail: 0 只返回文本列表；1 返回 ``[(bbox, text, confidence), ...]``
                （默认与 EasyOCR 一致，为 1）
            **kwargs: 其它引擎专属参数（``paragraph``、``contrast_ths`` 等），
                非 EasyOCR 后端会忽略。
        """
        self.initialize()
        items = self.recognize(image)
        if detail == 0:
            return [item.text for item in items]
        return [item.to_easyocr() for item in items]

    def __enter__(self) -> OCRBackend:
        self.initialize()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.shutdown()
        return False

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<{type(self).__name__} languages={self.languages} gpu={self.gpu}>"


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def encode_png_base64(image: np.ndarray) -> str:
    """把 BGR 图像编码成 base64 PNG，供 HTTP 类后端使用。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("图像编码失败")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def full_image_bbox(image: np.ndarray) -> BBox:
    """整图边界框，用于没有坐标信息的后端。"""
    h, w = image.shape[:2]
    return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]


def clamp_confidence(value: Any) -> float:
    """把各种来源的置信度收敛到 [0, 1]。"""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf != conf:  # NaN
        return 0.0
    if conf > 1.0:
        # Tesseract 的 conf 是 0~100，PaddleOCR/Umi 是 0~1
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def normalize_bbox(points: Any) -> BBox | None:
    """把各种点集（x1,y1,x2,y2 / 4 点 / numpy）统一成 4 点列表。"""
    if points is None:
        return None
    if hasattr(points, "tolist"):
        points = points.tolist()
    if not isinstance(points, (list, tuple)) or not points:
        return None
    try:
        if len(points) == 4 and all(isinstance(v, (int, float)) for v in points):
            x1, y1, x2, y2 = (int(v) for v in points)
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return [[int(p[0]), int(p[1])] for p in points]
    except (TypeError, ValueError, IndexError):
        return None
