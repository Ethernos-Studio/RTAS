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
# @brief PaddleOCR 后端

"""PaddleOCR 后端（PP-OCRv4/v5，中文识别准确率高）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendInfo, OCRBackend, OCRText, clamp_confidence, normalize_bbox
from .deps import spec_exists

# EasyOCR 语言名 -> PaddleOCR 语言名
_LANG_MAP = {
    "ch_sim": "ch",
    "ch_tra": "chinese_cht",
    "en": "en",
    "ja": "japan",
    "ko": "korean",
    "ru": "ru",
    "fr": "fr",
    "de": "german",
}


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"
    display_name = "PaddleOCR"
    kind = "local"
    supports_boxes = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reader: Any = None

    @classmethod
    def probe(cls) -> BackendInfo:
        available = spec_exists("paddleocr") and spec_exists("paddle")
        reason = "" if available else "未安装 paddleocr，请执行: uv sync --extra paddle"
        return BackendInfo(cls.name, cls.display_name, available, cls.kind, reason)

    def _paddle_lang(self) -> str:
        for lang in self.languages:
            if lang in _LANG_MAP:
                return _LANG_MAP[lang]
        return "ch"

    def initialize(self) -> None:
        if self._reader is not None:
            return
        import importlib

        paddleocr = importlib.import_module("paddleocr")
        lang = self._paddle_lang()
        base = {"lang": lang, "use_angle_cls": True, "show_log": False}
        # PaddleOCR 2.x 用 use_gpu；3.x 改用 device 且移除了 show_log
        for kwargs in (
            {**base, "use_gpu": self.gpu},
            {**base, "device": "gpu" if self.gpu else "cpu"},
            {"lang": lang},
        ):
            try:
                self._reader = paddleocr.PaddleOCR(**kwargs)
                return
            except (TypeError, ValueError):
                continue
        raise RuntimeError("PaddleOCR 初始化失败：接口版本不兼容")

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        self.initialize()
        raw: Any = None
        try:
            raw = self._reader.ocr(image, cls=True)
        except TypeError:
            raw = self._reader.ocr(image)
        except AttributeError:  # PaddleOCR 3.x
            raw = None
        if raw is None and hasattr(self._reader, "predict"):
            raw = self._reader.predict(image)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: Any) -> list[OCRText]:
        results: list[OCRText] = []
        if not raw:
            return results

        # paddleocr 3.x: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            for page in raw:
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                for i, text in enumerate(texts):
                    score = scores[i] if i < len(scores) else 0.0
                    bbox = normalize_bbox(polys[i]) if i < len(polys) else None
                    results.append(
                        OCRText(str(text).strip(), clamp_confidence(score), bbox)
                    )
            return results

        # paddleocr 2.x: [[ [box, (text, score)], ... ]]
        pages = raw
        if len(raw) == 1 and isinstance(raw[0], list):
            pages = raw[0]
        for line in pages:
            if not line:
                continue
            try:
                bbox, payload = line
                text, score = payload
            except (TypeError, ValueError):
                continue
            results.append(
                OCRText(str(text).strip(), clamp_confidence(score), normalize_bbox(bbox))
            )
        return results

    def shutdown(self) -> None:
        self._reader = None
