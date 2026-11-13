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
# @brief EasyOCR 后端

"""EasyOCR 后端（项目默认引擎，识别中文+英文效果稳定）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BackendInfo, OCRBackend, OCRText, clamp_confidence, normalize_bbox
from .deps import EASYOCR_AVAILABLE, TORCH_AVAILABLE, cuda_available, ensure_easyocr


class EasyOCRBackend(OCRBackend):
    name = "easyocr"
    display_name = "EasyOCR"
    kind = "local"
    supports_boxes = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reader: Any = None

    @classmethod
    def probe(cls) -> BackendInfo:
        available = EASYOCR_AVAILABLE and TORCH_AVAILABLE
        reason = "" if available else "未安装 easyocr / torch，请执行: uv sync"
        return BackendInfo(cls.name, cls.display_name, available, cls.kind, reason)

    def initialize(self) -> None:
        if self._reader is not None:
            return
        easyocr = ensure_easyocr()
        gpu = self.gpu and cuda_available()
        self._reader = easyocr.Reader(
            self.languages,
            gpu=gpu,
            verbose=bool(self.options.get("verbose", False)),
        )

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        self.initialize()
        raw = self._reader.readtext(image, detail=1, paragraph=False)
        results: list[OCRText] = []
        for bbox, text, score in raw:
            results.append(
                OCRText(str(text).strip(), clamp_confidence(score), normalize_bbox(bbox))
            )
        return results

    def readtext(self, image: np.ndarray, detail: int = 1, **kwargs: Any) -> list[Any]:
        """保留 EasyOCR 原生参数（``contrast_ths``、``paragraph`` 等）。"""
        self.initialize()
        return self._reader.readtext(image, detail=detail, **kwargs)

    def shutdown(self) -> None:
        self._reader = None
