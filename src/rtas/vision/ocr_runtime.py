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
# @brief OCR 运行时门面（向后兼容层）

"""OCR 运行时门面。

历史上这里只负责延迟导入 EasyOCR；现在实际实现位于 :mod:`rtas.vision.ocr`，
本模块保留原有 API（``create_reader`` / ``cuda_available`` / ``EASYOCR_AVAILABLE``），
让视觉模块不必关心具体引擎。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .ocr import (
    BACKENDS,
    BackendInfo,
    OCRBackend,
    OCRText,
    any_available,
    available_backend_name,
    configure,
    create_backend,
    default_engine_name,
    engine_names,
    list_backends,
)
from .ocr.deps import (
    EASYOCR_AVAILABLE,
    TORCH_AVAILABLE,
    cuda_available,
    ensure_easyocr,
    ensure_torch,
    spec_exists,
)

__all__ = [
    "BACKENDS",
    "BackendInfo",
    "EASYOCR_AVAILABLE",
    "OCR_AVAILABLE",
    "OCRBackend",
    "OCRText",
    "TORCH_AVAILABLE",
    "any_available",
    "available_backend_name",
    "configure",
    "create_backend",
    "create_reader",
    "cuda_available",
    "default_engine_name",
    "engine_names",
    "ensure_easyocr",
    "ensure_torch",
    "list_backends",
    "spec_exists",
]

#: 是否存在任意可用 OCR 引擎（不再仅限于 EasyOCR）
OCR_AVAILABLE: bool = any_available()


def create_reader(
    languages: Sequence[str],
    gpu: bool = False,
    **kwargs: Any,
) -> OCRBackend:
    """创建 OCR 后端（保持 EasyOCR ``Reader`` 的调用风格）。

    ``kwargs`` 中可带 ``engine`` 指定引擎，其余参数透传给后端。
    """
    engine = kwargs.pop("engine", None)
    backend = create_backend(engine, languages=languages, gpu=gpu, **kwargs)
    backend.initialize()
    return backend
