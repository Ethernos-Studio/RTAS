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
# @brief OCR 后端注册表与工厂

"""OCR 后端注册表。

内置 4 种引擎，统一通过 :func:`create_backend` 创建、:func:`list_backends` 查询：

===========  ==========================================  ==================
名称          说明                                        坐标
===========  ==========================================  ==================
``easyocr``   EasyOCR（默认，中文/英文稳定）              ✔
``paddleocr`` PaddleOCR（PP-OCR，中文准确率高）            ✔
``umi``       Umi-OCR（本机 HTTP 服务，免安装运行库）      ✔
``tesseract`` Tesseract（系统可执行文件）                  ✔
===========  ==========================================  ==================

选择顺序：``configure()``/``--ocr`` > 环境变量 ``RTAS_OCR_ENGINE`` > ``auto``
（按 easyocr → paddleocr → umi → tesseract 取第一个可用）。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any

from .base import BackendInfo, OCRBackend, OCRText
from .deps import (
    EASYOCR_AVAILABLE as EASYOCR_AVAILABLE,
)
from .deps import (
    TORCH_AVAILABLE as TORCH_AVAILABLE,
)
from .deps import (
    cuda_available as cuda_available,
)
from .deps import (
    ensure_easyocr as ensure_easyocr,
)
from .deps import (
    ensure_torch as ensure_torch,
)
from .deps import (
    spec_exists as spec_exists,
)
from .easyocr_backend import EasyOCRBackend
from .paddleocr_backend import PaddleOCRBackend
from .tesseract_backend import TesseractBackend
from .umi_backend import UmiOCRBackend

logger = logging.getLogger(__name__)

__all__ = [
    "BackendInfo",
    "OCRBackend",
    "OCRText",
    "BACKENDS",
    "BACKEND_PREFERENCE",
    "any_available",
    "available_backend_name",
    "configure",
    "create_backend",
    "current_engine",
    "default_engine_name",
    "engine_names",
    "list_backends",
]

BACKEND_PREFERENCE: tuple[str, ...] = (
    "easyocr",
    "paddleocr",
    "umi",
    "tesseract",
)

BACKENDS: dict[str, type[OCRBackend]] = {
    cls.name: cls
    for cls in (
        EasyOCRBackend,
        PaddleOCRBackend,
        UmiOCRBackend,
        TesseractBackend,
    )
}

_state: dict[str, str | None] = {"engine": None, "url": None}


def configure(engine: str | None = None, url: str | None = None) -> None:
    """设置本次进程使用的默认引擎与地址（CLI 用）。"""
    if engine is not None:
        _state["engine"] = None if engine in ("", "auto") else engine
    if url is not None:
        _state["url"] = url or None


def current_engine() -> str:
    """返回当前配置的引擎名（可能是 ``auto``）。"""
    return _state["engine"] or os.environ.get("RTAS_OCR_ENGINE") or "auto"


def engine_names() -> list[str]:
    """全部可用引擎名（含 ``auto``）。"""
    return ["auto", *BACKENDS.keys()]


def list_backends() -> list[BackendInfo]:
    """探测全部后端，返回可用性列表（不加载模型）。"""
    return [cls.probe() for cls in BACKENDS.values()]


def available_backend_name() -> str | None:
    """按优先级返回第一个可用的后端名。"""
    for name in BACKEND_PREFERENCE:
        try:
            if BACKENDS[name].probe().available:
                return name
        except Exception as exc:  # pragma: no cover - 探测不应失败
            logger.debug(f"探测 OCR 后端 {name} 失败: {exc}")
    return None


def any_available() -> bool:
    return available_backend_name() is not None


def default_engine_name() -> str:
    """解析最终使用的引擎名；没有可用引擎时抛出 RuntimeError。"""
    configured = current_engine()
    if configured != "auto":
        if configured not in BACKENDS:
            raise ValueError(
                f"未知 OCR 引擎: {configured}，可选: {', '.join(BACKENDS)}"
            )
        return configured
    name = available_backend_name()
    if name is None:
        details = "; ".join(
            f"{info.name}: {info.reason}" for info in list_backends() if not info.available
        )
        raise RuntimeError(
            "没有可用的 OCR 引擎。请安装 easyocr（uv sync）、PaddleOCR "
            "（uv sync --extra paddle）、Tesseract，或启动 Umi-OCR。\n" + details
        )
    return name


def create_backend(
    name: str | None = None,
    languages: Sequence[str] | None = None,
    gpu: bool = False,
    url: str | None = None,
    **options: Any,
) -> OCRBackend:
    """创建 OCR 后端实例。

    Args:
        name: 引擎名；``None``/``auto`` 时按优先级自动选择
        languages: 语言列表（EasyOCR 命名，如 ``['ch_sim', 'en']``）
        gpu: 是否尝试使用 GPU
        url: HTTP 后端地址（Umi-OCR / DeepSeek-OCR）
        **options: 透传给具体后端

    Raises:
        ValueError: 引擎名不存在
        RuntimeError: 引擎不可用
    """
    resolved = name or default_engine_name()
    if resolved in ("auto", "", None):
        resolved = default_engine_name()

    cls = BACKENDS.get(str(resolved))
    if cls is None:
        raise ValueError(
            f"未知 OCR 引擎: {resolved}，可选: {', '.join(BACKENDS)}"
        )

    endpoint = url or _state.get("url")
    if endpoint and resolved in ("umi", "deepseek"):
        options.setdefault("url", endpoint)

    backend = cls(languages=languages, gpu=gpu, **options)

    info = cls.probe()
    if not info.available:
        raise RuntimeError(
            f"OCR 引擎 {cls.display_name} 不可用: {info.reason}"
        )
    return backend
