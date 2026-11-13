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
# @brief 可选依赖的探测与惰性导入

"""重量级 OCR 依赖的探测与惰性导入。

``import easyocr`` 会连带加载 torch、torchvision、scikit-image，
``import paddleocr``、``transformers`` 同理。RTAS 的 CLI 启动时必须避免这些开销，
所以这里只用 :func:`importlib.util.find_spec` 做毫秒级探测，
真正的 import 推迟到第一次创建后端时。
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_lock = threading.RLock()
_modules: dict[str, Any] = {}


def spec_exists(module_name: str) -> bool:
    """判断模块是否已安装（不会真正导入）。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def ensure_module(module_name: str, friendly_name: str = "") -> Any:
    """导入并缓存模块；失败时抛出带安装提示的 ImportError。"""
    cached = _modules.get(module_name)
    if cached is not None:
        return cached

    with _lock:
        cached = _modules.get(module_name)
        if cached is not None:
            return cached
        if not spec_exists(module_name):
            label = friendly_name or module_name
            raise ImportError(
                f"{label} 未安装。请执行: uv sync --extra {_extra_for(module_name)}"
            )
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            label = friendly_name or module_name
            raise ImportError(f"{label} 导入失败（依赖不完整）: {exc}") from exc
        _modules[module_name] = module
        return module


def _extra_for(module_name: str) -> str:
    mapping = {
        "easyocr": "easyocr",
        "paddleocr": "paddle",
        "paddle": "paddle",
        "pytesseract": "tesseract",
    }
    return mapping.get(module_name, module_name)


EASYOCR_AVAILABLE: bool = spec_exists("easyocr")
TORCH_AVAILABLE: bool = spec_exists("torch")


def ensure_easyocr() -> Any:
    """返回 easyocr 模块（首次调用约 10~20 秒）。"""
    return ensure_module("easyocr", "EasyOCR")


def ensure_torch() -> Any:
    """返回 torch 模块。"""
    return ensure_module("torch", "PyTorch")


def cuda_available() -> bool:
    """检测 CUDA 是否可用；任何异常都按不可用处理。"""
    if not TORCH_AVAILABLE:
        return False
    try:
        return bool(ensure_torch().cuda.is_available())
    except Exception as exc:  # pragma: no cover - 依赖驱动环境
        logger.debug(f"CUDA检测失败: {exc}")
        return False
