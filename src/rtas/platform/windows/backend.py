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
# @brief Windows 操作后端

"""Windows 平台后端。

截图复用 ``rtas.tools.window_capture``（Windows Graphics Capture），
输入使用 ``rtas.platform.windows.input``（PostMessage 后台点击）。
"""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Sequence

import numpy as np

from ...tools.window_capture import CaptureError, capture_window, find_window_by_hwnd
from ..base import ClickMethod, InputBackend, WindowRef
from . import input as win_input
from .window import ensure_window_ready, find_game_window, to_ref

logger = logging.getLogger(__name__)


class WindowsBackend(InputBackend):
    """Win32 实现（仅 Windows）。"""

    name = "windows"

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsBackend 只能在 Windows 上使用")

    def find_window(
        self,
        keywords: Sequence[str] | None = None,
        hwnd: int | None = None,
    ) -> WindowRef:
        return ensure_window_ready(find_game_window(keywords, hwnd))

    def click(
        self,
        window: WindowRef,
        x: int,
        y: int,
        method: ClickMethod = ClickMethod.WINDOW,
    ) -> bool:
        method = ClickMethod(method)
        if method is ClickMethod.WINDOW:
            return win_input.click_at_cursor(window.hwnd, int(x), int(y))
        return win_input.click(window.hwnd, int(x), int(y), method=method.value)

    def key(self, window: WindowRef, key: str, method: ClickMethod = ClickMethod.WINDOW) -> bool:
        return win_input.key(window.hwnd, key, method=ClickMethod(method).value)

    def capture(self, window: WindowRef) -> np.ndarray:
        info = find_window_by_hwnd(window.hwnd)
        if info is None:
            raise LookupError(f"窗口已关闭: 0x{window.hwnd:X}")
        if info.minimized:
            ensure_window_ready(to_ref(info))
            info = find_window_by_hwnd(window.hwnd) or info
        try:
            image = capture_window(info)
        except CaptureError as exc:
            raise RuntimeError(f"截图失败: {exc}") from exc
        return np.asarray(image)[:, :, ::-1].copy()  # RGB -> BGR

    def is_admin(self) -> bool:
        try:
            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - 非 Windows
            return False


__all__ = ["WindowsBackend"]
