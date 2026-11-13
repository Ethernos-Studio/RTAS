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
# @brief Windows 窗口查找

"""Windows 窗口查找（复用 ``rtas.tools.window_capture`` 的枚举能力）。"""

from __future__ import annotations

from collections.abc import Sequence

from ...tools.window_capture import (
    WindowInfo,
    find_window_by_hwnd,
    find_windows,
    list_windows,
)
from ..base import WindowRef

DEFAULT_KEYWORDS = ("明日方舟", "arknights", "Arknights")

#: 游戏本体进程名（优先于标题匹配）
GAME_PROCESSES = ("arknights.exe", "games.exe")

#: 明确不是游戏窗口的进程（MAA 标题里也含「明日方舟」）
EXCLUDED_PROCESSES = ("maa.exe", "python.exe", "pythonw.exe")


def _is_excluded(process: str) -> bool:
    return process.lower() in {p.lower() for p in EXCLUDED_PROCESSES}


def to_ref(info: WindowInfo) -> WindowRef:
    return WindowRef(
        hwnd=info.hwnd,
        title=info.title,
        process=info.process,
        class_name=info.class_name,
        width=info.client_rect.width,
        height=info.client_rect.height,
        minimized=info.minimized,
    )


def _window_score(info: WindowInfo) -> tuple:
    """窗口优先级：游戏本体 > 启动器 > 其它 > 非最小化 > 客户区更大。

    ``Games.exe`` 是客户端启动器，标题同样叫「明日方舟」、窗口还比游戏大；
    ``Arknights.exe`` 才是真正的游戏画面，必须排在它前面，否则截到/点到的都是启动器。
    """
    process = (info.process or "").lower()
    if process == "arknights.exe":
        rank = 0
    elif process in GAME_PROCESSES:
        rank = 1
    else:
        rank = 2
    return (rank, info.minimized, -info.client_rect.area)


def find_game_window(
    keywords: Sequence[str] | None = None,
    hwnd: int | None = None,
    *,
    include_minimized: bool = True,
) -> WindowRef:
    """查找游戏窗口。

    优先按进程名（Arknights.exe / Games.exe）匹配，避免误选标题里带
    「明日方舟」的其它程序（例如 MAA 主窗口）。

    Args:
        keywords: 标题/进程名关键字，默认「明日方舟 / arknights」
        hwnd: 直接指定窗口句柄
        include_minimized: 是否接受最小化窗口

    Raises:
        LookupError: 没有找到窗口
    """
    if hwnd:
        info = find_window_by_hwnd(int(hwnd))
        if info is None:
            raise LookupError(f"找不到窗口句柄 0x{int(hwnd):X}")
        return to_ref(info)

    matches = [
        window
        for window in find_windows(
            tuple(keywords or DEFAULT_KEYWORDS), include_minimized=include_minimized
        )
        if not _is_excluded(window.process)
    ]
    if not matches:
        raise LookupError(
            "找不到《明日方舟》窗口，请先启动游戏；可用 `rtas capture --list` 查看窗口标题"
        )
    matches.sort(key=_window_score)
    return to_ref(matches[0])


def ensure_window_ready(ref: WindowRef) -> WindowRef:
    """确保窗口可交互：最小化时临时恢复，客户区尺寸为 0 时重新读取。

    恢复使用 ``SW_SHOWNOACTIVATE`` 而不是 ``SW_RESTORE``，避免把游戏窗口
    切到前台、把用户正在用的窗口压下去。
    """
    from ...tools.window_capture import _wait_for_client_area, user32

    if not ref.minimized and ref.width > 0 and ref.height > 0:
        return ref

    info = find_window_by_hwnd(ref.hwnd)
    if info is None:
        raise LookupError(f"窗口已关闭: 0x{ref.hwnd:X}")

    if info.minimized:
        user32.ShowWindow(info.hwnd, 4)  # SW_SHOWNOACTIVATE
        _wait_for_client_area(info.hwnd, timeout=1.5)
        info = find_window_by_hwnd(ref.hwnd) or info

    return to_ref(info)


__all__ = [
    "DEFAULT_KEYWORDS",
    "EXCLUDED_PROCESSES",
    "GAME_PROCESSES",
    "WindowInfo",
    "ensure_window_ready",
    "find_game_window",
    "find_window_by_hwnd",
    "find_windows",
    "list_windows",
    "to_ref",
]
