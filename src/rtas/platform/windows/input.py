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
# @brief Win32 后台输入（PostMessage / SendMessage）

"""Win32 后台输入实现。

核心是给窗口直接投递鼠标/键盘消息（``PostMessage``），因此：

- 不移动系统光标，不抢焦点，用户可以继续用电脑；
- 游戏窗口被遮挡、在后台也能收到点击（明日方舟 PC 端实测有效，
  与 MAA 的 ``PostMsg-WindowPos`` 方案一致）；
- 坐标使用**客户区坐标**，与截图坐标系一致，无需换算窗口边框。

注意：Windows 的 UIPI 机制要求同权限级别才能投递消息，所以操作游戏需要
以管理员身份运行 RTAS（与 MAA 的要求相同）。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import sys
import time
from collections.abc import Sequence

IS_WINDOWS = sys.platform == "win32"

WM_MOUSEMOVE = 0x0200
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_ACTIVATE = 0x0006
MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002

SMTO_ABORTIFHUNG = 0x0002
SMTO_BLOCK = 0x0001

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

if IS_WINDOWS:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    ]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.SendMessageTimeoutW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
        wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t),
    ]
    user32.SendMessageTimeoutW.restype = wintypes.LPARAM
    user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
    user32.MapVirtualKeyW.restype = wintypes.UINT
    user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    user32.SetCursorPos.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
else:  # pragma: no cover - 非 Windows 环境仅用于导入
    user32 = None


def pack_xy(x: int, y: int) -> int:
    """把客户区坐标打包成 Win32 消息的 lParam。"""
    return ((int(y) & 0xFFFF) << 16) | (int(x) & 0xFFFF)


def _send(hwnd: int, message: int, wparam: int, lparam: int, timeout_ms: int = 500) -> bool:
    """SendMessageTimeout：同步发送但不会因为窗口卡死而永久阻塞。"""
    assert user32 is not None
    result = ctypes.c_size_t(0)
    ok = user32.SendMessageTimeoutW(
        hwnd, message, wparam, lparam, SMTO_ABORTIFHUNG, timeout_ms, ctypes.byref(result)
    )
    return bool(ok)


def _dispatch(hwnd: int, message: int, wparam: int, lparam: int, send: bool) -> bool:
    if user32 is None:
        raise RuntimeError("Win32 输入仅在 Windows 上可用")
    if send:
        return _send(hwnd, message, wparam, lparam)
    return bool(user32.PostMessageW(hwnd, message, wparam, lparam))


def click(
    hwnd: int,
    x: int,
    y: int,
    *,
    method: str = "post",
    delay: float = 0.03,
    button: str = "left",
) -> bool:
    """在窗口客户区 (x, y) 点击。

    Args:
        hwnd: 窗口句柄
        x, y: 客户区像素坐标
        method: ``post``（默认，后台）/ ``send`` / ``cursor``
        delay: 按下与抬起之间的间隔
        button: ``left`` 或 ``right``
    """
    if method == "cursor":
        return _click_cursor(hwnd, x, y, button=button, delay=delay)

    send = method == "send"
    lparam = pack_xy(x, y)
    down, up = (
        (WM_LBUTTONDOWN, WM_LBUTTONUP) if button == "left" else (WM_RBUTTONDOWN, WM_RBUTTONUP)
    )
    button_flag = MK_LBUTTON if button == "left" else MK_RBUTTON

    # 先移动一次，某些引擎需要先收到 WM_MOUSEMOVE 才认为鼠标进入控件
    _dispatch(hwnd, WM_MOUSEMOVE, 0, lparam, send)
    _dispatch(hwnd, down, button_flag, lparam, send)
    time.sleep(delay)
    _dispatch(hwnd, up, 0, lparam, send)
    return True


def _click_cursor(hwnd: int, x: int, y: int, *, button: str = "left", delay: float = 0.03) -> bool:
    """真实光标点击（会抢鼠标，仅作兜底）。"""
    assert user32 is not None
    point = wintypes.POINT(int(x), int(y))
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return False
    user32.SetCursorPos(point.x, point.y)
    time.sleep(0.02)
    down = MOUSEEVENTF_LEFTDOWN if button == "left" else 0x0008  # RIGHTDOWN
    up = MOUSEEVENTF_LEFTUP if button == "left" else 0x0010      # RIGHTUP
    user32.mouse_event(down, 0, 0, 0, 0)
    time.sleep(delay)
    user32.mouse_event(up, 0, 0, 0, 0)
    return True


_VK_NAMES = {
    "esc": 0x1B,
    "escape": 0x1B,
    "enter": 0x0D,
    "tab": 0x09,
    "space": 0x20,
    "backspace": 0x08,
    "delete": 0x2E,
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
}
_VK_NAMES.update({f"f{i}": 0x6F + i for i in range(1, 13)})


def virtual_key(name: str) -> int:
    """把按键名转换成虚拟键码（支持字母、数字、F1~F12 与常用键）。"""
    key = name.strip().lower()
    if key in _VK_NAMES:
        return _VK_NAMES[key]
    if len(key) == 1:
        return ord(key.upper())
    raise ValueError(f"不支持的按键: {name}")


def key(hwnd: int, name: str, *, method: str = "post") -> bool:
    """向窗口发送一次按键（按下 + 抬起）。"""
    vk = virtual_key(name)
    scan = user32.MapVirtualKeyW(vk, 0) if user32 is not None else 0
    down_lparam = 1 | (scan << 16)
    up_lparam = down_lparam | (0xC0000000)
    send = method == "send"
    _dispatch(hwnd, WM_KEYDOWN, vk, down_lparam, send)
    time.sleep(0.02)
    _dispatch(hwnd, WM_KEYUP, vk, up_lparam, send)
    return True


def click_sequence(
    hwnd: int,
    points: Sequence[tuple[int, int]],
    *,
    method: str = "post",
    interval: float = 0.3,
    window_pos: bool = False,
) -> bool:
    """依次点击多个坐标（用于「打开导航栏 → 点首页」这类多步操作）。"""
    for index, (x, y) in enumerate(points):
        if window_pos:
            click_at_cursor(hwnd, x, y, method=method)
        else:
            click(hwnd, x, y, method=method)
        if index < len(points) - 1:
            time.sleep(interval)
    return True


# ---------------------------------------------------------------------------
# 窗口对齐式后台点击（MAA 的 PostMessageWithWindowPos 思路）
# ---------------------------------------------------------------------------

def _client_origin_on_screen(hwnd: int) -> tuple[int, int] | None:
    """返回客户区左上角的屏幕坐标。"""
    assert user32 is not None
    point = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None
    return point.x, point.y


def _window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回窗口矩形 (left, top, right, bottom)。"""
    assert user32 is not None
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    return rect.left, rect.top, rect.right, rect.bottom


def _cursor_pos() -> tuple[int, int] | None:
    assert user32 is not None
    point = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    return point.x, point.y


def _move_window(hwnd: int, left: int, top: int, *, timeout: float = 0.3) -> bool:
    """只移动窗口位置，不改尺寸、不激活、不同步等待。

    ``SWP_ASYNCWINDOWPOS`` 让窗口线程自己处理移动，调用会立刻返回，
    因此这里再轮询一次位置，确保落位后再投递鼠标消息。
    """
    assert user32 is not None
    SWP_NOSIZE = 0x0001
    SWP_NOZORDER = 0x0004
    SWP_NOACTIVATE = 0x0010
    SWP_ASYNCWINDOWPOS = 0x4000
    ok = user32.SetWindowPos(
        hwnd, 0, int(left), int(top), 0, 0,
        SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_ASYNCWINDOWPOS,
    )
    if not ok:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rect = _window_rect(hwnd)
        if rect is not None and abs(rect[0] - left) <= 1 and abs(rect[1] - top) <= 1:
            return True
        time.sleep(0.01)
    return True


def align_window_to_point(hwnd: int, x: int, y: int) -> tuple[bool, tuple[int, int] | None]:
    """移动窗口，使客户区坐标 (x, y) 正好落在当前真实光标位置下。

    Unity 客户端在判定点击时会参考真实光标位置：单纯 PostMessage 的 lParam
    坐标不够，必须让目标点与光标重合。移动窗口而不是移动光标，用户的手感
    不受影响（MAA 的 PostMessageWithWindowPos 即此思路）。

    Returns:
        (是否成功, 原窗口左上角坐标)
    """
    cursor = _cursor_pos()
    origin = _client_origin_on_screen(hwnd)
    rect = _window_rect(hwnd)
    if cursor is None or origin is None or rect is None:
        return False, None

    # 客户区原点相对窗口左上角的边框偏移
    border_x = origin[0] - rect[0]
    border_y = origin[1] - rect[1]
    new_left = cursor[0] - int(x) - border_x
    new_top = cursor[1] - int(y) - border_y

    if abs(new_left - rect[0]) <= 1 and abs(new_top - rect[1]) <= 1:
        return True, (rect[0], rect[1])

    if not _move_window(hwnd, new_left, new_top):
        return False, (rect[0], rect[1])
    return True, (rect[0], rect[1])


def click_at_cursor(
    hwnd: int,
    x: int,
    y: int,
    *,
    method: str = "post",
    delay: float = 0.03,
    button: str = "left",
    restore_window: bool = True,
) -> bool:
    """窗口对齐式点击：移动窗口让 (x, y) 落在光标下，再投递鼠标消息。

    全程不移动系统光标、不抢焦点，窗口位置在点击后恢复。
    """
    aligned, original = align_window_to_point(hwnd, x, y)
    if not aligned:
        return False
    try:
        # 让 Unity 认为窗口处于激活状态，否则会丢弃鼠标消息
        _dispatch(hwnd, WM_ACTIVATE, 1, 0, method == "send")  # WA_ACTIVE
        time.sleep(0.01)
        return click(hwnd, x, y, method=method, delay=delay, button=button)
    finally:
        if restore_window and original is not None:
            time.sleep(0.02)
            _move_window(hwnd, original[0], original[1])
