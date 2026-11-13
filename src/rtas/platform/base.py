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
# @brief 跨平台操作后端接口

"""跨平台操作后端。

RTAS 是通用的《明日方舟》操作代理库，窗口查找、截图、鼠标/键盘注入都属
平台相关能力。接口定义在这里，具体实现放在 ``rtas.platform.<平台>``：

- ``rtas.platform.windows``：Win32（PostMessage / SendMessage / 光标）
- 其它平台：暂未实现，调用时会抛 :class:`UnsupportedPlatform`

新增平台只需要实现 :class:`InputBackend` 并在 :func:`get_backend` 中注册。
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class UnsupportedPlatform(RuntimeError):
    """当前平台不支持该操作。"""


class ClickMethod(str, Enum):
    """鼠标点击方式。

    - ``window``：移动窗口让目标点落在真实光标下，再投递消息。不移动光标、
      不抢焦点，是 Unity 客户端（明日方舟 PC 端）唯一可靠的后台方式，
      对应 MAA 的 ``PostMessageWithWindowPos``
    - ``post``：PostMessage，纯消息点击；模拟器窗口可用，Unity 客户端会忽略
    - ``send``：SendMessage，同步等待窗口处理
    - ``cursor``：真实光标移动 + 点击，会抢鼠标，仅作兜底
    """

    WINDOW = "window"
    POST = "post"
    SEND = "send"
    CURSOR = "cursor"


@dataclass(frozen=True)
class WindowRef:
    """平台无关的窗口引用。"""

    hwnd: int
    title: str
    process: str
    class_name: str
    width: int
    height: int
    minimized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "process": self.process,
            "class_name": self.class_name,
            "width": self.width,
            "height": self.height,
            "minimized": self.minimized,
        }


class InputBackend(ABC):
    """窗口、输入与截图的平台后端。"""

    name: str = "unknown"

    @abstractmethod
    def find_window(
        self,
        keywords: Sequence[str] | None = None,
        hwnd: int | None = None,
    ) -> WindowRef:
        """按关键字或句柄查找游戏窗口，找不到时抛 ``LookupError``。"""

    @abstractmethod
    def click(
        self,
        window: WindowRef,
        x: int,
        y: int,
        method: ClickMethod = ClickMethod.POST,
    ) -> bool:
        """在窗口客户区坐标 (x, y) 点击。"""

    @abstractmethod
    def key(self, window: WindowRef, key: str, method: ClickMethod = ClickMethod.POST) -> bool:
        """向窗口发送按键（如 ``esc``、``f1``、``enter``）。"""

    @abstractmethod
    def capture(self, window: WindowRef) -> Any:
        """截取窗口客户区，返回 BGR ``numpy.ndarray``。"""

    @abstractmethod
    def is_admin(self) -> bool:
        """当前进程是否以管理员身份运行。"""


def get_backend() -> InputBackend:
    """返回当前平台的后端实例。"""
    if sys.platform == "win32":
        from .windows import WindowsBackend

        return WindowsBackend()
    raise UnsupportedPlatform(
        f"当前平台 {sys.platform} 尚未实现输入/截图后端；"
        "RTAS 目前只提供 Windows 实现，可在 rtas/platform/ 下新增平台。"
    )
