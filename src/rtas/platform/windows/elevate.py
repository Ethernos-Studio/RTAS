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
# @brief 以管理员身份重启自身（绕过 UIPI 限制）

"""自提权工具。

《明日方舟》PC 端进程的完整性级别高于普通进程，Windows 的 UIPI 会拦截
低权限进程发来的 ``SetWindowPos`` / ``PostMessage``（表现为错误 5、
点击无任何效果）。因此后台点击必须以管理员身份运行，这与 MAA 一致。

``relaunch_as_admin`` 通过 ``ShellExecuteW(..., "runas", ...)`` 弹出 UAC
重新启动同一个命令行，成功启动后调用方应直接退出。
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Sequence

IS_WINDOWS = sys.platform == "win32"

SW_SHOWNORMAL = 1

#: ShellExecuteW 返回值 <= 32 表示失败
_SE_ERR_CANCELLED = 1223


def is_admin() -> bool:
    """当前进程是否以管理员身份运行。"""
    if not IS_WINDOWS:
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - 非 Windows
        return False


def _quote(arg: str) -> str:
    """按 Windows 命令行规则给参数加引号。"""
    if not arg or any(ch in arg for ch in ' \t"'):
        escaped = arg.replace('"', '\\"')
        return f'"{escaped}"'
    return arg


def build_command(argv: Sequence[str] | None = None) -> tuple[str, str]:
    """拆出「可执行文件」与「参数串」，供 ``ShellExecuteW`` 使用。"""
    argv = list(sys.argv if argv is None else argv)
    if not argv:
        raise ValueError("argv 不能为空")

    args = [a for a in argv[1:] if a != "--elevate"]

    # 打包后的可执行文件直接重启自身
    if getattr(sys, "frozen", False):
        return argv[0], " ".join(_quote(a) for a in args)

    # 源码运行：用当前解释器重启 -m rtas
    python = sys.executable or "python"
    return python, " ".join(_quote(a) for a in ["-m", "rtas", *args])


def relaunch_as_admin(
    argv: Sequence[str] | None = None,
    *,
    cwd: str | None = None,
) -> bool:
    """用 UAC 以管理员身份重启当前命令。

    Args:
        argv: 命令行参数（默认 ``sys.argv``）
        cwd: 新进程工作目录（默认当前目录）

    Returns:
        是否成功启动提权进程；用户在 UAC 弹窗点「否」时返回 ``False``。
    """
    if not IS_WINDOWS:
        raise RuntimeError("自提权仅 Windows 支持")
    if is_admin():
        return False

    executable, params = build_command(argv)
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.ShellExecuteW.restype = ctypes.c_void_p
    result = shell32.ShellExecuteW(
        None,
        "runas",
        executable,
        params,
        cwd or os.getcwd(),
        SW_SHOWNORMAL,
    )
    return int(result or 0) > 32


__all__ = ["build_command", "is_admin", "relaunch_as_admin"]
