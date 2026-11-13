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
# @brief Windows 平台后端

"""Windows 平台实现：窗口查找、后台输入、截图。"""

from .backend import WindowsBackend
from .elevate import build_command, is_admin, relaunch_as_admin
from .input import click, click_sequence, key, pack_xy, virtual_key
from .window import DEFAULT_KEYWORDS, find_game_window, to_ref

__all__ = [
    "DEFAULT_KEYWORDS",
    "WindowsBackend",
    "build_command",
    "click",
    "click_sequence",
    "find_game_window",
    "is_admin",
    "key",
    "pack_xy",
    "relaunch_as_admin",
    "to_ref",
    "virtual_key",
]
