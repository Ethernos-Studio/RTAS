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
# @brief 平台后端注册

"""平台后端。

当前实现了 Windows（``rtas.platform.windows``）。其它平台可以照此新增一个
子包并在 :func:`rtas.platform.base.get_backend` 中注册。
"""

from .base import (
    ClickMethod,
    InputBackend,
    UnsupportedPlatform,
    WindowRef,
    get_backend,
)

__all__ = [
    "ClickMethod",
    "InputBackend",
    "UnsupportedPlatform",
    "WindowRef",
    "get_backend",
]
