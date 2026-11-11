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

"""Tests for the window screenshot helper."""

from __future__ import annotations

import sys

import pytest

from rtas.tools import window_capture


def test_default_temp_dir_points_at_project_temp() -> None:
    assert window_capture.default_temp_dir().name == "__temp__"
    assert window_capture.default_temp_dir().is_absolute()


def test_unique_path_is_unique(tmp_path) -> None:
    first = window_capture.unique_path(tmp_path, "shot")
    first.touch()
    second = window_capture.unique_path(tmp_path, "shot")

    assert first != second
    assert second.parent == tmp_path
    assert not second.exists()


def test_unique_path_sanitizes_prefix(tmp_path) -> None:
    path = window_capture.unique_path(tmp_path, "../evil")

    assert path.parent == tmp_path
    assert ".." not in path.name
    assert "\\" not in path.name
    assert "/" not in path.name


def test_matches_keywords() -> None:
    assert window_capture._matches_keywords(
        "明日方舟", "Arknights.exe", window_capture.DEFAULT_KEYWORDS
    )
    assert window_capture._matches_keywords(
        "ARknights", "UnityPlayer.exe", window_capture.DEFAULT_KEYWORDS
    )
    assert not window_capture._matches_keywords(
        "Notepad", "notepad.exe", window_capture.DEFAULT_KEYWORDS
    )


def test_region_area() -> None:
    region = window_capture.Region(10, 20, 30, 40)

    assert region.area == 1200


def test_bundled_wgc_component_is_present() -> None:
    from rtas.tools import _wgc

    assert _wgc.DLL_PATH.is_file()
    assert _wgc.DLL_PATH.with_name("LICENSE").is_file()


@pytest.mark.skipif(sys.platform != "win32", reason="window enumeration is Windows-only")
def test_list_windows_smoke() -> None:
    windows = window_capture.list_windows()

    assert isinstance(windows, list)
    for window in windows:
        assert window.hwnd > 0
        assert window.client_rect.width >= 0
        assert window.client_rect.height >= 0
