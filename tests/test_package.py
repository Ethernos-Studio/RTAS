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

"""Package-level metadata and license compliance checks."""

from __future__ import annotations

from pathlib import Path

import rtas

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "rtas"


def test_package_metadata() -> None:
    assert rtas.__version__ == "1.0.0"
    assert rtas.__author__ == "Ethernos Studio"
    assert rtas.__license__ == "AGPL-3.0-or-later"


def test_every_python_file_carries_the_agpl_header() -> None:
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    assert files, "no Python sources found"

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: AGPL-3.0-or-later" in text, path
        assert "GNU Affero General Public License" in text, path
        assert "Rhodeisland Tactical Autonomous System" in text, path
