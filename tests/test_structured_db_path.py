# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Structured database path configuration tests."""

from __future__ import annotations

import logging
from pathlib import Path

from rtas.cli import DataCommands
from rtas.data.providers.data_manager import CacheConfig, DataManager, ManagerConfig


def test_cli_sources_share_workspace_structured_database() -> None:
    expected_path = Path("cache") / "rtas_structured.db"

    for source in ("assets", "kengxxiao"):
        commands = DataCommands(logging.getLogger(__name__), source=source)
        config = commands._create_manager_config()

        assert config.cache.structured_db_path == expected_path


def test_custom_structured_db_path_overrides_cache_directory(tmp_path: Path) -> None:
    structured_db_path = tmp_path / "shared" / "rtas_structured.db"
    manager = DataManager(
        ManagerConfig(
            cache=CacheConfig(
                cache_dir=tmp_path / "source-cache",
                structured_db_path=structured_db_path,
            )
        )
    )

    manager._init_structured_database()
    try:
        assert manager.structured_db is not None
        assert manager.structured_db.db_path == structured_db_path
    finally:
        manager.shutdown()
