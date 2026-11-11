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

"""
RTAS 数据模块

提供游戏数据的同步、查询和管理功能

Author: RTAS Data Team
Version: 1.0.0
"""

from .providers import DataManager, GitHubDataProvider, PRTSDataProvider
from .providers.data_manager import ManagerConfig, CacheConfig
from .providers.github_provider import SyncConfig
from .providers.prts_provider import PRTSConfig
from .models import (
    Operator, OperatorSkill, OperatorTalent, OperatorPhase,
    Stage, StageDrop, StageCondition,
    Item, ItemType,
    Enemy, EnemyAbility,
    ArkDataModel, DataSource, DataVersion
)

__version__ = "1.0.0"

__all__ = [
    # 数据管理器
    'DataManager',
    'ManagerConfig',
    'CacheConfig',
    # 数据提供者
    'GitHubDataProvider',
    'SyncConfig',
    'PRTSDataProvider',
    'PRTSConfig',
    # 数据模型
    'Operator',
    'OperatorSkill',
    'OperatorTalent',
    'OperatorPhase',
    'Stage',
    'StageDrop',
    'StageCondition',
    'Item',
    'ItemType',
    'Enemy',
    'EnemyAbility',
    'ArkDataModel',
    'DataSource',
    'DataVersion',
]
