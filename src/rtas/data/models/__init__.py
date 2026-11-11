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
明日方舟数据模型定义

包含干员、关卡、物品等核心数据模型
"""

from .operator import Operator, OperatorSkill, OperatorTalent, OperatorPhase
from .stage import Stage, StageDrop, StageCondition
from .item import Item, ItemType
from .enemy import Enemy, EnemyAbility
from .base import ArkDataModel, DataSource, DataVersion

__all__ = [
    # 干员相关
    'Operator',
    'OperatorSkill',
    'OperatorTalent',
    'OperatorPhase',
    # 关卡相关
    'Stage',
    'StageDrop',
    'StageCondition',
    # 物品相关
    'Item',
    'ItemType',
    # 敌人相关
    'Enemy',
    'EnemyAbility',
    # 基础模型
    'ArkDataModel',
    'DataSource',
    'DataVersion',
]
