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
敌人数据模型

Author: RTAS Data Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .base import ArkDataModel


class EnemyLevel(Enum):
    """敌人等级"""
    NORMAL = "NORMAL"
    ELITE = "ELITE"
    BOSS = "BOSS"


@dataclass
class EnemyAbility:
    """敌人能力/特性"""
    ability_id: str
    ability_name: str
    description: str
    icon_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'ability_id': self.ability_id,
            'ability_name': self.ability_name,
            'description': self.description,
            'icon_id': self.icon_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnemyAbility':
        return cls(
            ability_id=data.get('ability_id', ''),
            ability_name=data.get('ability_name', ''),
            description=data.get('description', ''),
            icon_id=data.get('icon_id', '')
        )


@dataclass
class Enemy(ArkDataModel):
    """
    敌人数据模型

    包含敌人的所有核心信息
    """
    # 基础信息
    enemy_level: EnemyLevel = EnemyLevel.NORMAL
    description: str = ""

    # 属性
    max_hp: int = 0
    atk: int = 0
    def_: int = 0
    magic_resistance: float = 0.0
    move_speed: float = 1.0
    attack_speed: float = 100.0
    base_attack_time: float = 1.0

    # 其他属性
    hp_recovery_per_sec: float = 0.0
    mass_level: int = 0
    taunt_level: int = 0

    # 免疫状态
    stun_immune: bool = False
    silence_immune: bool = False
    sleep_immune: bool = False
    frozen_immune: bool = False

    # 能力
    abilities: List[EnemyAbility] = field(default_factory=list)

    # 图标
    icon_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base.update({
            'enemy_level': self.enemy_level.value,
            'description': self.description,
            'max_hp': self.max_hp,
            'atk': self.atk,
            'def': self.def_,
            'magic_resistance': self.magic_resistance,
            'move_speed': self.move_speed,
            'attack_speed': self.attack_speed,
            'base_attack_time': self.base_attack_time,
            'hp_recovery_per_sec': self.hp_recovery_per_sec,
            'mass_level': self.mass_level,
            'taunt_level': self.taunt_level,
            'stun_immune': self.stun_immune,
            'silence_immune': self.silence_immune,
            'sleep_immune': self.sleep_immune,
            'frozen_immune': self.frozen_immune,
            'abilities': [a.to_dict() for a in self.abilities],
            'icon_id': self.icon_id
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Enemy':
        """从字典创建"""
        base = ArkDataModel.from_dict(data)

        return cls(
            id=base.id,
            name=base.name,
            source=base.source,
            version=base.version,
            metadata=base.metadata,
            enemy_level=EnemyLevel(data.get('enemy_level', 'NORMAL')),
            description=data.get('description', ''),
            max_hp=data.get('max_hp', 0),
            atk=data.get('atk', 0),
            def_=data.get('def', 0),
            magic_resistance=data.get('magic_resistance', 0.0),
            move_speed=data.get('move_speed', 1.0),
            attack_speed=data.get('attack_speed', 100.0),
            base_attack_time=data.get('base_attack_time', 1.0),
            hp_recovery_per_sec=data.get('hp_recovery_per_sec', 0.0),
            mass_level=data.get('mass_level', 0),
            taunt_level=data.get('taunt_level', 0),
            stun_immune=data.get('stun_immune', False),
            silence_immune=data.get('silence_immune', False),
            sleep_immune=data.get('sleep_immune', False),
            frozen_immune=data.get('frozen_immune', False),
            abilities=[EnemyAbility.from_dict(a) for a in data.get('abilities', [])],
            icon_id=data.get('icon_id', ''),
            raw_data=data.get('raw_data', {})
        )

    @property
    def is_boss(self) -> bool:
        """是否为Boss"""
        return self.enemy_level == EnemyLevel.BOSS

    @property
    def is_elite(self) -> bool:
        """是否为精英"""
        return self.enemy_level == EnemyLevel.ELITE

    def get_dps(self) -> float:
        """计算每秒伤害"""
        if self.base_attack_time > 0:
            return self.atk / self.base_attack_time * (self.attack_speed / 100)
        return 0.0

    def get_effective_hp(self) -> float:
        """计算等效生命值（考虑防御和法抗）"""
        physical_ehp = self.max_hp * (1 + self.def_ / 100)
        magic_ehp = self.max_hp * (1 + self.magic_resistance / 100)
        return max(physical_ehp, magic_ehp)
