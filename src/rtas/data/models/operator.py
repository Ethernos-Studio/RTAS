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
干员数据模型

Author: RTAS Data Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

from .base import ArkDataModel, DataSource


class OperatorProfession(Enum):
    """干员职业"""
    MEDIC = "MEDIC"           # 医疗
    WARRIOR = "WARRIOR"       # 近卫
    SPECIAL = "SPECIAL"       # 特种
    SNIPER = "SNIPER"         # 狙击
    PIONEER = "PIONEER"       # 先锋
    TANK = "TANK"             # 重装
    CASTER = "CASTER"         # 术师
    SUPPORT = "SUPPORT"       # 辅助

    @classmethod
    def from_string(cls, value) -> 'OperatorProfession':
        """从字符串创建"""
        try:
            if isinstance(value, str):
                return cls(value.upper())
            return cls.WARRIOR  # 默认近卫
        except ValueError:
            return cls.WARRIOR  # 默认近卫


class OperatorRarity(Enum):
    """干员稀有度"""
    TIER_1 = 1  # 1星
    TIER_2 = 2  # 2星
    TIER_3 = 3  # 3星
    TIER_4 = 4  # 4星
    TIER_5 = 5  # 5星
    TIER_6 = 6  # 6星

    @classmethod
    def from_string(cls, value) -> 'OperatorRarity':
        """从字符串或数字创建"""
        try:
            # 如果已经是数字，直接转换
            if isinstance(value, int):
                return cls(value)

            # 处理字符串
            if isinstance(value, str):
                value = value.upper().strip()

                # 处理 "TIER_6" 格式 - 提取数字部分
                if value.startswith('TIER_'):
                    try:
                        tier_num = int(value.split('_')[1])
                        return cls(tier_num)
                    except (ValueError, IndexError):
                        pass

                # 尝试直接匹配枚举名
                try:
                    return cls[value]
                except KeyError:
                    pass

                # 尝试从数字字符串解析
                try:
                    return cls(int(value))
                except ValueError:
                    pass

            return cls.TIER_3  # 默认3星
        except (ValueError, TypeError):
            return cls.TIER_3  # 默认3星


class PositionType(Enum):
    """部署位置类型"""
    MELEE = "MELEE"      # 近战位
    RANGED = "RANGED"    # 远程位
    ALL = "ALL"          # 均可
    NONE = "NONE"        # 无（用于装置、陷阱等）

    @classmethod
    def from_string(cls, value: str) -> 'PositionType':
        """从字符串创建，支持未知类型回退到NONE"""
        try:
            return cls(value)
        except ValueError:
            return cls.NONE


@dataclass
class OperatorPhase:
    """干员精英化阶段数据"""
    phase_index: int  # 0=未精英, 1=精一, 2=精二
    max_level: int
    max_hp: int
    atk: int
    def_: int
    magic_resistance: float
    cost: int
    block_count: int
    attack_speed: float
    respawn_time: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'phase_index': self.phase_index,
            'max_level': self.max_level,
            'max_hp': self.max_hp,
            'atk': self.atk,
            'def': self.def_,
            'magic_resistance': self.magic_resistance,
            'cost': self.cost,
            'block_count': self.block_count,
            'attack_speed': self.attack_speed,
            'respawn_time': self.respawn_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperatorPhase':
        return cls(
            phase_index=data.get('phase_index', 0),
            max_level=data.get('max_level', 1),
            max_hp=data.get('max_hp', 0),
            atk=data.get('atk', 0),
            def_=data.get('def', 0),
            magic_resistance=data.get('magic_resistance', 0.0),
            cost=data.get('cost', 0),
            block_count=data.get('block_count', 1),
            attack_speed=data.get('attack_speed', 100.0),
            respawn_time=data.get('respawn_time', 70)
        )


@dataclass
class OperatorSkill:
    """干员技能数据"""
    skill_id: str
    skill_name: str
    description: str
    sp_cost: int
    sp_initial: int
    duration: float
    icon_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'skill_id': self.skill_id,
            'skill_name': self.skill_name,
            'description': self.description,
            'sp_cost': self.sp_cost,
            'sp_initial': self.sp_initial,
            'duration': self.duration,
            'icon_id': self.icon_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperatorSkill':
        return cls(
            skill_id=data.get('skill_id', ''),
            skill_name=data.get('skill_name', ''),
            description=data.get('description', ''),
            sp_cost=data.get('sp_cost', 0),
            sp_initial=data.get('sp_initial', 0),
            duration=data.get('duration', 0.0),
            icon_id=data.get('icon_id', '')
        )


@dataclass
class OperatorTalent:
    """干员天赋数据"""
    talent_id: str
    talent_name: str
    description: str
    unlock_phase: int
    unlock_level: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'talent_id': self.talent_id,
            'talent_name': self.talent_name,
            'description': self.description,
            'unlock_phase': self.unlock_phase,
            'unlock_level': self.unlock_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperatorTalent':
        return cls(
            talent_id=data.get('talent_id', ''),
            talent_name=data.get('talent_name', ''),
            description=data.get('description', ''),
            unlock_phase=data.get('unlock_phase', 0),
            unlock_level=data.get('unlock_level', 1)
        )


@dataclass
class Operator(ArkDataModel):
    """
    干员数据模型

    包含干员的所有核心信息：基础属性、技能、天赋等
    """
    # 基础信息
    appellation: str = ""                    # 英文名/代号
    profession: OperatorProfession = OperatorProfession.WARRIOR
    sub_profession_id: str = ""              # 子职业ID
    rarity: OperatorRarity = OperatorRarity.TIER_3
    position: PositionType = PositionType.MELEE
    description: str = ""

    # 获取信息
    item_usage: str = ""                     # 干员用途描述
    item_desc: str = ""                      # 干员描述
    obtain_approach: str = ""                # 获取方式

    # 标签
    tag_list: List[str] = field(default_factory=list)

    # 属性数据
    phases: List[OperatorPhase] = field(default_factory=list)

    # 技能
    skills: List[OperatorSkill] = field(default_factory=list)

    # 天赋
    talents: List[OperatorTalent] = field(default_factory=list)

    # 潜能
    max_potential_level: int = 5
    potential_item_id: str = ""

    # 其他
    nation_id: str = ""                      # 势力/国家ID
    group_id: Optional[str] = None           # 组织ID
    team_id: Optional[str] = None            # 小队ID
    display_number: str = ""                 # 编号
    is_not_obtainable: bool = False          # 是否无法获取
    is_sp_char: bool = False                 # 是否为异格干员

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base.update({
            'appellation': self.appellation,
            'profession': self.profession.value,
            'sub_profession_id': self.sub_profession_id,
            'rarity': self.rarity.value,
            'position': self.position.value,
            'description': self.description,
            'item_usage': self.item_usage,
            'item_desc': self.item_desc,
            'obtain_approach': self.obtain_approach,
            'tag_list': self.tag_list,
            'phases': [p.to_dict() for p in self.phases],
            'skills': [s.to_dict() for s in self.skills],
            'talents': [t.to_dict() for t in self.talents],
            'max_potential_level': self.max_potential_level,
            'potential_item_id': self.potential_item_id,
            'nation_id': self.nation_id,
            'group_id': self.group_id,
            'team_id': self.team_id,
            'display_number': self.display_number,
            'is_not_obtainable': self.is_not_obtainable,
            'is_sp_char': self.is_sp_char
        })
        return base

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operator':
        """从字典创建"""
        base = ArkDataModel.from_dict(data)

        return cls(
            id=base.id,
            name=base.name,
            source=base.source,
            version=base.version,
            metadata=base.metadata,
            appellation=data.get('appellation', ''),
            profession=OperatorProfession.from_string(data.get('profession', 'WARRIOR')),
            sub_profession_id=data.get('sub_profession_id', ''),
            rarity=OperatorRarity.from_string(data.get('rarity', 'TIER_3')),
            position=PositionType.from_string(data.get('position', 'MELEE')),
            description=data.get('description', ''),
            item_usage=data.get('item_usage', ''),
            item_desc=data.get('item_desc', ''),
            obtain_approach=data.get('obtain_approach', ''),
            tag_list=data.get('tag_list', []),
            phases=[OperatorPhase.from_dict(p) for p in data.get('phases', [])],
            skills=[OperatorSkill.from_dict(s) for s in data.get('skills', [])],
            talents=[OperatorTalent.from_dict(t) for t in data.get('talents', [])],
            max_potential_level=data.get('max_potential_level', 5),
            potential_item_id=data.get('potential_item_id', ''),
            nation_id=data.get('nation_id', ''),
            group_id=data.get('group_id'),
            team_id=data.get('team_id'),
            display_number=data.get('display_number', ''),
            is_not_obtainable=data.get('is_not_obtainable', False),
            is_sp_char=data.get('is_sp_char', False),
            raw_data=data.get('raw_data', {})
        )

    @property
    def is_melee(self) -> bool:
        """是否为近战位"""
        return self.position == PositionType.MELEE

    @property
    def is_ranged(self) -> bool:
        """是否为远程位"""
        return self.position == PositionType.RANGED

    @property
    def stars(self) -> int:
        """星级（1-6）"""
        return self.rarity.value

    def get_phase(self, phase_index: int) -> Optional[OperatorPhase]:
        """获取指定精英化阶段的属性"""
        for phase in self.phases:
            if phase.phase_index == phase_index:
                return phase
        return None

    def get_max_attributes(self) -> Optional[OperatorPhase]:
        """获取最高阶段的属性"""
        if not self.phases:
            return None
        return max(self.phases, key=lambda p: p.phase_index)
