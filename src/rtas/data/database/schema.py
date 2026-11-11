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
数据库Schema定义

定义所有表结构和索引

Author: RTAS Data Team
Version: 1.0.0
"""

from typing import List, Tuple


class DatabaseSchema:
    """数据库Schema定义"""

    # 干员主表
    OPERATORS_TABLE = """
    CREATE TABLE IF NOT EXISTS operators (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        appellation TEXT,
        profession TEXT NOT NULL,
        rarity INTEGER NOT NULL,
        position TEXT,
        tag_list TEXT,  -- JSON数组
        description TEXT,
        max_phases INTEGER DEFAULT 0,
        can_use_general_potential_item BOOLEAN DEFAULT 0,
        can_use_activity_potential_item BOOLEAN DEFAULT 0,
        potential_item_id TEXT,
        activity_potential_item_id TEXT,
        team_id INTEGER,
        display_number TEXT,
        group_id TEXT,
        nation_id TEXT,
        is_not_obtainable BOOLEAN DEFAULT 0,
        is_sp_char BOOLEAN DEFAULT 0,
        is_robot BOOLEAN DEFAULT 0,
        source TEXT,
        version TEXT,
        raw_data TEXT,  -- 完整原始数据JSON
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    # 干员精英化阶段表
    OPERATOR_PHASES_TABLE = """
    CREATE TABLE IF NOT EXISTS operator_phases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operator_id TEXT NOT NULL,
        phase_index INTEGER NOT NULL,
        max_level INTEGER NOT NULL,
        max_hp INTEGER NOT NULL,
        atk INTEGER NOT NULL,
        def INTEGER NOT NULL,
        magic_resistance REAL NOT NULL,
        cost INTEGER NOT NULL,
        block_count INTEGER NOT NULL,
        attack_speed REAL NOT NULL,
        respawn_time INTEGER NOT NULL,
        FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE,
        UNIQUE(operator_id, phase_index)
    )
    """

    # 干员技能表
    OPERATOR_SKILLS_TABLE = """
    CREATE TABLE IF NOT EXISTS operator_skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operator_id TEXT NOT NULL,
        skill_id TEXT NOT NULL,
        skill_name TEXT,
        description TEXT,
        sp_cost INTEGER DEFAULT 0,
        sp_initial INTEGER DEFAULT 0,
        duration REAL DEFAULT 0,
        skill_type TEXT,
        FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE
    )
    """

    # 干员天赋表
    OPERATOR_TALENTS_TABLE = """
    CREATE TABLE IF NOT EXISTS operator_talents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operator_id TEXT NOT NULL,
        talent_id TEXT,
        talent_name TEXT,
        description TEXT,
        unlock_phase INTEGER DEFAULT 0,
        unlock_level INTEGER DEFAULT 1,
        required_potential_rank INTEGER DEFAULT 0,
        FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE
    )
    """

    # 干员潜能表
    OPERATOR_POTENTIALS_TABLE = """
    CREATE TABLE IF NOT EXISTS operator_potentials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        operator_id TEXT NOT NULL,
        rank INTEGER NOT NULL,
        description TEXT,
        buff_id TEXT,
        FOREIGN KEY (operator_id) REFERENCES operators(id) ON DELETE CASCADE,
        UNIQUE(operator_id, rank)
    )
    """

    # 关卡主表
    STAGES_TABLE = """
    CREATE TABLE IF NOT EXISTS stages (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        code TEXT,
        stage_type TEXT NOT NULL,
        difficulty TEXT,
        description TEXT,
        zone_id TEXT,
        zone_name TEXT,
        level_id TEXT,
        ap_cost INTEGER DEFAULT 0,
        ap_fail_return INTEGER DEFAULT 0,
        exp_gain INTEGER DEFAULT 0,
        gold_gain INTEGER DEFAULT 0,
        can_practice BOOLEAN DEFAULT 1,
        can_battle_replay BOOLEAN DEFAULT 1,
        can_continuous_battle BOOLEAN DEFAULT 0,
        is_hard_stage BOOLEAN DEFAULT 0,
        is_story_only BOOLEAN DEFAULT 0,
        is_stage_patch BOOLEAN DEFAULT 0,
        danger_level TEXT,
        loading_pic_id TEXT,
        main_stage_id TEXT,
        boss_id TEXT,
        source TEXT,
        version TEXT,
        raw_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    # 关卡掉落表
    STAGE_DROPS_TABLE = """
    CREATE TABLE IF NOT EXISTS stage_drops (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        item_name TEXT,
        drop_type TEXT NOT NULL,  -- ONCE, NORMAL, COMPLETE, ADDITIONAL, SPECIAL
        occ_percent TEXT,  -- ALWAYS, SOMETIMES, ALMOST
        count INTEGER DEFAULT 1,
        weight INTEGER DEFAULT 0,
        FOREIGN KEY (stage_id) REFERENCES stages(id) ON DELETE CASCADE
    )
    """

    # 关卡解锁条件表
    STAGE_CONDITIONS_TABLE = """
    CREATE TABLE IF NOT EXISTS stage_conditions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stage_id TEXT NOT NULL,
        condition_type TEXT NOT NULL,  -- STAGE, LEVEL, etc.
        condition_value TEXT NOT NULL,
        FOREIGN KEY (stage_id) REFERENCES stages(id) ON DELETE CASCADE
    )
    """

    # 物品主表
    ITEMS_TABLE = """
    CREATE TABLE IF NOT EXISTS items (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        item_type TEXT NOT NULL,
        description TEXT,
        rarity INTEGER DEFAULT 0,
        icon_id TEXT,
        usage TEXT,
        obtain_approach TEXT,
        classify_type TEXT,
        item_usage TEXT,
        sort_id INTEGER DEFAULT 0,
        price INTEGER DEFAULT 0,
        max_count INTEGER DEFAULT 0,
        max_usage_count INTEGER DEFAULT 0,
        stack_num INTEGER DEFAULT 0,
        is_consumable BOOLEAN DEFAULT 0,
        is_gift BOOLEAN DEFAULT 0,
        is_furniture BOOLEAN DEFAULT 0,
        is_material BOOLEAN DEFAULT 0,
        is_exp_card BOOLEAN DEFAULT 0,
        is_character BOOLEAN DEFAULT 0,
        source TEXT,
        version TEXT,
        raw_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    # 物品合成配方表
    ITEM_RECIPES_TABLE = """
    CREATE TABLE IF NOT EXISTS item_recipes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        item_id TEXT NOT NULL,
        cost_gold INTEGER DEFAULT 0,
        FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
    )
    """

    # 配方材料表
    RECIPE_MATERIALS_TABLE = """
    CREATE TABLE IF NOT EXISTS recipe_materials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipe_id INTEGER NOT NULL,
        item_id TEXT NOT NULL,
        item_name TEXT,
        count INTEGER NOT NULL,
        FOREIGN KEY (recipe_id) REFERENCES item_recipes(id) ON DELETE CASCADE,
        FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
    )
    """

    # 敌人主表
    ENEMIES_TABLE = """
    CREATE TABLE IF NOT EXISTS enemies (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        enemy_level TEXT NOT NULL,  -- NORMAL, ELITE, BOSS
        description TEXT,
        max_hp INTEGER DEFAULT 0,
        atk INTEGER DEFAULT 0,
        def INTEGER DEFAULT 0,
        magic_resistance REAL DEFAULT 0,
        move_speed REAL DEFAULT 1.0,
        attack_speed REAL DEFAULT 100.0,
        base_attack_time REAL DEFAULT 1.0,
        hp_recovery_per_sec REAL DEFAULT 0.0,
        sp_recovery_per_sec REAL DEFAULT 0.0,
        mass_level INTEGER DEFAULT 0,
        taunt_level INTEGER DEFAULT 0,
        ep_damage_resistance REAL DEFAULT 0,
        ep_resistance REAL DEFAULT 0,
        stun_immune BOOLEAN DEFAULT 0,
        silence_immune BOOLEAN DEFAULT 0,
        sleep_immune BOOLEAN DEFAULT 0,
        frozen_immune BOOLEAN DEFAULT 0,
        levitate_immune BOOLEAN DEFAULT 0,
        disarmed_combat_immune BOOLEAN DEFAULT 0,
        feared_immune BOOLEAN DEFAULT 0,
        icon_id TEXT,
        sort_id INTEGER DEFAULT 0,
        source TEXT,
        version TEXT,
        raw_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    # 敌人能力表
    ENEMY_ABILITIES_TABLE = """
    CREATE TABLE IF NOT EXISTS enemy_abilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enemy_id TEXT NOT NULL,
        ability_id TEXT,
        ability_name TEXT,
        description TEXT,
        icon_id TEXT,
        FOREIGN KEY (enemy_id) REFERENCES enemies(id) ON DELETE CASCADE
    )
    """

    # 敌人攻击模式表
    ENEMY_ATTACKS_TABLE = """
    CREATE TABLE IF NOT EXISTS enemy_attacks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enemy_id TEXT NOT NULL,
        attack_type TEXT,
        damage_type TEXT,
        attack_range REAL,
        attack_speed REAL,
        attack_times INTEGER DEFAULT 1,
        FOREIGN KEY (enemy_id) REFERENCES enemies(id) ON DELETE CASCADE
    )
    """

    # 数据版本表
    VERSION_INFO_TABLE = """
    CREATE TABLE IF NOT EXISTS version_info (
        source TEXT PRIMARY KEY,
        version TEXT NOT NULL,
        commit_hash TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    # 索引定义
    INDEXES: List[Tuple[str, str, str]] = [
        # 干员索引
        ('idx_operators_profession', 'operators', 'profession'),
        ('idx_operators_rarity', 'operators', 'rarity'),
        ('idx_operators_position', 'operators', 'position'),
        ('idx_operators_name', 'operators', 'name'),
        ('idx_operators_nation', 'operators', 'nation_id'),
        ('idx_operators_team', 'operators', 'team_id'),

        # 干员关联表索引
        ('idx_op_phases_operator', 'operator_phases', 'operator_id'),
        ('idx_op_skills_operator', 'operator_skills', 'operator_id'),
        ('idx_op_talents_operator', 'operator_talents', 'operator_id'),
        ('idx_op_potentials_operator', 'operator_potentials', 'operator_id'),

        # 关卡索引
        ('idx_stages_type', 'stages', 'stage_type'),
        ('idx_stages_code', 'stages', 'code'),
        ('idx_stages_zone', 'stages', 'zone_id'),
        ('idx_stages_name', 'stages', 'name'),
        ('idx_stages_difficulty', 'stages', 'difficulty'),

        # 关卡关联表索引
        ('idx_stage_drops_stage', 'stage_drops', 'stage_id'),
        ('idx_stage_drops_item', 'stage_drops', 'item_id'),
        ('idx_stage_conditions_stage', 'stage_conditions', 'stage_id'),

        # 物品索引
        ('idx_items_type', 'items', 'item_type'),
        ('idx_items_rarity', 'items', 'rarity'),
        ('idx_items_name', 'items', 'name'),
        ('idx_items_material', 'items', 'is_material'),

        # 物品关联表索引
        ('idx_item_recipes_item', 'item_recipes', 'item_id'),
        ('idx_recipe_materials_recipe', 'recipe_materials', 'recipe_id'),
        ('idx_recipe_materials_item', 'recipe_materials', 'item_id'),

        # 敌人索引
        ('idx_enemies_level', 'enemies', 'enemy_level'),
        ('idx_enemies_name', 'enemies', 'name'),
        ('idx_enemies_sort', 'enemies', 'sort_id'),

        # 敌人关联表索引
        ('idx_enemy_abilities_enemy', 'enemy_abilities', 'enemy_id'),
        ('idx_enemy_attacks_enemy', 'enemy_attacks', 'enemy_id'),
    ]

    # 所有表的创建顺序（考虑外键依赖）
    TABLES = [
        OPERATORS_TABLE,
        OPERATOR_PHASES_TABLE,
        OPERATOR_SKILLS_TABLE,
        OPERATOR_TALENTS_TABLE,
        OPERATOR_POTENTIALS_TABLE,
        STAGES_TABLE,
        STAGE_DROPS_TABLE,
        STAGE_CONDITIONS_TABLE,
        ITEMS_TABLE,
        ITEM_RECIPES_TABLE,
        RECIPE_MATERIALS_TABLE,
        ENEMIES_TABLE,
        ENEMY_ABILITIES_TABLE,
        ENEMY_ATTACKS_TABLE,
        VERSION_INFO_TABLE,
    ]

    @classmethod
    def get_create_statements(cls) -> List[str]:
        """获取所有建表语句"""
        return cls.TABLES

    @classmethod
    def get_index_statements(cls) -> List[str]:
        """获取所有索引创建语句"""
        statements = []
        for index_name, table, column in cls.INDEXES:
            statements.append(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column})"
            )
        return statements
