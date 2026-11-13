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
# @brief 关卡推演包：敌人走位、阻挡与输出的简单模拟

"""关卡推演（sim）。

外部库用法::

    from rtas.sim import OperatorSpec, Plan, Simulator, load_level

    level = load_level("ArknightsGameData/zh_CN/gamedata/levels/obt/main/level_main_01-07.json")

    # 1) 只看走位：某个时刻敌人都在哪（无阻挡）
    sim = Simulator(level)
    result = sim.run(duration=90)
    result.save_snapshot("t30.png", time=30)
    print(result.frame_at(30).as_dict())

    # 2) 放一个阻挡 1 的近卫，看击杀/漏怪与视频
    plan = Plan(duration=120, fps=20)
    plan.add(OperatorSpec(name="近卫", tile=(3, 2), deploy_time=5,
                          block=1, damage=600, attack_interval=1.0))
    result = Simulator(level, plan).run()
    result.export_video("run.mp4")
    result.save_json("run.json")

命令行入口见 ``rtas sim map / run / tiles``。
"""

from .engine import SimulationResult, Simulator, simulate
from .loader import LevelDataError, load_level, load_plan, parse_operator_spec
from .model import (
    AttackMode,
    Buildable,
    EnemyFrame,
    EnemyProfile,
    Frame,
    OperatorFrame,
    OperatorSpec,
    Plan,
    Route,
    SimLevel,
    SpawnEvent,
    TileCell,
    TileKind,
)
from .render import DEFAULT_CELL, MapRenderer, export_video, save_snapshot

__all__ = [
    "AttackMode",
    "Buildable",
    "DEFAULT_CELL",
    "EnemyFrame",
    "EnemyProfile",
    "Frame",
    "LevelDataError",
    "MapRenderer",
    "OperatorFrame",
    "OperatorSpec",
    "Plan",
    "Route",
    "SimLevel",
    "SimulationResult",
    "Simulator",
    "simulate",
    "SpawnEvent",
    "TileCell",
    "TileKind",
    "export_video",
    "load_level",
    "load_plan",
    "parse_operator_spec",
    "save_snapshot",
]
