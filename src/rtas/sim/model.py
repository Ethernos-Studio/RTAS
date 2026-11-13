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
# @brief 关卡推演的数据模型（与渲染、GUI 解耦）

"""关卡推演数据模型。

这里只放「纯数据」：地块、路线、敌人档案、干员配置、帧快照。
不依赖 OpenCV / PIL，方便被外部库直接引用，也方便单元测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class TileKind(str, Enum):
    """地块类型（渲染用）。"""

    FORBIDDEN = "forbidden"  # 地图外 / 不可通行
    FLOOR = "floor"          # 可部署地面
    ROAD = "road"            # 敌人路径（通常不可部署）
    WALL = "wall"            # 墙 / 高台侧面
    START = "start"          # 红门（敌人出生）
    END = "end"              # 蓝门（保护目标）
    HOLE = "hole"            # 地穴


class Buildable(str, Enum):
    """可部署类型。"""

    NONE = "none"
    MELEE = "melee"
    RANGED = "ranged"
    ALL = "all"


@dataclass(frozen=True)
class TileCell:
    """单个地块。"""

    kind: TileKind = TileKind.FORBIDDEN
    buildable: Buildable = Buildable.NONE
    height: int = 0          # 0=地面, 1=高台
    walkable: bool = False   # 地面敌人是否可通行
    flyable: bool = False    # 飞行敌人是否可通行

    def can_deploy(self, mode: str = "melee") -> bool:
        """是否可以部署指定类型的干员。"""
        if self.buildable is Buildable.NONE:
            return False
        if self.buildable is Buildable.ALL:
            return True
        return self.buildable.value == mode


_FORBIDDEN = TileCell()


@dataclass
class EnemyProfile:
    """敌人档案（决定血量、速度、防御与阻挡权重）。"""

    enemy_id: str
    name: str = ""
    hp: float = 1000.0
    speed: float = 0.7          # 格/秒（已乘关卡 move_multiplier）
    defense: float = 0.0
    weight: float = 1.0         # 占用阻挡数
    flying: bool = False
    leak_damage: int = 1        # 漏怪扣的生命值

    def as_dict(self) -> dict[str, Any]:
        return {
            "enemy_id": self.enemy_id,
            "name": self.name or self.enemy_id,
            "hp": self.hp,
            "speed": round(self.speed, 4),
            "defense": self.defense,
            "weight": self.weight,
            "flying": self.flying,
            "leak_damage": self.leak_damage,
        }


@dataclass
class Route:
    """敌人行进路线：顶点序列（行, 列，浮点，单位=格）。"""

    route_id: str
    points: list[tuple[float, float]] = field(default_factory=list)
    flying: bool = False
    _cumulative: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._recompute()

    def _recompute(self) -> None:
        self._cumulative = [0.0]
        total = 0.0
        for index in range(len(self.points) - 1):
            row0, col0 = self.points[index]
            row1, col1 = self.points[index + 1]
            total += math.hypot(row1 - row0, col1 - col0)
            self._cumulative.append(total)

    @property
    def length(self) -> float:
        return self._cumulative[-1] if self._cumulative else 0.0

    def position_at(self, distance: float) -> tuple[float, float]:
        """按已行进距离取插值坐标（行, 列）。"""
        if not self.points:
            return (0.0, 0.0)
        if distance <= 0:
            return self.points[0]
        if distance >= self.length:
            return self.points[-1]

        for index in range(len(self.points) - 1):
            start = self._cumulative[index]
            end = self._cumulative[index + 1]
            if distance <= end:
                span = end - start
                ratio = 0.0 if span <= 0 else (distance - start) / span
                row0, col0 = self.points[index]
                row1, col1 = self.points[index + 1]
                return (row0 + (row1 - row0) * ratio, col0 + (col1 - col0) * ratio)
        return self.points[-1]

    def tile_at(self, distance: float) -> tuple[int, int]:
        row, col = self.position_at(distance)
        return (int(round(row)), int(round(col)))

    def distance_at(self, row: float, col: float) -> float:
        """路径上距离给定点最近的弧长（用于把被阻挡的敌人吸附到格心）。"""
        if len(self.points) < 2:
            return 0.0
        best_error = float("inf")
        best_distance = 0.0
        for index in range(len(self.points) - 1):
            row0, col0 = self.points[index]
            row1, col1 = self.points[index + 1]
            seg_len = self._cumulative[index + 1] - self._cumulative[index]
            if seg_len <= 0:
                continue
            delta_row, delta_col = row1 - row0, col1 - col0
            ratio = ((row - row0) * delta_row + (col - col0) * delta_col) / (seg_len ** 2)
            ratio = max(0.0, min(1.0, ratio))
            proj_row, proj_col = row0 + delta_row * ratio, col0 + delta_col * ratio
            error = math.hypot(proj_row - row, proj_col - col)
            if error < best_error:
                best_error = error
                best_distance = self._cumulative[index] + ratio * seg_len
        return best_distance

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "length": round(self.length, 3),
            "points": [[round(r, 2), round(c, 2)] for r, c in self.points],
        }


@dataclass
class SpawnEvent:
    """一次敌人生成（绝对时间，秒）。"""

    time: float
    enemy_id: str
    route_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "enemy_id": self.enemy_id,
            "route_id": self.route_id,
        }


class AttackMode(str, Enum):
    """攻击方式。"""

    MELEE = "melee"    # 只攻击自己阻挡到的敌人
    RANGED = "ranged"  # 攻击射程内最靠前的敌人（不需要阻挡）


@dataclass
class OperatorSpec:
    """干员配置：什么时候放、放哪、挡几个、怎么打。"""

    name: str = "干员"
    tile: tuple[int, int] = (0, 0)
    deploy_time: float = 0.0
    retreat_time: float | None = None
    block: int = 1               # 阻挡数
    damage: float = 500.0        # 每次攻击伤害（物理）
    attack_interval: float = 1.0  # 攻击间隔（秒）
    attack_mode: AttackMode = AttackMode.MELEE
    range_tiles: float = 1.5     # 远程攻击半径（格）

    def __post_init__(self) -> None:
        self.tile = (int(self.tile[0]), int(self.tile[1]))
        if isinstance(self.attack_mode, str):
            self.attack_mode = AttackMode(self.attack_mode)
        if self.block < 0:
            self.block = 0
        if self.attack_interval <= 0:
            raise ValueError("attack_interval 必须大于 0")

    @property
    def dps(self) -> float:
        return self.damage / self.attack_interval

    def is_active(self, time: float) -> bool:
        if time < self.deploy_time:
            return False
        if self.retreat_time is not None and time >= self.retreat_time:
            return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tile": list(self.tile),
            "deploy_time": self.deploy_time,
            "retreat_time": self.retreat_time,
            "block": self.block,
            "damage": self.damage,
            "attack_interval": self.attack_interval,
            "attack_mode": self.attack_mode.value,
            "range_tiles": self.range_tiles,
            "dps": round(self.dps, 2),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperatorSpec:
        tile = data.get("tile") or data.get("position") or (0, 0)
        return cls(
            name=str(data.get("name", "干员")),
            tile=(int(tile[0]), int(tile[1])),
            deploy_time=float(data.get("deploy_time", data.get("time", 0.0))),
            retreat_time=(None if data.get("retreat_time") is None
                          else float(data["retreat_time"])),
            block=int(data.get("block", 1)),
            damage=float(data.get("damage", data.get("atk", 500.0))),
            attack_interval=float(data.get("attack_interval", data.get("interval", 1.0))),
            attack_mode=AttackMode(str(data.get("attack_mode", data.get("mode", "melee")))),
            range_tiles=float(data.get("range_tiles", data.get("range", 1.5))),
        )


@dataclass
class Plan:
    """一次推演的配置：干员计划 + 时长/步长/帧率。"""

    operators: list[OperatorSpec] = field(default_factory=list)
    duration: float = 240.0
    dt: float = 0.1
    fps: float = 20.0
    speed_scale: float = 1.0   # 全局速度倍率（>1 更快）

    def add(self, operator: OperatorSpec) -> Plan:
        self.operators.append(operator)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration": self.duration,
            "dt": self.dt,
            "fps": self.fps,
            "speed_scale": self.speed_scale,
            "operators": [op.as_dict() for op in self.operators],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        plan = cls(
            duration=float(data.get("duration", 240.0)),
            dt=float(data.get("dt", 0.1)),
            fps=float(data.get("fps", 20.0)),
            speed_scale=float(data.get("speed_scale", 1.0)),
        )
        for item in data.get("operators", []):
            plan.operators.append(OperatorSpec.from_dict(item))
        return plan


@dataclass
class SimLevel:
    """推演用的关卡：地块网格 + 路线 + 生成时间轴 + 敌人档案。"""

    level_id: str
    name: str = ""
    grid: list[list[TileCell]] = field(default_factory=list)
    routes: dict[str, Route] = field(default_factory=dict)
    spawns: list[SpawnEvent] = field(default_factory=list)
    enemies: dict[str, EnemyProfile] = field(default_factory=dict)
    life: int = 10
    move_multiplier: float = 1.0
    source: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- 基本属性 ---------------------------------------------------------
    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def total_enemies(self) -> int:
        return len(self.spawns)

    @property
    def last_spawn_time(self) -> float:
        return self.spawns[-1].time if self.spawns else 0.0

    def tile(self, row: int, col: int) -> TileCell:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        return _FORBIDDEN

    def is_walkable(self, row: int, col: int, flying: bool = False) -> bool:
        cell = self.tile(row, col)
        return cell.flyable if flying else cell.walkable

    def deployable(self, mode: str = "any") -> list[tuple[int, int]]:
        """所有可部署地块（mode: melee/ranged/any）。"""
        result: list[tuple[int, int]] = []
        for row, line in enumerate(self.grid):
            for col, cell in enumerate(line):
                if mode == "any":
                    if cell.buildable is not Buildable.NONE:
                        result.append((row, col))
                elif cell.can_deploy(mode):
                    result.append((row, col))
        return result

    def profile(self, enemy_id: str) -> EnemyProfile:
        """取敌人档案；未知敌人给一份可用的默认值。"""
        found = self.enemies.get(enemy_id)
        if found is not None:
            return found
        default = EnemyProfile(enemy_id=enemy_id, name=enemy_id)
        self.enemies[enemy_id] = default
        return default

    def routes_for(self, enemy_id: str) -> list[Route]:
        route_ids = {spawn.route_id for spawn in self.spawns if spawn.enemy_id == enemy_id}
        return [self.routes[rid] for rid in route_ids if rid in self.routes]

    # -- 导出 -------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "life": self.life,
            "move_multiplier": self.move_multiplier,
            "total_enemies": self.total_enemies,
            "last_spawn_time": round(self.last_spawn_time, 3),
            "routes": [route.as_dict() for route in self.routes.values()],
            "spawns": [spawn.as_dict() for spawn in self.spawns],
            "enemies": [profile.as_dict() for profile in self.enemies.values()],
        }

    def iter_route_points(self) -> Iterable[Sequence[tuple[float, float]]]:
        for route in self.routes.values():
            yield route.points


# =============================================================================
# 帧快照
# =============================================================================

@dataclass
class EnemyFrame:
    """某一帧里的一个敌人。"""

    uid: int
    enemy_id: str
    name: str
    row: float
    col: float
    hp: float
    max_hp: float
    blocked_by: int | None = None
    flying: bool = False
    progress: float = 0.0
    weight: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "enemy_id": self.enemy_id,
            "name": self.name,
            "row": round(self.row, 3),
            "col": round(self.col, 3),
            "hp": round(self.hp, 1),
            "max_hp": self.max_hp,
            "blocked_by": self.blocked_by,
            "flying": self.flying,
            "progress": round(self.progress, 4),
            "weight": self.weight,
        }


@dataclass
class OperatorFrame:
    """某一帧里的一个干员。"""

    index: int
    name: str
    row: int
    col: int
    active: bool
    blocked: int
    capacity: int
    kills: int = 0
    damage: float = 0.0
    attack_mode: str = "melee"

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "row": self.row,
            "col": self.col,
            "active": self.active,
            "blocked": self.blocked,
            "capacity": self.capacity,
            "kills": self.kills,
            "damage": round(self.damage, 1),
            "attack_mode": self.attack_mode,
        }


@dataclass
class Frame:
    """某个时刻的完整画面状态。"""

    time: float
    enemies: list[EnemyFrame] = field(default_factory=list)
    operators: list[OperatorFrame] = field(default_factory=list)
    killed: int = 0
    leaked: int = 0
    spawned: int = 0
    life: int = 0
    max_life: int = 0

    @property
    def blocked(self) -> int:
        return sum(1 for enemy in self.enemies if enemy.blocked_by is not None)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "killed": self.killed,
            "leaked": self.leaked,
            "spawned": self.spawned,
            "life": self.life,
            "max_life": self.max_life,
            "alive": len(self.enemies),
            "blocked": self.blocked,
            "enemies": [enemy.as_dict() for enemy in self.enemies],
            "operators": [op.as_dict() for op in self.operators],
        }


__all__ = [
    "AttackMode",
    "Buildable",
    "EnemyFrame",
    "EnemyProfile",
    "Frame",
    "OperatorFrame",
    "OperatorSpec",
    "Plan",
    "Route",
    "SimLevel",
    "SpawnEvent",
    "TileCell",
    "TileKind",
]
