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
# @brief 关卡 / 计划加载：RTAS 简化格式 + 明日方舟原版关卡格式

"""关卡加载。

支持两种输入，统一转换成 :class:`rtas.sim.model.SimLevel`：

1. **RTAS 简化格式** —— ``rtas level`` 用的那种 JSON（``tiles`` / ``routes`` / ``waves``），
   敌人属性写在 ``enemies`` 里，适合手写和测试；
2. **明日方舟原版关卡** —— ``ArknightsGameData/zh_CN/gamedata/levels/**.json``，
   会结合 ``levels/enemydata/enemy_database.json`` 读取敌人血量/速度/阻挡权重，
   路线缺失 ``checkpoints`` 时用 A* 在可通行地块上补全。
"""

from __future__ import annotations

import heapq
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional

from .model import (
    AttackMode,
    Buildable,
    EnemyProfile,
    OperatorSpec,
    Plan,
    Route,
    SimLevel,
    SpawnEvent,
    TileCell,
    TileKind,
)

logger = logging.getLogger(__name__)

#: 原版 tileKey → 地块类型
_TILE_KINDS = {
    "tile_forbidden": TileKind.FORBIDDEN,
    "tile_wall": TileKind.WALL,
    "tile_road": TileKind.ROAD,
    "tile_floor": TileKind.FLOOR,
    "tile_start": TileKind.START,
    "tile_end": TileKind.END,
    "tile_hole": TileKind.HOLE,
    "tile_air": TileKind.ROAD,
}

#: 简化格式 TileType(int) → 地块类型
_SIMPLE_KINDS = {
    0: TileKind.FORBIDDEN,
    1: TileKind.FLOOR,
    2: TileKind.WALL,
    3: TileKind.START,
    4: TileKind.END,
    5: TileKind.START,
    6: TileKind.END,
    7: TileKind.HOLE,
    8: TileKind.FLOOR,
    9: TileKind.FLOOR,
    10: TileKind.FLOOR,
}

_BUILDABLE = {
    "NONE": Buildable.NONE,
    "MELEE": Buildable.MELEE,
    "RANGED": Buildable.RANGED,
    "ALL": Buildable.ALL,
}

_ENEMY_DB_CACHE: dict[str, dict[str, Any]] = {}


class LevelDataError(ValueError):
    """关卡数据不合法。"""


# =============================================================================
# 对外入口
# =============================================================================

def load_level(
    source: str | Path | None = None,
    *,
    data: Optional[dict[str, Any]] = None,
    enemy_db: str | Path | dict[str, Any] | None = None,
) -> SimLevel:
    """加载关卡。

    Args:
        source: 关卡 JSON 路径
        data: 直接传入的关卡字典（优先于 ``source``）
        enemy_db: 敌人数据库路径/字典；原版关卡会尝试在旁边自动查找
            ``enemydata/enemy_database.json``

    Returns:
        :class:`SimLevel`
    """
    path = Path(source) if source else None
    if data is None:
        if path is None:
            raise LevelDataError("需要提供关卡文件路径或关卡数据")
        data = _read_json(path)

    if not isinstance(data, dict):
        raise LevelDataError("关卡数据必须是 JSON 对象")

    if "mapData" in data:
        level = _from_game_json(data, path, enemy_db)
    elif "tiles" in data or "levelId" in data:
        level = _from_simple_json(data)
    else:
        raise LevelDataError(
            "无法识别的关卡格式：既不是 RTAS 简化格式（tiles/routes/waves），"
            "也不是明日方舟原版关卡（mapData/routes/waves）"
        )

    level.source = str(path) if path else "inline"
    if level.rows == 0 or level.cols == 0:
        raise LevelDataError("关卡地图为空")
    if not level.routes:
        raise LevelDataError("关卡没有任何可用路线")

    missing = sorted({spawn.route_id for spawn in level.spawns
                      if spawn.route_id not in level.routes})
    if missing:
        logger.warning(f"有敌人引用了不存在的路线 {missing}，这些敌人会被跳过")
    return level


def load_plan(source: str | Path | None = None, *,
              data: Optional[dict[str, Any]] = None) -> Plan:
    """加载推演计划（干员部署 + 时长/帧率）。"""
    if data is None:
        if source is None:
            return Plan()
        data = _read_json(Path(source))
    return Plan.from_dict(data)


def parse_operator_spec(text: str) -> OperatorSpec:
    """解析一行为内联干员配置。

    例：``name=近卫,t=5,tile=3:2,block=1,damage=600,interval=1.0,mode=melee,range=1.5``

    支持的键（含别名）：
    ``name``、``t/time/deploy_time``、``tile/pos``（``行:列``）、``block``、
    ``damage/atk``、``interval/attack_interval``、``mode/attack_mode``、``range``、
    ``retreat/retreat_time``。
    """
    fields: dict[str, str] = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"无法解析干员配置「{chunk}」，应形如 name=近卫,t=5,tile=3:2,block=1,damage=600"
            )
        key, value = chunk.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    def pick(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in fields:
                return fields[name]
        return default

    tile_raw = str(pick("tile", "pos", default="0:0"))
    tile = _parse_tile(tile_raw)

    mode_raw = str(pick("mode", "attack_mode", "attack", default="melee")).lower()
    mode = AttackMode.RANGED if mode_raw in ("ranged", "range", "远程", "r") else AttackMode.MELEE

    retreat_raw = pick("retreat", "retreat_time")
    return OperatorSpec(
        name=str(pick("name", "id", default="干员")),
        tile=tile,
        deploy_time=float(pick("t", "time", "deploy_time", "deploy", default=0.0)),
        retreat_time=None if retreat_raw in (None, "", "none") else float(retreat_raw),
        block=int(pick("block", "block_count", "阻挡", default=1)),
        damage=float(pick("damage", "atk", "攻击", default=500.0)),
        attack_interval=float(pick("interval", "attack_interval", "攻速", default=1.0)),
        attack_mode=mode,
        range_tiles=float(pick("range", "range_tiles", default=1.5)),
    )


# =============================================================================
# 简化格式
# =============================================================================

def _from_simple_json(data: dict[str, Any]) -> SimLevel:
    map_width = int(data.get("mapWidth") or 0)
    map_height = int(data.get("mapHeight") or 0)

    tiles_raw = data.get("tiles") or []
    if not map_width:
        map_width = max((int(t.get("col", 0)) for t in tiles_raw), default=-1) + 1
    if not map_height:
        map_height = max((int(t.get("row", 0)) for t in tiles_raw), default=-1) + 1

    grid = [[TileCell(TileKind.FORBIDDEN, Buildable.NONE, 0, False, False)
             for _ in range(map_width)] for _ in range(map_height)]

    for tile_raw in tiles_raw:
        row = int(tile_raw.get("row", 0))
        col = int(tile_raw.get("col", 0))
        if not (0 <= row < map_height and 0 <= col < map_width):
            continue
        kind = _SIMPLE_KINDS.get(int(tile_raw.get("tileType", 0)), TileKind.FORBIDDEN)
        buildable = {0: Buildable.NONE, 1: Buildable.MELEE,
                     2: Buildable.RANGED, 3: Buildable.ALL}.get(
            int(tile_raw.get("buildableType", 0)), Buildable.NONE)
        height = 1 if float(tile_raw.get("height", 0.0)) > 0 else 0
        walkable = kind not in (TileKind.FORBIDDEN, TileKind.WALL)
        flyable = kind is not TileKind.FORBIDDEN
        grid[row][col] = TileCell(kind, buildable, height, walkable, flyable)

    routes: dict[str, Route] = {}
    for route_raw in data.get("routes") or []:
        route_id = str(route_raw.get("id") or route_raw.get("route_id") or len(routes))
        points = [
            (float(point.get("row", 0)), float(point.get("col", 0)))
            for point in route_raw.get("points") or []
        ]
        if len(points) < 2:
            continue
        routes[route_id] = Route(route_id=route_id, points=points,
                                 flying=bool(route_raw.get("flying", False)))

    spawns: list[SpawnEvent] = []
    cursor = 0.0
    for wave_raw in data.get("waves") or []:
        cursor += float(wave_raw.get("preDelay", 0.0))
        wave_span = 0.0
        for spawn_raw in wave_raw.get("spawns") or []:
            spawn_delay = float(spawn_raw.get("delay", 0.0))
            interval = float(spawn_raw.get("interval", 1.0))
            count = int(spawn_raw.get("count", 1))
            route_id = str(spawn_raw.get("routeId") or spawn_raw.get("route_id") or "")
            for index in range(count):
                spawns.append(SpawnEvent(cursor + spawn_delay + index * interval,
                                         str(spawn_raw.get("enemyId", "unknown")), route_id))
            wave_span = max(wave_span, spawn_delay + max(0, count - 1) * interval)
        cursor += wave_span + float(wave_raw.get("postDelay", 0.0))

    spawns.sort(key=lambda item: item.time)

    profiles: dict[str, EnemyProfile] = {}
    for enemy_id, raw in (data.get("enemies") or {}).items():
        if not isinstance(raw, dict):
            continue
        profiles[str(enemy_id)] = EnemyProfile(
            enemy_id=str(enemy_id),
            name=str(raw.get("name", enemy_id)),
            hp=float(raw.get("hp", raw.get("maxHp", 1000.0))),
            speed=float(raw.get("speed", raw.get("moveSpeed", 0.7))),
            defense=float(raw.get("defense", raw.get("def", 0.0))),
            weight=float(raw.get("weight", 1.0)),
            flying=bool(raw.get("flying", False)),
            leak_damage=int(raw.get("leak_damage", 1)),
        )

    return SimLevel(
        level_id=str(data.get("levelId") or data.get("id") or "custom"),
        name=str(data.get("name", "")),
        grid=grid,
        routes=routes,
        spawns=spawns,
        enemies=profiles,
        life=int(data.get("life", data.get("maxLifePoint", 10))),
        move_multiplier=float(data.get("moveMultiplier", 1.0)),
        meta={"format": "rtas"},
    )


# =============================================================================
# 明日方舟原版关卡
# =============================================================================

def _from_game_json(
    data: dict[str, Any],
    path: Optional[Path],
    enemy_db: str | Path | dict[str, Any] | None,
) -> SimLevel:
    options = data.get("options") or {}
    move_multiplier = float(options.get("moveMultiplier") or 1.0)

    map_data = data.get("mapData") or {}
    rows_raw = map_data.get("map") or []
    tiles_raw = map_data.get("tiles") or []
    if not rows_raw or not tiles_raw:
        raise LevelDataError("原版关卡缺少 mapData.map / mapData.tiles")

    height = len(rows_raw)
    width = len(rows_raw[0]) if rows_raw else 0
    grid = [[TileCell(TileKind.FORBIDDEN, Buildable.NONE, 0, False, False)
             for _ in range(width)] for _ in range(height)]

    for row, line in enumerate(rows_raw):
        for col, tile_index in enumerate(line):
            if not isinstance(tile_index, int) or not (0 <= tile_index < len(tiles_raw)):
                continue
            raw = tiles_raw[tile_index]
            kind = _TILE_KINDS.get(str(raw.get("tileKey", "")), TileKind.FLOOR)
            buildable = _BUILDABLE.get(str(raw.get("buildableType", "NONE")), Buildable.NONE)
            height_type = 1 if str(raw.get("heightType", "LOWLAND")) == "HIGHLAND" else 0

            passable = str(raw.get("passableMask", "ALL")).upper()
            walkable = passable in ("ALL", "WALK_ONLY")
            flyable = passable in ("ALL", "FLY_ONLY")
            if kind in (TileKind.START, TileKind.END) and passable == "NONE":
                walkable = flyable = True
            if kind is TileKind.WALL:
                walkable = False
            grid[row][col] = TileCell(kind, buildable, height_type, walkable, flyable)

    routes: dict[str, Route] = {}
    for index, route_raw in enumerate(data.get("routes") or []):
        route_id = str(index)
        start = route_raw.get("startPosition") or {}
        end = route_raw.get("endPosition") or {}
        start_pos = (int(start.get("row", 0)), int(start.get("col", 0)))
        end_pos = (int(end.get("row", 0)), int(end.get("col", 0)))

        # 红门 → 途经点 → 蓝门，构成完整折线
        checkpoints: list[tuple[float, float]] = []
        for checkpoint in route_raw.get("checkpoints") or []:
            if str(checkpoint.get("type", "MOVE")).upper() != "MOVE":
                continue
            position = checkpoint.get("position") or {}
            checkpoints.append((float(position.get("row", 0)),
                                float(position.get("col", 0))))

        if start_pos == end_pos:
            if not checkpoints:
                continue  # 未使用的占位路线
            raw_points = checkpoints
        elif checkpoints:
            raw_points = [(float(start_pos[0]), float(start_pos[1])),
                          *checkpoints,
                          (float(end_pos[0]), float(end_pos[1]))]
        else:
            # 没有途经点：在可通行地块上找一条路，避免直线穿墙
            diagonal = bool(route_raw.get("allowDiagonalMove", True))
            grid_path = _astar(grid, start_pos, end_pos, diagonal=diagonal)
            raw_points = ([(float(row), float(col)) for row, col in grid_path]
                          if len(grid_path) >= 2
                          else [(float(start_pos[0]), float(start_pos[1])),
                                (float(end_pos[0]), float(end_pos[1]))])

        points: list[tuple[float, float]] = [raw_points[0]]
        for point in raw_points[1:]:
            if (abs(point[0] - points[-1][0]) > 1e-9
                    or abs(point[1] - points[-1][1]) > 1e-9):
                points.append(point)

        if len(points) < 2:
            logger.warning(f"路线 {route_id} 起点与终点相同，已跳过")
            continue

        routes[route_id] = Route(route_id=route_id, points=points)

        # 原版地图有些关卡的门格还是 tile_road，用路线端点补成红门/蓝门
        _mark_gate(grid, start_pos, TileKind.START)
        _mark_gate(grid, end_pos, TileKind.END)

    profiles = _game_enemy_profiles(data.get("enemyDbRefs") or [], path, enemy_db,
                                   move_multiplier)

    spawns = _game_spawns(data.get("waves") or [])

    # 波次里出现但 enemyDbRefs 没登记的敌人，补一份默认档案（速度受 moveMultiplier 影响）
    for spawn in spawns:
        if spawn.enemy_id not in profiles:
            profiles[spawn.enemy_id] = EnemyProfile(
                enemy_id=spawn.enemy_id, name=spawn.enemy_id,
                speed=0.7 * move_multiplier,
            )

    return SimLevel(
        level_id=str(data.get("levelId") or (path.stem if path else "unknown")),
        name=str(data.get("name") or (path.stem if path else "")),
        grid=grid,
        routes=routes,
        spawns=spawns,
        enemies=profiles,
        life=int(options.get("maxLifePoint", 10)),
        move_multiplier=move_multiplier,
        meta={
            "format": "arknights",
            "initial_cost": options.get("initialCost"),
            "max_cost": options.get("maxCost"),
            "cost_increase_time": options.get("costIncreaseTime"),
            "character_limit": options.get("characterLimit"),
        },
    )


def _mark_gate(grid: list[list[TileCell]], pos: tuple[int, int], kind: TileKind) -> None:
    row, col = pos
    if not (0 <= row < len(grid) and grid and 0 <= col < len(grid[0])):
        return
    cell = grid[row][col]
    if cell.kind in (TileKind.START, TileKind.END):
        return
    grid[row][col] = TileCell(kind, cell.buildable, cell.height, True, True)


def _game_spawns(waves: list[dict[str, Any]]) -> list[SpawnEvent]:
    """按原版 waves/fragments/actions 计算绝对生成时间。"""
    spawns: list[SpawnEvent] = []
    cursor = 0.0
    for wave in waves:
        cursor += float(wave.get("preDelay", 0.0))
        wave_span = 0.0
        for fragment in wave.get("fragments") or []:
            fragment_start = cursor + float(fragment.get("preDelay", 0.0))
            fragment_span = 0.0
            for action in fragment.get("actions") or []:
                if str(action.get("actionType", "")).upper() != "SPAWN":
                    continue
                count = int(action.get("count", 1))
                interval = float(action.get("interval", 1.0))
                offset = float(action.get("preDelay", 0.0))
                route_id = str(action.get("routeIndex", -1))
                enemy_id = str(action.get("key", "unknown"))
                for index in range(count):
                    spawns.append(SpawnEvent(fragment_start + offset + index * interval,
                                             enemy_id, route_id))
                fragment_span = max(fragment_span, offset + max(0, count - 1) * interval)
            wave_span = max(wave_span, float(fragment.get("preDelay", 0.0)) + fragment_span)
        cursor += wave_span + float(wave.get("postDelay", 0.0))
    spawns.sort(key=lambda item: item.time)
    return spawns


def _game_enemy_profiles(
    refs: Iterable[dict[str, Any]],
    level_path: Optional[Path],
    enemy_db: str | Path | dict[str, Any] | None,
    move_multiplier: float,
) -> dict[str, EnemyProfile]:
    db = _resolve_enemy_db(level_path, enemy_db)
    index = _index_enemy_db(db) if db else {}

    profiles: dict[str, EnemyProfile] = {}
    for ref in refs:
        enemy_id = str(ref.get("id") or ref.get("key") or "")
        if not enemy_id:
            continue
        level_index = int(ref.get("level", 0) or 0)
        data = _enemy_data(index.get(enemy_id), level_index)
        if data is None:
            profiles[enemy_id] = EnemyProfile(enemy_id=enemy_id,
                                              name=enemy_id,
                                              speed=0.7 * move_multiplier)
            continue

        attributes = data.get("attributes") or {}
        weight = _attr(attributes, "blockVolume", 1.0)
        if weight <= 0:  # 游戏里 0/-1 表示未定义，按 1 处理
            weight = 1.0
        speed = _attr(attributes, "moveSpeed", 0.7)
        if speed <= 0:
            speed = 0.7
        profiles[enemy_id] = EnemyProfile(
            enemy_id=enemy_id,
            name=_attr_text(data.get("name")) or enemy_id,
            hp=_attr(attributes, "maxHp", 1000.0),
            speed=speed * move_multiplier,
            defense=_attr(attributes, "def", 0.0),
            weight=weight,
            flying=str(_attr_text(data.get("motion")) or "WALK").upper() == "FLY",
            leak_damage=max(1, int(_attr(data, "lifePointReduce", 1) or 1)),
        )
    return profiles


def _resolve_enemy_db(level_path: Optional[Path],
                      enemy_db: str | Path | dict[str, Any] | None) -> Optional[dict[str, Any]]:
    if isinstance(enemy_db, dict):
        return enemy_db
    if enemy_db is not None:
        return _read_json(Path(enemy_db))
    if level_path is None:
        return None
    for parent in level_path.parents:
        candidate = parent / "enemydata" / "enemy_database.json"
        if candidate.exists():
            return _read_json_cached(candidate)
    return None


def _index_enemy_db(db: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    entries = db.get("enemies") if isinstance(db, dict) else db
    index: dict[str, list[dict[str, Any]]] = {}
    for entry in entries or []:
        key = entry.get("Key") or entry.get("key")
        value = entry.get("Value") or entry.get("value")
        if key is not None and isinstance(value, list):
            index[str(key)] = value
    return index


def _enemy_data(levels: Optional[list[dict[str, Any]]], level_index: int) -> Optional[dict[str, Any]]:
    if not levels:
        return None
    for item in levels:
        if int(item.get("level", -1)) == level_index:
            return item.get("enemyData") or item.get("enemy_data") or {}
    return (levels[0].get("enemyData") or levels[0].get("enemy_data") or None)


def _attr(attributes: dict[str, Any], key: str, default: float) -> float:
    raw = attributes.get(key)
    if isinstance(raw, dict):
        value = raw.get("m_value")
        if isinstance(value, (int, float)):
            return float(value)
    elif isinstance(raw, (int, float)):
        return float(raw)
    return float(default)


def _attr_text(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        value = raw.get("m_value")
        return str(value) if value is not None else None
    if isinstance(raw, str):
        return raw
    return None


# =============================================================================
# 工具
# =============================================================================

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json_cached(path: Path) -> dict[str, Any]:
    key = str(path.resolve())
    cached = _ENEMY_DB_CACHE.get(key)
    if cached is not None:
        return cached
    data = _read_json(path)
    _ENEMY_DB_CACHE[key] = data
    return data


def _parse_tile(text: str) -> tuple[int, int]:
    for separator in (":", "x", "X", "/", "-"):
        if separator in text:
            row, _, col = text.partition(separator)
            return (int(row), int(col))
    raise ValueError(f"无法解析坐标「{text}」，应写成 行:列，例如 3:2")


def _astar(
    grid: list[list[TileCell]],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    diagonal: bool = True,
) -> list[tuple[int, int]]:
    """在可通行地块上找一条最短路径（返回经过的格子）。"""
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not (0 <= start[0] < height and 0 <= start[1] < width):
        return []
    if not (0 <= end[0] < height and 0 <= end[1] < width):
        return []

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(node: tuple[int, int]) -> float:
        dr = abs(node[0] - end[0])
        dc = abs(node[1] - end[1])
        if diagonal:
            return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)
        return dr + dc

    open_heap: list[tuple[float, int, tuple[int, int]]] = [(heuristic(start), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost: dict[tuple[int, int], float] = {start: 0.0}
    counter = 0
    visited: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        if current == end:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        visited.add(current)

        for delta_row, delta_col in moves:
            neighbor = (current[0] + delta_row, current[1] + delta_col)
            if neighbor in visited:
                continue
            if not (0 <= neighbor[0] < height and 0 <= neighbor[1] < width):
                continue
            cell = grid[neighbor[0]][neighbor[1]]
            if not cell.walkable:
                continue
            if delta_row and delta_col:
                # 不允许贴着墙角斜穿
                side_a = grid[current[0] + delta_row][current[1]]
                side_b = grid[current[0]][current[1] + delta_col]
                if not (side_a.walkable and side_b.walkable):
                    continue
            step = math.sqrt(2) if delta_row and delta_col else 1.0
            new_cost = cost[current] + step
            if new_cost < cost.get(neighbor, float("inf")):
                cost[neighbor] = new_cost
                came_from[neighbor] = current
                counter += 1
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), counter, neighbor))

    return []


__all__ = [
    "LevelDataError",
    "load_level",
    "load_plan",
    "parse_operator_spec",
]
