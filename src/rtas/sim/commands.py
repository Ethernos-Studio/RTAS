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
# @brief `rtas sim` 子命令实现

"""``rtas sim`` 子命令：位置图 / 推演视频 / 地块清单。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .engine import SimulationResult, Simulator
from .loader import LevelDataError, load_level, load_plan, parse_operator_spec
from .model import Buildable, OperatorSpec, Plan, SimLevel, TileKind
from .render import MapRenderer, save_snapshot

logger = logging.getLogger(__name__)

USAGE = """\
可用子命令:
  rtas sim map   -l level.json -t 30 [-t 45,60] [-o shot.png]   某时刻敌人位置图（无阻挡）
  rtas sim run   -l level.json [-p plan.json] [--op ...] [-o run.mp4] [--json data.json]
  rtas sim tiles -l level.json [--plan-template plan.json]      列出可部署地块 / 生成计划模板

关卡文件可以是 RTAS 简化格式，也可以是明日方舟原版关卡
（ArknightsGameData/zh_CN/gamedata/levels/**.json，会自动读取旁边的 enemydata/enemy_database.json）。
"""


def run(args: Any, logger_: logging.Logger) -> int:
    """``rtas sim`` 入口。"""
    command = getattr(args, "sim_command", None)
    try:
        if command == "map":
            return _command_map(args)
        if command == "run":
            return _command_run(args)
        if command == "tiles":
            return _command_tiles(args)
    except LevelDataError as exc:
        logger_.error(f"关卡数据错误: {exc}")
        return 1
    except FileNotFoundError as exc:
        logger_.error(str(exc))
        return 1
    except ValueError as exc:
        logger_.error(f"参数错误: {exc}")
        return 1

    print(USAGE)
    return 0


# =============================================================================
# map：某时刻的敌人位置图
# =============================================================================

def _command_map(args: Any) -> int:
    level = _load_level(args)
    times = _parse_times(args.time)

    plan = Plan(duration=max(times) + 1.0, fps=20.0)
    result = Simulator(level, plan).run(duration=max(times) + 1.0)
    snapshots = [(time, result.frame_at(time)) for time in times]

    if args.json:
        payload = {
            "file": str(args.level),
            "level_id": level.level_id,
            "level_name": level.name,
            "rows": level.rows,
            "cols": level.cols,
            "snapshots": [
                {
                    "time": round(time, 2),
                    "alive": len(frame.enemies),
                    "enemies": [enemy.as_dict() for enemy in frame.enemies],
                }
                for time, frame in snapshots
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.output:
        renderer = MapRenderer(level, cell=args.cell)
        paths: list[Path] = []
        for time, frame in snapshots:
            target = Path(args.output)
            if len(snapshots) > 1:
                target = target.with_name(f"{target.stem}_{time:07.1f}{target.suffix or '.png'}")
            paths.append(save_snapshot(level, frame, target, renderer=renderer))
        print(f"关卡: {level.level_id}  {level.rows}x{level.cols}  敌人 {level.total_enemies}")
        for (time, frame), path in zip(snapshots, paths, strict=True):
            print(f"  t={time:6.2f}s  场上 {len(frame.enemies):2d} 人  ->  {path}")
        return 0

    print("=" * 74)
    print(f"关卡: {level.level_id}  {level.name}  {level.rows}x{level.cols}  "
          f"敌人 {level.total_enemies}  最后一个生成 {level.last_spawn_time:.1f}s")
    print("=" * 74)
    for time, frame in snapshots:
        print(f"t = {time:.2f}s   场上 {len(frame.enemies)} 人   "
              f"（无阻挡推演，仅按路线移动）")
        for enemy in sorted(frame.enemies, key=lambda item: -item.progress):
            print(f"    {enemy.name:<10s} ({enemy.row:5.2f}, {enemy.col:5.2f})  "
                  f"进度 {enemy.progress * 100:5.1f}%  HP {enemy.hp:.0f}/{enemy.max_hp:.0f}")
        if not frame.enemies:
            print("    （该时刻场上没有敌人）")
    print("=" * 74)
    return 0


# =============================================================================
# run：带干员的推演
# =============================================================================

def _command_run(args: Any) -> int:
    level = _load_level(args)
    plan = load_plan(args.plan) if args.plan else Plan()

    for text in args.op or []:
        plan.operators.append(parse_operator_spec(text))

    if args.duration is not None:
        plan.duration = float(args.duration)
    if args.dt is not None:
        plan.dt = float(args.dt)
    if args.fps is not None:
        plan.fps = float(args.fps)
    if args.speed_scale is not None:
        plan.speed_scale = float(args.speed_scale)

    _warn_operator_tiles(level, plan.operators)

    result = Simulator(level, plan).run()
    _print_summary(result)

    outputs: list[str] = []
    if args.json_out:
        path = result.save_json(args.json_out, include_frames=not args.no_frames)
        outputs.append(f"数据: {path}")
    if args.video:
        total = len(result.frames)
        print(f"正在渲染视频：{total} 帧（{plan.fps:.0f} fps）…")
        path = result.export_video(args.video, renderer=MapRenderer(level, cell=args.cell))
        outputs.append(f"视频: {path}")
    for line in outputs:
        print(line)
    return 0


def _warn_operator_tiles(level: SimLevel, operators: list[OperatorSpec]) -> None:
    for operator in operators:
        row, col = operator.tile
        cell = level.tile(row, col)
        if not (0 <= row < level.rows and 0 <= col < level.cols):
            logger.warning(f"干员「{operator.name}」坐标 ({row}, {col}) 超出地图范围")
            continue
        if cell.buildable is Buildable.NONE:
            logger.warning(
                f"干员「{operator.name}」放在 ({row}, {col})，该地块不可部署"
                f"（{cell.kind.value}），仍按配置参与推演"
            )


# =============================================================================
# tiles：地块清单 / 计划模板
# =============================================================================

def _command_tiles(args: Any) -> int:
    level = _load_level(args)
    deployable = level.deployable()

    if args.json:
        payload = {
            "file": str(args.level),
            "level_id": level.level_id,
            "level_name": level.name,
            "rows": level.rows,
            "cols": level.cols,
            "last_spawn_time": round(level.last_spawn_time, 2),
            "deployable": [
                {
                    "row": row,
                    "col": col,
                    "mode": level.tile(row, col).buildable.value,
                    "height": level.tile(row, col).height,
                }
                for row, col in deployable
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=" * 74)
        print(f"关卡: {level.level_id}  {level.name}  {level.rows}x{level.cols}")
        print("图例: m近战位  r远程位  a通用位  .地面  ^高台  #墙  S红门  E蓝门  o地穴  空格=不可通行")
        print("=" * 74)
        for row in range(level.rows):
            line = []
            for col in range(level.cols):
                cell = level.tile(row, col)
                if cell.kind is TileKind.START:
                    line.append("S")
                elif cell.kind is TileKind.END:
                    line.append("E")
                elif cell.kind is TileKind.HOLE:
                    line.append("o")
                elif cell.buildable is Buildable.ALL:
                    line.append("a")
                elif cell.buildable is Buildable.MELEE:
                    line.append("m")
                elif cell.buildable is Buildable.RANGED:
                    line.append("r")
                elif cell.kind is TileKind.WALL:
                    line.append("#")
                elif cell.kind is TileKind.FORBIDDEN:
                    line.append(" ")
                else:
                    line.append("^" if cell.height else ".")
            print(f"  r{row:02d}  " + "".join(line))
        print("=" * 74)
        print(f"可部署地块 {len(deployable)} 个（近战 "
              f"{len(level.deployable('melee'))} / 远程 {len(level.deployable('ranged'))}）")
        print("  " + "  ".join(f"({row},{col})" for row, col in deployable[:24])
              + ("  …" if len(deployable) > 24 else ""))

    if args.plan_template:
        template = _plan_template(level, deployable)
        target = Path(args.plan_template)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(template, handle, ensure_ascii=False, indent=2)
        print(f"计划模板已保存: {target}")
    return 0


def _plan_template(level: SimLevel, deployable: list[tuple[int, int]]) -> dict[str, Any]:
    melee = level.deployable("melee") or deployable
    ranged = level.deployable("ranged") or deployable
    tile_melee = melee[0] if melee else (0, 0)
    tile_ranged = ranged[0] if ranged else tile_melee
    return {
        "duration": round(max(120.0, level.last_spawn_time + 90.0), 1),
        "dt": 0.1,
        "fps": 20,
        "speed_scale": 1.0,
        "operators": [
            {
                "name": "近卫",
                "tile": list(tile_melee),
                "deploy_time": 5.0,
                "block": 1,
                "damage": 600,
                "attack_interval": 1.0,
                "attack_mode": "melee",
                "range_tiles": 1.5,
            },
            {
                "name": "狙击",
                "tile": list(tile_ranged),
                "deploy_time": 8.0,
                "block": 0,
                "damage": 400,
                "attack_interval": 1.2,
                "attack_mode": "ranged",
                "range_tiles": 3.0,
            },
        ],
    }


# =============================================================================
# 工具
# =============================================================================

def _load_level(args: Any) -> SimLevel:
    return load_level(args.level, enemy_db=getattr(args, "enemy_db", None))


def _parse_times(values: Optional[list[str]]) -> list[float]:
    if not values:
        return [30.0]
    times: list[float] = []
    for value in values:
        for chunk in str(value).replace("，", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            times.append(float(chunk))
    if not times:
        raise ValueError("--time 至少需要一个时刻，例如 -t 30 或 -t 30,60")
    return times


def _print_summary(result: SimulationResult) -> None:
    level = result.level
    summary = result.summary
    print("=" * 74)
    print(f"关卡: {level.level_id}  {level.name}  {level.rows}x{level.cols}  "
          f"敌人 {summary['total_enemies']}  生命 {summary['max_life']}")
    print(f"推演: 时长 {summary['duration']:.1f}s  步长 {summary.get('dt', result.plan.dt)}s  "
          f"{result.plan.fps:.0f} fps  速度倍率 {result.plan.speed_scale:g}")
    print(f"结果: 击杀 {summary['killed']}  漏怪 {summary['leaked']}  "
          f"生命 {summary['life']}/{summary['max_life']}  "
          f"最高同时阻挡 {summary['blocked_peak']}  "
          f"清场 {summary['cleared_at'] if summary['cleared_at'] is not None else '-'}s")
    if summary["operators"]:
        print("干员:")
        for operator in summary["operators"]:
            print(f"  - {operator['name']:<8s} @({operator['tile'][0]},{operator['tile'][1]})  "
                  f"部署 {operator['deploy_time']:.1f}s  阻挡 {operator['block']}  "
                  f"伤害 {operator['damage']:.0f}  击杀 {operator['kills']}  "
                  f"DPS {operator['dps']:.1f}")
    else:
        print("干员: 无（无阻挡推演）")
    print("=" * 74)


__all__ = ["USAGE", "run"]
