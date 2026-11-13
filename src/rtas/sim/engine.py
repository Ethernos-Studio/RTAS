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
# @brief 关卡推演引擎：移动 / 阻挡 / 攻击 / 漏怪

"""关卡推演引擎。

规则（简化但可解释）：

- 敌人按 ``spawns`` 的绝对时间在路线起点（红门）出现，以 ``speed`` 格/秒沿路线前进；
- 地面敌人走到干员所在格时，如果该干员还有阻挡余量（``block`` ≥ 已阻挡权重和 +
  该敌人 ``weight``）就被阻挡，停在原地；飞行敌人无视阻挡；
- 干员按 ``attack_interval`` 冷却出手，``melee`` 只打自己阻挡的敌人，
  ``ranged`` 打射程内（``range_tiles``）最靠前的敌人；伤害扣防御，
  最低保底 5%；
- 敌人血量归零记为击杀，走到路线终点记为漏怪并扣生命值；
- 干员到达 ``retreat_time`` 或退场后释放阻挡。

引擎只做数值推演、不依赖 OpenCV；渲染与视频导出在 :mod:`rtas.sim.render`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .model import (
    AttackMode,
    EnemyFrame,
    EnemyProfile,
    Frame,
    OperatorFrame,
    OperatorSpec,
    Plan,
    Route,
    SimLevel,
)

#: 伤害保底比例（防御再高也至少打出 5%）
_MIN_DAMAGE_RATIO = 0.05


@dataclass
class _EnemyRuntime:
    uid: int
    profile: EnemyProfile
    route: Route
    distance: float = 0.0
    hp: float = 0.0
    blocked_by: Optional[int] = None
    alive: bool = True
    state: str = "moving"      # moving / blocked / dead / leaked
    spawn_time: float = 0.0

    @property
    def progress(self) -> float:
        """沿路线完成度 0~1。"""
        length = self.route.length
        return (self.distance / length) if length > 0 else 1.0

    @property
    def position(self) -> tuple[float, float]:
        return self.route.position_at(self.distance)


@dataclass
class _OperatorRuntime:
    index: int
    spec: OperatorSpec
    cooldown: float = 0.0
    blocked: list[int] = field(default_factory=list)
    damage_done: float = 0.0
    kills: int = 0
    deployed: bool = False
    retreated: bool = False

    def active_at(self, time: float) -> bool:
        return self.deployed and not self.retreated and self.spec.is_active(time)


class Simulator:
    """把关卡 + 干员计划推演成逐帧状态。"""

    def __init__(self, level: SimLevel, plan: Optional[Plan] = None) -> None:
        self.level = level
        self.plan = plan or Plan()

    # -- 对外入口 ---------------------------------------------------------
    def run(
        self,
        *,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> SimulationResult:
        """运行推演。

        Args:
            duration: 最长推演时长（秒），默认取计划里的 ``duration``
            fps: 帧采样率，默认取计划里的 ``fps``
            dt: 物理步长，默认取计划里的 ``dt``
        """
        duration = float(duration if duration is not None else self.plan.duration)
        fps = float(fps if fps is not None else self.plan.fps) or 20.0
        dt = float(dt if dt is not None else self.plan.dt) or 0.1
        if dt <= 0:
            raise ValueError("dt 必须大于 0")
        if fps <= 0:
            raise ValueError("fps 必须大于 0")

        # 步长不能大于帧间隔，否则一帧都采不到（视频会变成快放）
        frame_interval = 1.0 / fps
        if dt > frame_interval + 1e-9:
            dt = frame_interval

        runner = _Runner(self.level, self.plan)
        frames: list[Frame] = []
        next_frame_time = 0.0
        time = 0.0
        steps = 0

        runner.deploy_due(time)
        frames.append(runner.snapshot(time))

        while time < duration - 1e-9:
            time += dt
            steps += 1
            runner.deploy_due(time)
            runner.step(time, dt)

            if time >= next_frame_time - 1e-9:
                frames.append(runner.snapshot(time))
                next_frame_time += frame_interval
                while next_frame_time <= time - 1e-9:  # 防止 fps 与 dt 不整除时漂移
                    next_frame_time += frame_interval

            if runner.finished():
                break

        if not frames or frames[-1].time < time - 1e-9:
            frames.append(runner.snapshot(time))

        summary = runner.summary(time, steps, dt)
        return SimulationResult(level=self.level, plan=self.plan,
                                frames=frames, events=runner.events, summary=summary)


class _Runner:
    """内部状态机（不对外暴露）。"""

    def __init__(self, level: SimLevel, plan: Plan) -> None:
        self.level = level
        self.plan = plan
        self.enemies: dict[int, _EnemyRuntime] = {}
        self.operators: list[_OperatorRuntime] = [
            _OperatorRuntime(index=index, spec=spec)
            for index, spec in enumerate(plan.operators)
        ]
        self.events: list[dict[str, Any]] = []
        self._spawns = sorted(level.spawns, key=lambda item: item.time)
        self._spawn_index = 0
        self._next_uid = 0
        self.killed = 0
        self.leaked = 0
        self.spawned = 0
        self.life = level.life
        self.blocked_peak = 0
        self._last_resolve_time = 0.0

    # -- 时间轴 -----------------------------------------------------------
    def deploy_due(self, time: float) -> None:
        """处理干员落地 / 撤退，以及到点的敌人生成。"""
        while self._spawn_index < len(self._spawns):
            spawn = self._spawns[self._spawn_index]
            if spawn.time > time + 1e-9:
                break
            self._spawn_index += 1
            self._spawn(spawn, time)

        for operator in self.operators:
            spec = operator.spec
            if not operator.deployed and time + 1e-9 >= spec.deploy_time:
                operator.deployed = True
                operator.cooldown = 0.0
                self.events.append({
                    "time": round(time, 3), "type": "deploy",
                    "operator": operator.index, "name": spec.name,
                    "tile": list(spec.tile), "block": spec.block,
                })
            if (operator.deployed and not operator.retreated
                    and spec.retreat_time is not None
                    and time + 1e-9 >= spec.retreat_time):
                self._retreat(operator, time)

    def _spawn(self, spawn, time: float) -> None:
        route = self.level.routes.get(spawn.route_id)
        if route is None:
            return
        profile = self.level.profile(spawn.enemy_id)
        uid = self._next_uid
        self._next_uid += 1
        self.enemies[uid] = _EnemyRuntime(
            uid=uid, profile=profile, route=route, distance=0.0,
            hp=profile.hp, spawn_time=spawn.time,
        )
        self.spawned += 1
        self.events.append({
            "time": round(time, 3), "type": "spawn", "uid": uid,
            "enemy": spawn.enemy_id, "name": profile.name,
            "route": spawn.route_id,
        })

    def _retreat(self, operator: _OperatorRuntime, time: float) -> None:
        operator.retreated = True
        for uid in list(operator.blocked):
            enemy = self.enemies.get(uid)
            if enemy is not None:
                enemy.blocked_by = None
                enemy.state = "moving"
        operator.blocked.clear()
        self.events.append({
            "time": round(time, 3), "type": "retreat",
            "operator": operator.index, "name": operator.spec.name,
        })

    # -- 每步推进 ---------------------------------------------------------
    def step(self, time: float, dt: float) -> None:
        speed_scale = self.plan.speed_scale
        alive = [enemy for enemy in self.enemies.values() if enemy.alive]
        # 越靠前的敌人先抢阻挡位
        alive.sort(key=lambda item: item.progress, reverse=True)

        for enemy in alive:
            if enemy.blocked_by is not None:
                operator = self.operators[enemy.blocked_by]
                if operator.active_at(time) and enemy.uid in operator.blocked:
                    continue  # 被阻挡：原地不动
                self._release(enemy)

            if enemy.blocked_by is None and not enemy.profile.flying:
                blocker = self._find_blocker(enemy, time)
                if blocker is not None:
                    self._block(enemy, blocker, time)
                    continue

            enemy.distance += enemy.profile.speed * speed_scale * dt
            if enemy.distance >= enemy.route.length:
                self._leak(enemy, time)

        self._attack(time, dt)
        self.blocked_peak = max(self.blocked_peak, self._blocked_count())

    def _find_blocker(self, enemy: _EnemyRuntime,
                      time: float) -> Optional[_OperatorRuntime]:
        tile = enemy.route.tile_at(enemy.distance)
        for operator in self.operators:
            if operator.spec.tile != tile or not operator.active_at(time):
                continue
            if operator.spec.block <= 0:
                continue
            used = sum(self.enemies[uid].profile.weight for uid in operator.blocked
                       if uid in self.enemies)
            if used + enemy.profile.weight <= operator.spec.block + 1e-9:
                return operator
        return None

    def _block(self, enemy: _EnemyRuntime, operator: _OperatorRuntime, time: float) -> None:
        # 吸附到干员格心：视觉上「站在格子里」，也避免在格子边缘反复进出
        center = enemy.route.distance_at(float(operator.spec.tile[0]),
                                         float(operator.spec.tile[1]))
        if abs(center - enemy.distance) <= 0.75:
            enemy.distance = center
        enemy.blocked_by = operator.index
        enemy.state = "blocked"
        operator.blocked.append(enemy.uid)
        self.events.append({
            "time": round(time, 3), "type": "block", "uid": enemy.uid,
            "enemy": enemy.profile.enemy_id, "operator": operator.index,
        })

    def _release(self, enemy: _EnemyRuntime) -> None:
        if enemy.blocked_by is None:
            return
        operator = self.operators[enemy.blocked_by]
        if enemy.uid in operator.blocked:
            operator.blocked.remove(enemy.uid)
        enemy.blocked_by = None
        enemy.state = "moving"

    def _attack(self, time: float, dt: float) -> None:
        for operator in self.operators:
            if not operator.active_at(time):
                continue
            operator.cooldown -= dt
            if operator.cooldown > 1e-9:
                continue
            target = self._pick_target(operator)
            if target is None:
                operator.cooldown = 0.0  # 没有目标：保持待发
                continue

            damage = max(operator.spec.damage - target.profile.defense,
                         operator.spec.damage * _MIN_DAMAGE_RATIO)
            target.hp -= damage
            operator.damage_done += damage
            operator.cooldown = operator.spec.attack_interval

            if target.hp <= 1e-9:
                self._kill(target, operator, time)

    def _pick_target(self, operator: _OperatorRuntime) -> Optional[_EnemyRuntime]:
        spec = operator.spec
        if spec.attack_mode is AttackMode.RANGED:
            row, col = spec.tile
            candidates = [
                enemy for enemy in self.enemies.values()
                if enemy.alive
                and math.hypot(enemy.position[0] - row, enemy.position[1] - col)
                <= spec.range_tiles + 1e-9
            ]
        else:
            candidates = [self.enemies[uid] for uid in operator.blocked
                          if uid in self.enemies and self.enemies[uid].alive]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.progress)

    def _kill(self, enemy: _EnemyRuntime, operator: _OperatorRuntime, time: float) -> None:
        enemy.hp = 0.0
        enemy.alive = False
        enemy.state = "dead"
        if enemy.uid in operator.blocked:
            operator.blocked.remove(enemy.uid)
        enemy.blocked_by = None
        operator.kills += 1
        self.killed += 1
        self._last_resolve_time = time
        self.events.append({
            "time": round(time, 3), "type": "kill", "uid": enemy.uid,
            "enemy": enemy.profile.enemy_id, "operator": operator.index,
        })

    def _leak(self, enemy: _EnemyRuntime, time: float) -> None:
        enemy.alive = False
        enemy.state = "leaked"
        self.leaked += 1
        self.life = max(0, self.life - enemy.profile.leak_damage)
        self._last_resolve_time = time
        self.events.append({
            "time": round(time, 3), "type": "leak", "uid": enemy.uid,
            "enemy": enemy.profile.enemy_id, "life": self.life,
        })

    def _blocked_count(self) -> int:
        return sum(len(operator.blocked) for operator in self.operators)

    # -- 状态输出 ---------------------------------------------------------
    def finished(self) -> bool:
        return (self._spawn_index >= len(self._spawns)
                and not any(enemy.alive for enemy in self.enemies.values()))

    def snapshot(self, time: float) -> Frame:
        enemies = [
            EnemyFrame(
                uid=enemy.uid,
                enemy_id=enemy.profile.enemy_id,
                name=enemy.profile.name or enemy.profile.enemy_id,
                row=enemy.position[0],
                col=enemy.position[1],
                hp=round(enemy.hp, 1),
                max_hp=enemy.profile.hp,
                blocked_by=enemy.blocked_by,
                flying=enemy.profile.flying,
                progress=round(enemy.progress, 4),
                weight=enemy.profile.weight,
            )
            for enemy in self.enemies.values() if enemy.alive
        ]
        operators = [
            OperatorFrame(
                index=operator.index,
                name=operator.spec.name,
                row=operator.spec.tile[0],
                col=operator.spec.tile[1],
                active=operator.active_at(time),
                blocked=len(operator.blocked),
                capacity=operator.spec.block,
                kills=operator.kills,
                damage=round(operator.damage_done, 1),
                attack_mode=operator.spec.attack_mode.value,
            )
            for operator in self.operators
        ]
        return Frame(time=time, enemies=enemies, operators=operators,
                     killed=self.killed, leaked=self.leaked, spawned=self.spawned,
                     life=self.life, max_life=self.level.life)

    def summary(self, time: float, steps: int, dt: float = 0.0) -> dict[str, Any]:
        return {
            "level_id": self.level.level_id,
            "level_name": self.level.name,
            "duration": round(time, 2),
            "steps": steps,
            "dt": round(dt, 4),
            "total_enemies": len(self._spawns),
            "spawned": self.spawned,
            "killed": self.killed,
            "leaked": self.leaked,
            "alive": sum(1 for enemy in self.enemies.values() if enemy.alive),
            "life": self.life,
            "max_life": self.level.life,
            "blocked_peak": self.blocked_peak,
            "first_spawn": round(self._spawns[0].time, 2) if self._spawns else None,
            "last_spawn": round(self._spawns[-1].time, 2) if self._spawns else None,
            "cleared_at": round(self._last_resolve_time, 2) if self.killed + self.leaked else None,
            "operators": [
                {
                    "index": operator.index,
                    "name": operator.spec.name,
                    "tile": list(operator.spec.tile),
                    "deploy_time": operator.spec.deploy_time,
                    "block": operator.spec.block,
                    "damage": round(operator.damage_done, 1),
                    "kills": operator.kills,
                    "dps": round(operator.damage_done / max(time, 1e-9), 2),
                }
                for operator in self.operators
            ],
        }


@dataclass
class SimulationResult:
    """推演结果：逐帧状态 + 事件 + 汇总。"""

    level: SimLevel
    plan: Plan
    frames: list[Frame]
    events: list[dict[str, Any]]
    summary: dict[str, Any]

    # -- 查询 -------------------------------------------------------------
    def frame_at(self, time: float) -> Frame:
        """取最接近 ``time`` 的帧。"""
        if not self.frames:
            raise ValueError("推演结果没有帧数据")
        if time <= self.frames[0].time:
            return self.frames[0]
        if time >= self.frames[-1].time:
            return self.frames[-1]
        return min(self.frames, key=lambda frame: abs(frame.time - time))

    def events_of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event.get("type") == kind]

    @property
    def duration(self) -> float:
        return self.frames[-1].time if self.frames else 0.0

    # -- 导出 -------------------------------------------------------------
    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": {
                "level_id": self.level.level_id,
                "name": self.level.name,
                "rows": self.level.rows,
                "cols": self.level.cols,
                "life": self.level.life,
            },
            "plan": self.plan.as_dict(),
            "summary": self.summary,
            "events": self.events,
        }
        if include_frames:
            payload["frames"] = [frame.as_dict() for frame in self.frames]
        return payload

    def save_json(self, path: str | Path, *, include_frames: bool = True) -> Path:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(include_frames=include_frames), handle,
                      ensure_ascii=False, indent=2)
        return target

    # -- 渲染（延迟导入，避免引擎依赖 PIL/OpenCV）--------------------------
    def render(self, time: Optional[float] = None, *, frame: Optional[Frame] = None,
               renderer: Any = None, cell: int = 64):
        """渲染某一时刻的画面，返回 ``PIL.Image``。"""
        from .render import MapRenderer

        renderer = renderer or MapRenderer(self.level, cell=cell)
        return renderer.render(frame or self.frame_at(time if time is not None else 0.0))

    def save_snapshot(self, path: str | Path, time: Optional[float] = None, *,
                      renderer: Any = None, cell: int = 64) -> Path:
        from .render import save_snapshot

        return save_snapshot(self.level, self.frame_at(time or 0.0), path,
                             renderer=renderer, cell=cell)

    def export_video(self, path: str | Path, *, fps: Optional[float] = None,
                     renderer: Any = None, cell: int = 64,
                     progress: Optional[Any] = None) -> Path:
        """导出视频（.mp4 用 OpenCV，.gif 用 PIL）。"""
        from .render import export_video

        return export_video(self.level, self.frames, path,
                            fps=fps or self.plan.fps, renderer=renderer,
                            cell=cell, progress=progress)


def simulate(
    level: SimLevel,
    operators: Optional[list[OperatorSpec]] = None,
    *,
    duration: float = 240.0,
    fps: float = 20.0,
    dt: float = 0.1,
    speed_scale: float = 1.0,
) -> SimulationResult:
    """一行完成推演的便捷入口。"""
    plan = Plan(operators=list(operators or []), duration=duration,
                fps=fps, dt=dt, speed_scale=speed_scale)
    return Simulator(level, plan).run()


__all__ = ["SimulationResult", "Simulator", "simulate"]
