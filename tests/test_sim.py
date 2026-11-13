# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""关卡推演测试：加载器 / 阻挡 / 攻击 / 漏怪 / 渲染 / CLI。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtas.sim import (
    AttackMode,
    OperatorSpec,
    Plan,
    Simulator,
    load_level,
    load_plan,
    parse_operator_spec,
    save_snapshot,
)
from rtas.sim.render import HUD_HEIGHT, MapRenderer

REAL_LEVEL = (
    Path(__file__).resolve().parents[1]
    / "ArknightsGameData/zh_CN/gamedata/levels/obt/main/level_main_01-07.json"
)


def make_level_data(**overrides):
    """一条 1x9 的直线通道：红门在 (1,0)，蓝门在 (1,8)。"""
    tiles = [{"row": 1, "col": 0, "tileType": 3}]
    tiles += [{"row": 1, "col": col, "tileType": 1, "buildableType": 3} for col in range(1, 8)]
    tiles.append({"row": 1, "col": 8, "tileType": 4})

    data = {
        "levelId": "test_01",
        "name": "测试关",
        "mapWidth": 9,
        "mapHeight": 3,
        "tiles": tiles,
        "routes": [{"id": "r1", "points": [{"row": 1, "col": 0}, {"row": 1, "col": 8}]}],
        "waves": [{
            "id": "w1",
            "spawns": [{"enemyId": "mob", "count": 3, "interval": 1.0, "routeId": "r1"}],
        }],
        "enemies": {"mob": {"name": "小怪", "hp": 1000, "speed": 1.0, "weight": 1}},
        "life": 5,
    }
    data.update(overrides)
    return data


@pytest.fixture
def level():
    return load_level(data=make_level_data())


# -- 加载 ---------------------------------------------------------------------

def test_load_simple_level(level):
    assert (level.rows, level.cols) == (3, 9)
    assert level.total_enemies == 3
    assert [round(spawn.time, 2) for spawn in level.spawns] == [0.0, 1.0, 2.0]
    assert level.routes["r1"].length == pytest.approx(8.0)
    assert level.tile(1, 0).kind.value == "start"
    assert level.tile(1, 8).kind.value == "end"
    assert (1, 4) in level.deployable("melee")


def test_load_rejects_unknown_format():
    from rtas.sim import LevelDataError

    with pytest.raises(LevelDataError):
        load_level(data={"foo": 1})


def test_load_game_format_with_astar_route():
    """原版格式：没有 checkpoints 时用 A* 在可通行地块上补全路线。"""
    tiles = [
        {"tileKey": "tile_start", "buildableType": "NONE", "passableMask": "ALL",
         "heightType": "LOWLAND"},
        {"tileKey": "tile_road", "buildableType": "MELEE", "passableMask": "ALL",
         "heightType": "LOWLAND"},
        {"tileKey": "tile_end", "buildableType": "NONE", "passableMask": "ALL",
         "heightType": "LOWLAND"},
        {"tileKey": "tile_forbidden", "buildableType": "NONE", "passableMask": "FLY_ONLY",
         "heightType": "HIGHLAND"},
    ]
    game_data = {
        "mapId": "test",
        "options": {"maxLifePoint": 3, "moveMultiplier": 0.5},
        "mapData": {
            "map": [[3, 3, 3], [0, 1, 2], [3, 3, 3]],
            "tiles": tiles,
        },
        "routes": [{
            "startPosition": {"row": 1, "col": 0},
            "endPosition": {"row": 1, "col": 2},
            "checkpoints": None,
            "allowDiagonalMove": True,
        }],
        "enemyDbRefs": [],
        "waves": [{
            "preDelay": 0.0,
            "fragments": [{
                "preDelay": 0.0,
                "actions": [{"actionType": "SPAWN", "key": "enemy_x", "count": 1,
                             "interval": 1.0, "routeIndex": 0, "preDelay": 0.0}],
            }],
        }],
    }
    level = load_level(data=game_data)
    assert level.rows == 3 and level.cols == 3
    assert level.life == 3
    assert level.routes["0"].points == [(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]
    assert level.tile(1, 0).kind.value == "start"
    assert level.tile(1, 2).kind.value == "end"
    # 未知敌人使用默认档案，速度乘以 moveMultiplier
    assert level.profile("enemy_x").speed == pytest.approx(0.7 * 0.5)


@pytest.mark.skipif(not REAL_LEVEL.exists(), reason="需要 ArknightsGameData")
def test_load_real_level():
    level = load_level(REAL_LEVEL)
    assert level.total_enemies > 0
    assert len(level.routes) > 3
    assert level.last_spawn_time > 0
    assert all(route.length > 2 for route in level.routes.values())
    assert level.profile("enemy_1007_slime_2").hp > 0
    assert level.profile("enemy_1007_slime_2").weight >= 1


# -- 引擎 ---------------------------------------------------------------------

def test_unblocked_enemies_leak(level):
    result = Simulator(level).run(duration=30)
    assert result.summary["spawned"] == 3
    assert result.summary["killed"] == 0
    assert result.summary["leaked"] == 3
    assert result.summary["life"] == 2          # 生命 5，每个漏怪扣 1
    assert result.summary["blocked_peak"] == 0


def test_blocker_holds_and_kills(level):
    plan = Plan(duration=40)
    plan.add(OperatorSpec(name="近卫", tile=(1, 4), deploy_time=0, block=1,
                          damage=1000, attack_interval=1.0))
    result = Simulator(level, plan).run()

    assert result.summary["blocked_peak"] == 1
    assert result.summary["killed"] >= 1
    assert result.summary["leaked"] < 3
    # 被阻挡的敌人不会越过干员所在格
    for frame in result.frames:
        for enemy in frame.enemies:
            if enemy.blocked_by is not None:
                assert (round(enemy.row), round(enemy.col)) == (1, 4)


def test_block_capacity_is_respected(level):
    """阻挡 1 的干员同一时间只能挡住一个敌人。"""
    plan = Plan(duration=40)
    plan.add(OperatorSpec(name="近卫", tile=(1, 4), block=1, damage=1, attack_interval=10.0))
    result = Simulator(level, plan).run()
    for frame in result.frames:
        assert frame.blocked <= 1


def test_flying_enemy_ignores_block(level):
    data = make_level_data(enemies={"mob": {"hp": 1000, "speed": 1.0, "flying": True}})
    flying_level = load_level(data=data)
    plan = Plan(duration=30)
    plan.add(OperatorSpec(name="近卫", tile=(1, 4), block=3, damage=0.1,
                          attack_interval=1.0))
    result = Simulator(flying_level, plan).run()
    assert result.summary["blocked_peak"] == 0
    assert result.summary["leaked"] == 3


def test_retreat_releases_block(level):
    plan = Plan(duration=30)
    plan.add(OperatorSpec(name="近卫", tile=(1, 4), deploy_time=0, retreat_time=6,
                          block=1, damage=1, attack_interval=10.0))
    result = Simulator(level, plan).run()
    after = [frame for frame in result.frames if frame.time > 6.5]
    assert after and all(frame.blocked == 0 for frame in after)
    assert any(event["type"] == "retreat" for event in result.events)


def test_ranged_operator_attacks_without_blocking(level):
    plan = Plan(duration=40)
    plan.add(OperatorSpec(name="狙击", tile=(1, 4), block=0, damage=1200,
                          attack_interval=0.5, attack_mode=AttackMode.RANGED,
                          range_tiles=3.0))
    result = Simulator(level, plan).run()
    assert result.summary["blocked_peak"] == 0     # 远程不阻挡
    assert result.summary["killed"] == 3


def test_defense_reduces_damage(level):
    data = make_level_data(enemies={"mob": {"hp": 100, "speed": 1.0, "defense": 100}})
    tanky = load_level(data=data)
    plan = Plan(duration=20)
    plan.add(OperatorSpec(name="近卫", tile=(1, 4), block=1, damage=100,
                          attack_interval=0.5))
    result = Simulator(tanky, plan).run()
    # 100 伤害打 100 防御 → 只保底 5%，需要 20 次攻击
    assert result.summary["killed"] <= 2


def test_plan_duration_caps_simulation(level):
    plan = Plan(duration=3.5)
    result = Simulator(level, plan).run()
    assert result.duration <= 3.5 + 0.2


def test_frame_rate_matches_video_speed(level):
    """dt 大于帧间隔时会被压到 1/fps，保证导出视频不会变成快放。"""
    plan = Plan(duration=10, fps=20, dt=0.5)
    result = Simulator(level, plan).run()
    expected = result.duration * 20
    assert abs(len(result.frames) - expected) <= 3


def test_simulate_helper(level):
    from rtas.sim import simulate

    result = simulate(level, duration=6, fps=10)
    assert result.summary["spawned"] >= 1
    assert result.frames and result.frames[-1].time <= 6.5


def test_result_serialization(level, tmp_path):
    result = Simulator(level, Plan(duration=15)).run()
    payload = result.to_dict(include_frames=True)
    text = json.dumps(payload, ensure_ascii=False)
    assert "frames" in payload and "summary" in payload
    assert "测试关" in text

    path = result.save_json(tmp_path / "out.json", include_frames=False)
    assert path.exists()
    assert "frames" not in json.loads(path.read_text(encoding="utf-8"))


# -- 渲染 ---------------------------------------------------------------------

def test_render_snapshot_size(level, tmp_path):
    result = Simulator(level, Plan(duration=5)).run()
    renderer = MapRenderer(level, cell=40)
    assert (renderer.width, renderer.height) == (level.cols * 40, level.rows * 40 + HUD_HEIGHT)

    image = result.render(time=2, cell=40)
    assert image.size == (renderer.width, renderer.height)

    target = save_snapshot(level, result.frame_at(2), tmp_path / "shot.png", cell=40)
    assert target.exists() and target.stat().st_size > 0


def test_export_gif(level, tmp_path):
    from rtas.sim import export_video

    result = Simulator(level, Plan(duration=6, fps=5)).run()
    path = export_video(level, result.frames, tmp_path / "run.gif", fps=5, cell=32)
    assert path.exists() and path.stat().st_size > 0


# -- 计划解析 -----------------------------------------------------------------

def test_parse_operator_spec():
    spec = parse_operator_spec(
        "name=近卫,t=5,tile=3:2,block=2,atk=600,interval=1.5,mode=ranged,range=2.5"
    )
    assert spec.name == "近卫"
    assert spec.tile == (3, 2)
    assert spec.deploy_time == 5
    assert spec.block == 2
    assert spec.damage == 600
    assert spec.attack_interval == 1.5
    assert spec.attack_mode is AttackMode.RANGED
    assert spec.range_tiles == 2.5


def test_parse_operator_spec_rejects_bad_field():
    with pytest.raises(ValueError):
        parse_operator_spec("近卫 5 3:2")


def test_load_plan(tmp_path):
    target = tmp_path / "plan.json"
    target.write_text(json.dumps({
        "duration": 100,
        "operators": [{"name": "近卫", "tile": [2, 3], "deploy_time": 4,
                       "block": 2, "damage": 800}],
    }), encoding="utf-8")
    plan = load_plan(target)
    assert plan.duration == 100
    assert plan.operators[0].tile == (2, 3)
    assert plan.operators[0].damage == 800


# -- CLI ----------------------------------------------------------------------

def test_cli_sim_map_and_run(level, tmp_path):
    from rtas import cli

    level_path = tmp_path / "level.json"
    level_path.write_text(json.dumps(make_level_data(), ensure_ascii=False), encoding="utf-8")

    code = cli.run_cli(["sim", "map", "-l", str(level_path), "-t", "3", "--json"])
    assert code == 0

    image = tmp_path / "t3.png"
    code = cli.run_cli(["sim", "map", "-l", str(level_path), "-t", "3", "-o", str(image)])
    assert code == 0 and image.exists()

    data_path = tmp_path / "result.json"
    code = cli.run_cli([
        "sim", "run", "-l", str(level_path),
        "--op", "name=近卫,t=0,tile=1:4,block=1,damage=900,interval=1.0",
        "-t", "20", "--json", str(data_path), "--no-frames",
    ])
    assert code == 0 and data_path.exists()
    summary = json.loads(data_path.read_text(encoding="utf-8"))["summary"]
    assert summary["spawned"] == 3


def test_cli_sim_tiles_writes_plan_template(tmp_path):
    from rtas import cli

    level_path = tmp_path / "level.json"
    level_path.write_text(json.dumps(make_level_data(), ensure_ascii=False), encoding="utf-8")
    template = tmp_path / "plan.json"

    code = cli.run_cli(["sim", "tiles", "-l", str(level_path),
                        "--plan-template", str(template)])
    assert code == 0 and template.exists()
    plan = json.loads(template.read_text(encoding="utf-8"))
    assert plan["operators"] and "tile" in plan["operators"][0]


def test_cli_sim_reports_missing_level(tmp_path):
    from rtas import cli

    code = cli.run_cli(["sim", "map", "-l", str(tmp_path / "nope.json")])
    assert code == 1
