# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""自动标注转换器：宽容解析 + 编号映射。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from auto_label_import import build_code_map, load_entries, pick, resolve_code  # noqa: E402


def test_pick_handles_aliases_and_case():
    entry = {"EnemyId": "enemy_1000_gopro", "SX": 980, "SY": 760, "T": 9.02}

    assert pick(entry, "name") == "enemy_1000_gopro"
    assert pick(entry, "screen_x") == 980
    assert pick(entry, "screen_y") == 760
    assert pick(entry, "time") == 9.02


def test_pick_returns_none_for_missing():
    assert pick({}, "screen_x") is None


def test_load_entries_json_array(tmp_path):
    path = tmp_path / "dump.json"
    path.write_text(json.dumps([{"t": 1.0}, {"t": 2.0}]), encoding="utf-8")

    assert len(load_entries(path)) == 2


def test_load_entries_jsonl(tmp_path):
    path = tmp_path / "dump.jsonl"
    path.write_text('{"t": 1.0}\n{"t": 2.0}\n', encoding="utf-8")

    assert len(load_entries(path)) == 2


def test_load_entries_wrapped_key(tmp_path):
    """工具常把数组包在 units/enemies 之类的键里。"""
    path = tmp_path / "dump.json"
    path.write_text(json.dumps({"units": [{"t": 1.0}, {"t": 2.0}, {"t": 3.0}]}), encoding="utf-8")

    assert len(load_entries(path)) == 3


def test_load_entries_csv(tmp_path):
    path = tmp_path / "dump.csv"
    path.write_text("t,key,sx,sy\n9.02,enemy_1000_gopro,980,760\n", encoding="utf-8")

    entries = load_entries(path)
    assert len(entries) == 1
    assert entries[0]["key"] == "enemy_1000_gopro"


def test_resolve_code_by_enemy_id_short_name_and_unknown():
    by_id, codes = build_code_map()

    assert resolve_code("enemy_1000_gopro", by_id, codes) == "O1"
    assert resolve_code("gopro", by_id, codes) == "O1"
    assert resolve_code("O1", by_id, codes) == "O1"
    assert resolve_code("enemy_9999_mystery", by_id, codes) == "other_enemy"
    assert resolve_code(None, by_id, codes) == "other_enemy"


def test_code_map_covers_catalog():
    """类别表里所有编号都应该能被映射出来（否则自动标注会大面积降级成 other_enemy）。"""
    catalog = json.loads((Path(__file__).resolve().parents[1] / "configs" / "enemies.json")
                         .read_text(encoding="utf-8"))
    by_id, codes = build_code_map()
    missing = [code for code in catalog["enemies"] if code not in codes]
    assert not missing, missing
