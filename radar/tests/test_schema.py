# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""点选 schema 与 JSONL 读写（不依赖 ultralytics）。"""

from __future__ import annotations

import json

import pytest

from eyesofpriestess.schema import IGNORE, POINT, Click, iter_clicks, load_clicks, write_clicks


def test_point_requires_class():
    with pytest.raises(ValueError):
        Click(frame="a.jpg", x=10, y=20).validate()


def test_ignore_requires_known_reason():
    with pytest.raises(ValueError):
        Click(frame="a.jpg", x=10, y=20, type=IGNORE, reason="bad").validate()

    click = Click(frame="a.jpg", x=10, y=20, type=IGNORE, reason="occluded").validate()
    assert click.cls is None and click.is_ignore


def test_point_drops_reason():
    click = Click(frame="a.jpg", x=10, y=20, cls="B1", type=POINT, reason="blurry").validate()
    assert click.reason is None


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "clicks" / "frame.jsonl"
    original = [
        Click(frame="a.jpg", x=10, y=20, cls="B1"),
        Click(frame="a.jpg", x=30, y=40, type=IGNORE, reason="blurry"),
    ]
    assert write_clicks(path, original) == 2

    loaded = list(iter_clicks(path))
    assert [click.cls for click in loaded] == ["B1", None]
    assert loaded[1].reason == "blurry"
    assert loaded[0].to_dict()["x"] == 10


def test_load_clicks_aggregates_directory(tmp_path):
    write_clicks(tmp_path / "a.jsonl", [Click(frame="a.jpg", x=1, y=2, cls="B1")], append=False)
    write_clicks(tmp_path / "b.jsonl", [Click(frame="b.jpg", x=3, y=4, cls="A3")], append=False)

    frames = load_clicks(tmp_path)
    assert set(frames) == {"a.jpg", "b.jpg"}
    assert frames["b.jpg"][0].cls == "A3"


def test_bad_line_is_skipped(tmp_path, capsys):
    path = tmp_path / "mixed.jsonl"
    good = json.dumps(Click(frame="a.jpg", x=1, y=2, cls="B1").to_dict(), ensure_ascii=False)
    path.write_text(good + "\n{not json}\n", encoding="utf-8")

    loaded = list(iter_clicks(path))
    assert len(loaded) == 1
    assert "解析失败" in capsys.readouterr().out
