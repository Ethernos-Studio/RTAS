# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""数据按需加载、干员匹配优先级、编队分析器缓存的测试。"""

from __future__ import annotations

import logging

import pytest

from rtas.cli import CLIConfig, DetectorCommands
from rtas.data.models.operator import (Operator, OperatorProfession,
                                       OperatorRarity, PositionType)
from rtas.data.operator_matcher import OperatorMatcher
from rtas.data.providers.data_manager import DataManager

LOGGER = logging.getLogger("rtas.test")


def _operator(op_id: str, name: str, **kwargs) -> Operator:
    return Operator(id=op_id, name=name, **kwargs)


class _StubProvider:
    """只记录被调用了哪些数据源方法。"""

    def __init__(self, calls: list[str]):
        self.calls = calls

    def get_operators(self, progress_callback=None):
        self.calls.append("operators")
        return [_operator("char_002_amiya", "阿米娅")]

    def get_stages(self, progress_callback=None):
        self.calls.append("stages")
        return []

    def get_items(self, progress_callback=None):
        self.calls.append("items")
        return []

    def get_enemies(self, progress_callback=None):
        self.calls.append("enemies")
        return []


def test_load_only_operators():
    """编队识别只用干员：其余三类数据不该被读取。"""
    manager = DataManager()
    calls: list[str] = []
    manager._github_provider = _StubProvider(calls)

    assert manager.load_all_data(kinds=("operators",)) is True

    assert calls == ["operators"]
    assert manager._operators_index and not manager._stages_index
    assert not manager._items_index and not manager._enemies_index


def test_load_uses_configured_kinds():
    """initialize(kinds=...) 之后，不带参数的 load_all_data 也按这个集合加载。"""
    manager = DataManager()
    calls: list[str] = []
    manager._github_provider = _StubProvider(calls)
    manager._kinds = ("operators",)

    manager.load_all_data()

    assert calls == ["operators"]


def test_load_all_kinds_by_default():
    manager = DataManager()
    calls: list[str] = []
    manager._github_provider = _StubProvider(calls)

    manager.load_all_data()

    assert calls == ["operators", "stages", "items", "enemies"]


def test_operator_is_token():
    """trap_/token_ 前缀或 notchar 子职业都不是可部署干员。"""
    assert _operator("trap_627_xb2amy", "阿米娅", sub_profession_id="notchar2").is_token
    assert _operator("token_10002_kalts_mon3tr", "Mon3tr", sub_profession_id="notchar1").is_token
    assert not _operator("char_002_amiya", "阿米娅", sub_profession_id="caster").is_token


class _StubManager:
    def __init__(self, operators):
        self._operators = operators

    def get_operators(self):
        return self._operators


def test_matcher_prefers_real_operator_over_token():
    """`trap_627_xb2amy` 也叫「阿米娅」，不能把 char_002_amiya 顶掉。"""
    operators = [
        _operator("trap_627_xb2amy", "阿米娅",
                  profession=OperatorProfession.WARRIOR, sub_profession_id="notchar2",
                  rarity=OperatorRarity.TIER_1, position=PositionType.MELEE),
        _operator("char_002_amiya", "阿米娅",
                  profession=OperatorProfession.CASTER, sub_profession_id="caster",
                  rarity=OperatorRarity.TIER_5),
    ]
    matcher = OperatorMatcher(_StubManager(operators))
    assert matcher.initialize()

    info = matcher.get_operator_info("阿米娅", 0, 80, confidence=1.0, confidence_threshold=0.7)

    assert info["matched"] is True
    assert info["operator_id"] == "char_002_amiya"
    assert info["profession"] == "CASTER"


def test_matcher_ignores_token_only_names():
    """召唤物不再参与匹配：`Mon3tr` 这种只有 token 才有的名字不要给出干员信息。"""
    operators = [
        _operator("token_10002_kalts_mon3tr", "Mon3tr",
                  profession=OperatorProfession.WARRIOR, sub_profession_id="notchar1",
                  rarity=OperatorRarity.TIER_6),
        _operator("char_003_kalts", "凯尔希",
                  profession=OperatorProfession.MEDIC, sub_profession_id="physician",
                  rarity=OperatorRarity.TIER_6),
    ]
    matcher = OperatorMatcher(_StubManager(operators))
    assert matcher.initialize()

    info = matcher.get_operator_info("Mon3tr", 0, 90, confidence=1.0, confidence_threshold=0.7)
    assert info["matched"] is False
    # 真干员仍然正常
    info = matcher.get_operator_info("凯尔希", 0, 90, confidence=1.0, confidence_threshold=0.7)
    assert info["operator_id"] == "char_003_kalts"


def test_matcher_drops_token_entries_on_load():
    """加载阶段就把 token 剔除，索引里只剩可部署干员。"""
    operators = [
        _operator("trap_221_ftshad", "空壳", sub_profession_id="notchar2"),
        _operator("char_102_texas", "德克萨斯", sub_profession_id="pioneer"),
    ]
    matcher = OperatorMatcher(_StubManager(operators))
    assert matcher.initialize()

    assert [op.id for op in matcher._operators] == ["char_102_texas"]
    assert "空壳" not in matcher._name_index


class _FakeAnalyzer:
    """记录创建次数的假分析器。"""

    created: list["_FakeAnalyzer"] = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.options = kwargs
        self.closed = False
        _FakeAnalyzer.created.append(self)

    def initialize(self):
        return True

    def shutdown(self):
        self.closed = True


@pytest.fixture
def fake_analyzer(monkeypatch):
    _FakeAnalyzer.created = []
    import rtas.vision.squad_analyzer as module

    monkeypatch.setattr(module, "SquadAnalyzer", _FakeAnalyzer)
    return _FakeAnalyzer


def test_squad_analyzer_is_cached(fake_analyzer):
    """同一套配置只初始化一次；换布局才重建，旧实例会被释放。"""
    commands = DetectorCommands(CLIConfig(), LOGGER)

    first = commands.squad_analyzer("selection", "auto")
    again = commands.squad_analyzer("selection", "auto")

    assert first is again
    assert len(fake_analyzer.created) == 1
    assert first.options == {"potential": None, "default_potential": 0}

    other = commands.squad_analyzer("edit", "auto")
    assert other is not first
    assert len(fake_analyzer.created) == 2
    assert first.closed is True


def test_squad_analyzer_cache_reacts_to_potential_options(fake_analyzer):
    """改了 --potential / --default-potential 必须重建，不能沿用旧实例。"""
    commands = DetectorCommands(CLIConfig(), LOGGER)

    commands.squad_analyzer("edit", "auto")
    commands.squad_analyzer("edit", "auto", potential=5)
    commands.squad_analyzer("edit", "auto", potential=5, default_potential=0)  # 与上一行同配置
    commands.squad_analyzer("edit", "auto", potential=5, default_potential=5)

    assert len(fake_analyzer.created) == 3
    assert fake_analyzer.created[-1].options == {"potential": 5, "default_potential": 5}


def test_squad_analyzer_close_releases(fake_analyzer):
    commands = DetectorCommands(CLIConfig(), LOGGER)
    analyzer = commands.squad_analyzer("selection", "auto")

    commands.close()

    assert analyzer.closed is True
    assert commands._squad_analyzer is None
