# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""图像类命令「省略 -i 直接截图」的测试（不依赖真实游戏窗口）。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest
from PIL import Image

import rtas.cli as cli
from rtas.tools import window_capture
from rtas.tools.window_capture import CaptureError, Region, WindowInfo

LOGGER = logging.getLogger("rtas.test")


def _window(hwnd: int, title: str, width: int, height: int,
            process: str = "arknights.exe") -> WindowInfo:
    rect = Region(0, 0, width, height)
    return WindowInfo(
        hwnd=hwnd,
        title=title,
        class_name="UnityWndClass",
        process=process,
        rect=rect,
        client_rect=rect,
        minimized=False,
    )


def _fake_args(**overrides) -> argparse.Namespace:
    values = {"window": None, "activate": False, "image": None}
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def captured(tmp_path, monkeypatch):
    """把窗口查找与截图都换成桩，返回记录下来的窗口与落盘目录。"""
    calls: dict[str, object] = {"windows": [], "activated": None}

    def fake_find(keywords=None, **kwargs):
        calls["keywords"] = keywords
        return calls["windows"]

    def fake_capture(window, *, activate=False, delay=0.0, client_area=True):
        calls["captured"] = window
        calls["activated"] = activate
        return Image.new("RGB", (1280, 720), (20, 20, 24))

    monkeypatch.setattr(window_capture, "find_windows", fake_find)
    monkeypatch.setattr(window_capture, "capture_window", fake_capture)
    monkeypatch.setattr(window_capture, "default_temp_dir", lambda: tmp_path / "shots")
    return calls


def test_grab_game_window_picks_largest_and_saves(captured):
    captured["windows"] = [
        _window(1, "明日方舟 - MuMu 启动器", 320, 200),
        _window(2, "明日方舟", 1280, 720),
    ]

    path = cli.grab_game_window(_fake_args(activate=True), LOGGER)

    assert path.exists() and path.parent.name == "shots"
    assert path.suffix == ".png"
    assert captured["captured"].hwnd == 2      # 客户区最大的那个
    assert captured["activated"] is True
    with Image.open(path) as image:
        assert image.size == (1280, 720)


def test_grab_game_window_uses_keyword(captured):
    captured["windows"] = [_window(3, "BlueStacks", 1280, 720)]

    cli.grab_game_window(_fake_args(window="BlueStacks"), LOGGER)

    assert captured["keywords"] == ["BlueStacks"]


def test_grab_game_window_ignores_editor_windows(captured):
    """标题里带 arknights_xxx.png 的 VS Code 窗口不能当成游戏。"""
    editor = _window(9, "arknights_20261113_192526.png - RTAS - Visual Studio Code",
                     1902, 1011, process="Code.exe")
    captured["windows"] = [editor]

    with pytest.raises(CaptureError):
        cli.grab_game_window(_fake_args(), LOGGER)


def test_grab_game_window_prefers_game_process(captured):
    """模拟器进程 + 「明日方舟」标题的可信度高于仅标题沾边的窗口。"""
    viewer = _window(4, "arknights 截图查看器", 1600, 900, process="imageglass.exe")
    game = _window(5, "明日方舟", 1280, 720, process="HD-Player.exe")
    captured["windows"] = [viewer, game]

    cli.grab_game_window(_fake_args(), LOGGER)

    assert captured["captured"].hwnd == 5


def test_grab_game_window_reports_missing_window(captured):
    captured["windows"] = []

    with pytest.raises(CaptureError) as excinfo:
        cli.grab_game_window(_fake_args(), LOGGER)

    assert "没有找到游戏窗口" in str(excinfo.value)
    assert "capture --list" in str(excinfo.value)


def test_screenshot_commands_are_the_image_ones():
    assert cli.SCREENSHOT_COMMANDS == {
        'detect', 'nav', 'ocr', 'gui', 'main-menu', 'squad', 'potential',
    }


def test_dispatch_captures_when_image_missing(tmp_path, monkeypatch):
    """省略 -i 时先截图，再把截到的路径交给命令。"""
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"fake")

    seen: dict[str, object] = {}
    monkeypatch.setattr(cli, "grab_game_window", lambda args, logger: shot)
    monkeypatch.setattr(
        cli, "run_nav_command",
        lambda args, logger: (seen.update(image=args.image), 0)[1],
    )

    parser = cli.create_parser()
    args = parser.parse_args(["nav", "--json"])
    code = cli.dispatch(args, parser, cli_config=cli.CLIConfig(), logger=LOGGER)

    assert code == 0
    assert seen["image"] == shot


def test_dispatch_keeps_given_image(tmp_path, monkeypatch):
    """给了 -i 就不该再去截图。"""
    shot = tmp_path / "given.png"
    shot.write_bytes(b"fake")

    monkeypatch.setattr(
        cli, "grab_game_window",
        lambda args, logger: pytest.fail("不应该截图"),
    )
    seen: dict[str, object] = {}
    seen["image"] = None
    monkeypatch.setattr(
        cli, "run_nav_command",
        lambda args, logger: (seen.update(image=args.image), 0)[1],
    )

    parser = cli.create_parser()
    args = parser.parse_args(["nav", "-i", str(shot)])
    assert cli.dispatch(args, parser, cli_config=cli.CLIConfig(), logger=LOGGER) == 0
    assert Path(seen["image"]).name == "given.png"


def test_dispatch_returns_error_when_capture_fails(monkeypatch):
    def boom(args, logger):
        raise CaptureError("没有找到游戏窗口")

    monkeypatch.setattr(cli, "grab_game_window", boom)

    parser = cli.create_parser()
    args = parser.parse_args(["squad"])
    assert cli.dispatch(args, parser, cli_config=cli.CLIConfig(), logger=LOGGER) == 1


def test_ocr_list_does_not_capture(monkeypatch):
    """`ocr --list` 只列引擎，不该去抓窗口。"""
    monkeypatch.setattr(
        cli, "grab_game_window",
        lambda args, logger: pytest.fail("--list 不应截图"),
    )

    parser = cli.create_parser()
    args = parser.parse_args(["ocr", "--list"])
    assert cli.dispatch(args, parser, cli_config=cli.CLIConfig(), logger=LOGGER) == 0
