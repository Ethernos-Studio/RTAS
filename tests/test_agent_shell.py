# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GameAgent / 交互式 shell 测试（用假后端，不接触真实游戏窗口）。"""

from __future__ import annotations

import io
import json
import sys

import numpy as np
import pytest

from rtas import agent as agent_module
from rtas.agent import GameAgent
from rtas.platform import ClickMethod, InputBackend, WindowRef
from rtas.platform.windows import input as win_input
from rtas.shell import InteractiveShell
from rtas.vision.nav_detector import NavMenuButtonMatch, NavigationState
from rtas.vision.ocr.base import OCRText


class FakeBackend(InputBackend):
    name = "fake"

    def __init__(self, image=None):
        self.image = np.zeros((720, 1280, 3), dtype=np.uint8) if image is None else image
        self.clicks: list[tuple[int, int, ClickMethod]] = []
        self.keys: list[tuple[str, ClickMethod]] = []
        self.window_ref = WindowRef(
            hwnd=0x1234, title="明日方舟", process="Arknights.exe",
            class_name="UnityWndClass", width=1280, height=720,
        )

    def find_window(self, keywords=None, hwnd=None):
        return self.window_ref

    def click(self, window, x, y, method=ClickMethod.WINDOW):
        self.clicks.append((int(x), int(y), ClickMethod(method)))
        return True

    def key(self, window, key, method=ClickMethod.WINDOW):
        self.keys.append((key, ClickMethod(method)))
        return True

    def capture(self, window):
        return self.image.copy()

    def is_admin(self):
        return False


class FakeOCR:
    def __init__(self, texts):
        self.texts = texts

    def recognize(self, image):
        return list(self.texts)

    def shutdown(self):
        pass


def _match(found=True, center=(233, 36)):
    return NavMenuButtonMatch(
        found=found, confidence=0.9 if found else 0.1,
        center=center if found else None, bbox=None, scale=1.0,
    )


@pytest.fixture
def fake_agent(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(agent_module, "get_backend", lambda: backend)
    agent = GameAgent()
    agent._window = backend.window_ref
    return agent, backend


def test_click_method_defaults_to_window(fake_agent):
    agent, backend = fake_agent
    assert agent.method is ClickMethod.WINDOW
    assert agent.click(10, 20) is True
    assert backend.clicks == [(10, 20, ClickMethod.WINDOW)]


def test_click_dry_run_does_not_touch_backend(fake_agent):
    agent, backend = fake_agent
    assert agent.click(10, 20, dry_run=True) is True
    assert backend.clicks == []


def test_key_uses_configured_method(fake_agent):
    agent, backend = fake_agent
    assert agent.key("esc") is True
    assert backend.keys == [("esc", ClickMethod.WINDOW)]


def test_screenshot_records_frame_size(fake_agent):
    agent, _ = fake_agent
    agent._backend.image = np.zeros((711, 1264, 3), dtype=np.uint8)
    image = agent.screenshot()
    assert image.shape[:2] == (711, 1264)
    assert (agent.window.width, agent.window.height) == (1280, 720)
    assert agent._frame_size == (1264, 711)


def test_to_client_scales_frame_coordinates(fake_agent, monkeypatch):
    agent, _ = fake_agent
    agent._backend.image = np.zeros((711, 1264, 3), dtype=np.uint8)
    agent.screenshot()
    # 截图坐标按比例换算到客户区
    assert agent.to_client(632, 355) == (round(632 * 1280 / 1264), round(355 * 720 / 711))
    assert agent.to_client(0, 0) == (0, 0)
    assert agent.to_client(1263, 710) == (
        round(1263 * 1280 / 1264),
        round(710 * 720 / 711),
    )

    # 未截图时按原值返回
    backend = FakeBackend()
    monkeypatch.setattr(agent_module, "get_backend", lambda: backend)
    fresh = GameAgent()
    assert fresh.to_client(10, 20) == (10, 20)


def test_click_frame_scales_before_clicking(fake_agent):
    agent, backend = fake_agent
    agent._backend.image = np.zeros((711, 1264, 3), dtype=np.uint8)
    agent.screenshot()
    agent.click_frame(632, 355)
    assert backend.clicks == [
        (round(632 * 1280 / 1264), round(355 * 720 / 711), ClickMethod.WINDOW)
    ]


def test_find_text_prefers_best_match(fake_agent, monkeypatch):
    agent, _ = fake_agent
    monkeypatch.setattr(
        GameAgent, "ocr",
        property(lambda self: FakeOCR([
            OCRText("编队", 0.9, [[100, 100], [160, 100], [160, 140], [100, 140]]),
            OCRText("干员", 0.9, [[300, 100], [360, 100], [360, 140], [300, 140]]),
        ])),
    )
    hit = agent.find_text("编队")
    assert hit is not None and hit.text == "编队"
    assert agent.find_text("不存在") is None


def test_read_text_roi_offsets_bbox(fake_agent, monkeypatch):
    agent, _ = fake_agent
    monkeypatch.setattr(
        GameAgent, "ocr",
        property(lambda self: FakeOCR([
            OCRText("首页", 0.8, [[10, 10], [50, 10], [50, 30], [10, 30]]),
        ])),
    )
    items = agent.read_text(roi=(0.0, 0.0, 0.5, 0.5))
    assert items[0].bbox[0] == [10, 10]

    items = agent.read_text(roi=(0.25, 0.25, 0.75, 0.75))
    assert items[0].bbox[0] == [10 + 320, 10 + 180]


def test_goto_main_menu_shortcut(fake_agent, monkeypatch):
    agent, backend = fake_agent
    monkeypatch.setattr(GameAgent, "nav", lambda self, image=None: (NavigationState.MAIN_MENU, _match()))
    result = agent.goto("首页")
    assert result["ok"] is True
    assert result["steps"] == [{"action": "already_main_menu"}]
    assert backend.clicks == []


def test_goto_opens_nav_bar_then_clicks_target(fake_agent, monkeypatch):
    agent, backend = fake_agent
    states = [(NavigationState.SUB_MENU, _match()), (NavigationState.SUB_MENU, _match())]
    monkeypatch.setattr(GameAgent, "nav", lambda self, image=None: states.pop(0))
    monkeypatch.setattr(
        GameAgent, "find_text",
        lambda self, target, image=None, **kw: OCRText(
            target, 0.95, [[900, 60], [980, 60], [980, 100], [900, 100]]
        ),
    )
    agent.action_delay = 0
    result = agent.goto("编队")
    assert result["ok"] is True
    assert backend.clicks[0][:2] == (233, 36)
    assert backend.clicks[1][:2] == (940, 80)
    assert [step["action"] for step in result["steps"]] == ["open_nav_bar", "click_target"]


def test_goto_reports_missing_target(fake_agent, monkeypatch):
    agent, _ = fake_agent
    monkeypatch.setattr(GameAgent, "nav", lambda self, image=None: (NavigationState.MAIN_MENU, _match(False)))
    monkeypatch.setattr(GameAgent, "find_text", lambda *a, **kw: None)
    agent.action_delay = 0
    result = agent.goto("不存在", attempts=1)
    assert result["ok"] is False
    assert "不存在" in result["error"]


def test_goto_dry_run_stops_after_nav_bar(fake_agent, monkeypatch):
    agent, backend = fake_agent
    agent.dry_run = True
    monkeypatch.setattr(GameAgent, "nav", lambda self, image=None: (NavigationState.SUB_MENU, _match()))
    result = agent.goto("编队")
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert backend.clicks == []


def test_shell_quiet_mode_emits_json_lines(fake_agent):
    agent, backend = fake_agent
    out = io.StringIO()
    shell = InteractiveShell(agent, quiet=True, stdout=out)
    stdin = io.StringIO("click 10 20\nkey esc\nbogus\nquit\n")
    assert shell.run(stdin) == 0

    lines = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [line.get("command") for line in lines] == ["click", "key", "bogus", "quit"]
    assert lines[0]["ok"] is True and lines[0]["x"] == 10
    assert lines[2]["ok"] is False and "未知命令" in lines[2]["error"]
    assert lines[3]["exit"] is True
    assert backend.clicks == [(10, 20, ClickMethod.WINDOW)]
    assert backend.keys == [("esc", ClickMethod.WINDOW)]


def test_shell_human_mode_writes_prompt(fake_agent):
    agent, _ = fake_agent
    out = io.StringIO()
    shell = InteractiveShell(agent, quiet=False, stdout=out)
    shell.run(io.StringIO("help\nquit\n"))
    text = out.getvalue()
    assert "RTAS 交互模式" in text
    assert "rtas> " in text
    assert "goto <菜单>" in text


def test_shell_status_reports_window_and_method(fake_agent):
    agent, _ = fake_agent
    out = io.StringIO()
    shell = InteractiveShell(agent, quiet=True, stdout=out)
    shell.execute("status")
    payload = json.loads(out.getvalue().splitlines()[0])
    assert payload["window"]["hwnd"] == 0x1234
    assert payload["method"] == "window"
    assert payload["backend"] == "fake"


def test_shell_rejects_bad_arguments(fake_agent):
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    assert shell.execute("click 1")["ok"] is False
    assert shell.execute("goto")["ok"] is False
    assert shell.execute("key")["ok"] is False


# -- shell 与 CLI 的命令集一致 -------------------------------------------------

def test_shell_runs_cli_command_with_flags(fake_agent):
    """带 - 选项的命令交给 CLI 解析，stdout 收进 JSON 的 output 字段。"""
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    payload = shell.execute("config --show")
    assert payload["ok"] is True and payload["exit_code"] == 0
    assert "当前配置" in payload["output"]
    shell.close()


def test_shell_accepts_global_options_before_command(fake_agent, tmp_path):
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    payload = shell.execute(f"--config {tmp_path / 'config.json'} config --show")
    assert payload["ok"] is True, payload
    assert "enable_gpu" in payload["output"]
    shell.close()


def test_shell_reports_cli_argument_errors(fake_agent):
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    payload = shell.execute("config --badflag")
    assert payload["ok"] is False
    assert payload["exit_code"] == 2
    assert "unrecognized arguments" in payload["error"]
    shell.close()


def test_shell_help_for_cli_subcommand(fake_agent):
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    payload = shell.execute("help detect")
    assert payload["ok"] is True
    assert "usage: rtas detect" in payload["output"]
    shell.close()


def test_shell_prefers_session_shortcut_without_flags(fake_agent, monkeypatch):
    """不带参数用常驻会话；一旦带选项就按 CLI 语法解析。"""
    agent, _ = fake_agent
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    monkeypatch.setattr(
        GameAgent, "nav",
        lambda self, image=None: (NavigationState.MAIN_MENU, _match(False)),
    )
    payload = shell.execute("nav")
    assert payload["ok"] is True and payload["state"] == "main_menu"

    failed = shell.execute("nav --json")  # CLI 的 nav 需要 -i
    assert failed["ok"] is False and failed["exit_code"] == 2
    assert "-i/--image" in failed["error"]
    shell.close()


def test_shell_reports_command_failure_from_logging(fake_agent, tmp_path):
    """转发的命令失败时，error 取自日志里的真实原因。"""
    agent, _ = fake_agent
    missing = tmp_path / "missing.png"
    shell = InteractiveShell(agent, quiet=True, stdout=io.StringIO())
    payload = shell.execute(f'nav -i "{missing}"')
    assert payload["ok"] is False and payload["exit_code"] == 1
    assert "不存在" in payload["error"]
    shell.close()


def test_shell_runs_image_cli_command_quiet(fake_agent, tmp_path):
    """shell 内执行 CLI 图像命令，协议通道只有一行 JSON。"""
    import cv2

    agent, _ = fake_agent
    image_path = tmp_path / "shot.png"
    cv2.imwrite(str(image_path), np.zeros((720, 1280, 3), dtype=np.uint8))

    out = io.StringIO()
    shell = InteractiveShell(agent, quiet=True, stdout=out)
    payload = shell.execute(f'nav -i "{image_path}" --json')
    assert payload["ok"] is True, payload
    assert json.loads(payload["output"])["state"] in {"main_menu", "sub_menu", "unknown"}
    assert len(out.getvalue().splitlines()) == 1
    shell.close()


def test_shell_tokenizer_keeps_windows_paths_and_quotes():
    assert InteractiveShell._tokenize(r'data operator -n "阿米娅"') == [
        "data", "operator", "-n", "阿米娅",
    ]
    assert InteractiveShell._tokenize(r"nav -i __temp__\shot.png") == [
        "nav", "-i", r"__temp__\shot.png",
    ]
    assert InteractiveShell._tokenize("   ") == []
    with pytest.raises(ValueError):
        InteractiveShell._tokenize('key "esc')


@pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 有 Win32 输入实现")
def test_pack_xy_roundtrip():
    value = win_input.pack_xy(300, 500)
    assert value & 0xFFFF == 300
    assert value >> 16 == 500


@pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 有 Win32 输入实现")
def test_virtual_key_names():
    assert win_input.virtual_key("esc") == 0x1B
    assert win_input.virtual_key("F1") == 0x70
    assert win_input.virtual_key("a") == ord("A")
    with pytest.raises(ValueError):
        win_input.virtual_key("nope")


@pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 有提权实现")
def test_build_command_uses_current_interpreter():
    from rtas.platform.windows.elevate import build_command

    executable, params = build_command(["rtas", "goto", "采购中心", "--dry-run"])
    assert executable == sys.executable
    assert params.startswith("-m rtas goto")
    assert params.endswith("--dry-run")


@pytest.mark.skipif(sys.platform != "win32", reason="仅 Windows 有提权实现")
def test_build_command_strips_elevate_flag():
    from rtas.platform.windows.elevate import build_command

    _, params = build_command(["rtas", "--elevate", "shell", "-q"])
    assert "--elevate" not in params
    assert "shell -q" in params
