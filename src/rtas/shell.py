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
# @brief 交互式模式（一次性加载，多次操作）

"""交互式模式。

CLI 每次执行都要重新加载截图/OCR 依赖，交互模式下只加载一次：

- 默认模式：带提示符 ``rtas> `` 的 REPL，输出人类可读文本；
- quiet 模式：从 stdin 逐行读命令、向 stdout 逐行写 JSON，
  适合被其它程序（如 UI、脚本）当作 stdio 服务调用。

**命令集与 CLI 完全一致**：``rtas <子命令> ...`` 怎么写，这里就怎么写，
例如 ``detect -i shot.png -f json``、``data stats``、``monitor -d 30 -n 0.5``。
命令行参数由同一个 ``argparse`` 解析器解析、由同一个 ``dispatch`` 执行，
因此参数、输出、退出码都与命令行相同；区别只是配置、日志、检测器
在会话内复用，不会每条命令重新初始化 OCR。

此外常用窗口操作提供不带参数的快捷写法（``nav`` / ``detect`` / ``ocr`` /
``goto`` / ``click`` / ``key`` / ``capture``），直接作用于会话已加载的窗口。
一旦带上 ``-`` 选项，就按 CLI 语义解析（如 ``nav -i shot.png --json``）。

quiet 模式协议：一行一条命令，返回一行 JSON（``ok`` 字段表示成功）。
日志默认写 stderr，不会污染 stdout 通道；被转发的 CLI 命令的标准输出
放在 ``output`` 字段，失败原因放在 ``error`` 字段。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import shlex
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, TextIO

from .agent import MENU_TARGETS, GameAgent

_LOGGER = logging.getLogger(__name__)

HELP_TEXT = """\
可用命令（第一部分是常驻会话快捷方式，直接作用于当前游戏窗口）:
  status                 显示窗口/权限/引擎状态
  nav                    判断主界面还是子菜单
  detect                 检测是否在战斗中
  ocr [top]              识别画面文字（可选只识别上半部分）
  goto <菜单>            进入指定界面（首页/编队/干员/终端/...）
  click <x> <y>          在截图坐标点击（自动换算到客户区）
  key <按键>             发送按键（esc/enter/f1...）
  capture [路径]         截图并保存（不给路径就存到 __temp__/arknights_<时间戳>.png）
  sleep <秒>             等待
  help [命令]            显示帮助；help <命令> 查看该命令的 CLI 用法
  quit / exit            退出

其余命令与 CLI 完全一样，带 - 选项时即按 CLI 语法解析:
  detect -i shot.png -f json        nav -i shot.png --json
  monitor -d 30 -n 0.5              capture --list
  gui -i shot.png --main-menu       main-menu -i shot.png -o out.jpg
  squad -i squad.png -j report.json level -l level.json --info
  text locate -i shot.png -q 开始   data stats / sync / operator -n "阿米娅"
  config --show                     ocr --list / --engine paddleocr
  --help                           完整的 CLI 帮助
"""

#: 常驻会话快捷命令：不带 - 选项时直接操作当前窗口（省去重新查找窗口/初始化）
SESSION_COMMANDS = frozenset({"nav", "detect", "ocr", "capture", "goto", "click", "key"})


def _has_flags(args: list[str]) -> bool:
    """参数里出现 ``-x`` / ``--xxx`` 时应当交给 CLI 解析器处理。"""
    return any(len(arg) > 1 and arg.startswith("-") for arg in args)


def _command_of(argv: list[str], names: set[str]) -> str:
    """从一整行参数里找出真正的子命令名（跳过 ``--no-gpu`` 这类全局选项）。"""
    for token in argv:
        if token in names:
            return token
    return argv[0] if argv else ""


class _WarningCollector(logging.Handler):
    """收集 RTAS 自己的告警/错误，quiet 模式下回填到 JSON 的 ``error`` 字段。"""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102 - logging 接口
        name = record.name or ""
        if name not in ("root", "") and not name.startswith("rtas"):
            return  # 忽略第三方库的告警
        try:
            self.messages.append(self.format(record))
        except Exception:  # pragma: no cover - 格式化失败时忽略即可
            pass


class InteractiveShell:
    """一次性加载资源的交互式会话。"""

    def __init__(
        self,
        agent: GameAgent,
        *,
        quiet: bool = False,
        stdout: TextIO | None = None,
        logger: logging.Logger | None = None,
        cli_config: Any = None,
    ) -> None:
        self.agent = agent
        self.quiet = quiet
        self.out = stdout or sys.stdout
        self._logger = logger
        self._cli_config = cli_config
        self._cli_commands = None  # 惰性创建，跨命令复用（含 OCR 检测器）
        self._parser: argparse.ArgumentParser | None = None
        self._cli_names: set[str] | None = None
        self._detector = None
        self._emitted = False

    @property
    def logger(self) -> logging.Logger:
        return self._logger or _LOGGER

    # -- 输出 -------------------------------------------------------------
    def _emit(self, payload: dict[str, Any], text: str = "") -> None:
        self._emitted = True
        if self.quiet:
            self.out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            self.out.write((text or payload.get("message", "")) + "\n")
        self.out.flush()

    # -- 命令解析 ---------------------------------------------------------
    @staticmethod
    def _tokenize(line: str) -> list[str]:
        """按 shell 习惯切分一行为参数列表。

        支持引号（``data operator -n "阿米娅"``），但不用反斜杠转义，
        这样 Windows 路径（``__temp__\\shot.png``）可以原样书写。
        """
        lexer = shlex.shlex(line, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ""   # 别把 # 当注释
        lexer.escape = ""       # 别把 \ 当转义
        return list(lexer)

    def _cli_parser(self) -> argparse.ArgumentParser:
        if self._parser is None:
            from .cli import create_parser

            self._parser = create_parser()
        return self._parser

    def _cli_command_names(self) -> set[str]:
        """CLI 支持的全部子命令名（与 argparse 定义自动同步）。"""
        if self._cli_names is None:
            names: set[str] = set()
            for action in self._cli_parser()._actions:
                if isinstance(action, argparse._SubParsersAction):
                    names.update(action.choices)
            self._cli_names = names
        return self._cli_names

    def _cli_context(self):
        """惰性准备 CLI 侧的配置 / 日志 / 命令对象。"""
        if self._cli_config is None or self._logger is None:
            from .cli import prepare

            config, prepared_logger = prepare(
                log_stream=sys.stderr if self.quiet else None,
            )
            self._cli_config = self._cli_config or config
            self._logger = self._logger or prepared_logger
        if self._cli_commands is None:
            from .cli import DetectorCommands

            self._cli_commands = DetectorCommands(self._cli_config, self.logger)
        return self._cli_config, self.logger, self._cli_commands

    def execute(self, line: str) -> dict[str, Any]:
        try:
            parts = self._tokenize(line)
        except ValueError as exc:
            return {"ok": False, "command": "", "error": f"命令解析失败: {exc}"}
        if not parts:
            return {"ok": True, "message": ""}

        cmd, args = parts[0], parts[1:]
        lowered = cmd.lower()

        try:
            if lowered in ("quit", "exit", "q"):
                return {"ok": True, "command": lowered, "message": "bye", "exit": True}
            if lowered in ("help", "?", "h"):
                if args and (args[0] in self._cli_command_names()
                             or args[0].startswith("-")):
                    return self._run_cli([args[0], "--help"])
                return {"ok": True, "command": lowered, "message": HELP_TEXT}
            if lowered == "shell":
                raise ValueError("已在交互模式中；如需新会话请先退出，再用 rtas shell 启动")
            if lowered == "status":
                return self._status()
            if lowered == "sleep":
                seconds = float(args[0]) if args else 1.0
                time.sleep(seconds)
                return {"ok": True, "command": lowered, "message": f"sleep {seconds}s"}

            # 常驻会话快捷方式：不带选项时用已加载的窗口/OCR
            if lowered in SESSION_COMMANDS and not _has_flags(args):
                return self._session_command(lowered, args)

            # 其余全部与 CLI 一致（data / monitor / gui / config / ocr --list ...）
            if lowered in self._cli_command_names() or cmd.startswith("-"):
                return self._run_cli(parts)

            raise ValueError(f"未知命令: {cmd}（输入 help 查看帮助）")
        except Exception as exc:  # noqa: BLE001 - 交互模式需要容错
            self.logger.debug(f"命令执行失败: {exc}")
            return {"ok": False, "command": cmd, "error": str(exc)}

    def _session_command(self, cmd: str, args: list[str]) -> dict[str, Any]:
        """执行常驻会话快捷命令。"""
        if cmd == "nav":
            return self._nav()
        if cmd == "detect":
            return self._detect()
        if cmd == "ocr":
            return self._ocr(args)
        if cmd == "capture":
            return self._capture(args)
        if cmd == "goto":
            if not args:
                raise ValueError("用法: goto <菜单名>，可用: " + "、".join(MENU_TARGETS))
            return self._goto(args[0])
        if cmd == "click":
            if len(args) < 2:
                raise ValueError("用法: click <x> <y>")
            return self._click(int(args[0]), int(args[1]))
        if cmd == "key":
            if not args:
                raise ValueError("用法: key <按键>")
            return self._key(args[0])
        raise ValueError(f"未知命令: {cmd}（输入 help 查看帮助）")

    def _run_cli(self, argv: list[str]) -> dict[str, Any]:
        """把一整行交给 CLI 执行，行为与 ``rtas <argv>`` 完全相同。"""
        from . import cli as cli_module

        # 显式指定全局选项时只影响本次执行，不污染会话缓存的配置
        one_shot = any(
            arg in ("--no-gpu", "--config", "--log-level")
            or arg.startswith(("--config=", "--log-level="))
            for arg in argv
        )
        if one_shot:
            config, cli_logger, commands = None, None, None
        else:
            config, cli_logger, commands = self._cli_context()

        captured_out, captured_err = io.StringIO(), io.StringIO()
        collector = _WarningCollector()
        # 挂在 root 上：rtas.* 与直接 logging.error(...) 的错误都能收集到
        capture_logger = logging.getLogger()
        code = 0
        capture_logger.addHandler(collector)
        try:
            kwargs = {
                "parser": self._cli_parser(),
                "cli_config": config,
                "logger": cli_logger,
                "commands": commands,
                "log_stream": sys.stderr if self.quiet else None,
            }
            if self.quiet:
                with redirect_stdout(captured_out), redirect_stderr(captured_err):
                    code = cli_module.run_cli(argv, **kwargs)
            else:
                code = cli_module.run_cli(argv, **kwargs)
        except SystemExit as exc:  # argparse 的 --help / 参数错误
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
        finally:
            capture_logger.removeHandler(collector)

        payload: dict[str, Any] = {
            "ok": code == 0,
            "command": _command_of(argv, self._cli_command_names()),
            "exit_code": code,
        }
        output = captured_out.getvalue().rstrip()
        if output:
            payload["output"] = output
        if code != 0:
            stderr_text = captured_err.getvalue().strip()
            log_text = "\n".join(collector.messages[-3:])
            # 参数解析错误只出现在 stderr；其它失败优先用日志里的原因
            detail = stderr_text if code == 2 else (log_text or stderr_text)
            payload["error"] = detail or "命令执行失败"

        if self.quiet:
            self._emit(payload)
        else:
            # CLI 自己已经把输出写到终端，这里不再重复打印
            self._emitted = True
        return payload

    # -- 具体命令 ---------------------------------------------------------
    def _status(self) -> dict[str, Any]:
        window = self.agent.window
        payload = {
            "ok": True,
            "command": "status",
            "window": window.as_dict(),
            "admin": self.agent.is_admin(),
            "backend": self.agent._backend.name,  # noqa: SLF001
            "method": self.agent.method.value,
            "dry_run": self.agent.dry_run,
        }
        text = (
            f"窗口: {window.title} ({window.width}x{window.height})\n"
            f"管理员: {'是' if payload['admin'] else '否（操作游戏建议以管理员运行）'}\n"
            f"点击方式: {payload['method']}  dry-run: {payload['dry_run']}"
        )
        self._emit(payload, text)
        return payload

    def _nav(self) -> dict[str, Any]:
        state, match = self.agent.nav()
        payload = {"ok": True, "command": "nav", "state": state.value, **match.as_dict()}
        text = f"状态: {state.value}  导航按钮: {match.center}  conf={match.confidence:.3f}"
        self._emit(payload, text)
        return payload

    def _detect(self) -> dict[str, Any]:
        if self._detector is None:
            from .vision.game_state_detector import DetectorConfig, GameStateDetector

            self._detector = GameStateDetector(
                DetectorConfig(use_gpu=self.agent._gpu),  # noqa: SLF001
                ocr_backend=self.agent.ocr,
            )
            self._detector.initialize()
        result = self._detector.detect(self.agent.screenshot(), use_smoothing=False)
        payload = {
            "ok": True,
            "command": "detect",
            "state": result.state.name,
            "confidence": round(result.confidence, 4),
            "text": result.raw_text,
        }
        self._emit(payload, f"状态: {result.state.name}  {result.raw_text!r}")
        return payload

    def _ocr(self, args: list[str]) -> dict[str, Any]:
        roi = (0.0, 0.0, 1.0, 0.45) if args and args[0] in ("top", "half", "上半") else None
        items = self.agent.read_text(roi=roi)
        payload = {
            "ok": True,
            "command": "ocr",
            "texts": [
                {"text": i.text, "confidence": round(i.confidence, 4),
                 "center": list(i.center) if i.center else None}
                for i in items
            ],
        }
        text = "\n".join(f"[{i.confidence:.2f}] {i.text} @{i.center}" for i in items)
        self._emit(payload, text or "(无文字)")
        return payload

    def _goto(self, target: str) -> dict[str, Any]:
        result = self.agent.goto(target)
        result["command"] = "goto"
        text = (
            f"进入「{target}」: {'成功' if result['ok'] else '失败'}\n"
            + "\n".join(f"  - {step}" for step in result.get("steps", []))
            + (f"\n  错误: {result.get('error')}" if not result["ok"] else "")
        )
        self._emit(result, text)
        return result

    def _click(self, x: int, y: int) -> dict[str, Any]:
        ok = self.agent.click_frame(x, y)
        payload = {"ok": ok, "command": "click", "x": x, "y": y}
        self._emit(payload, f"点击 ({x}, {y}) -> {ok}")
        return payload

    def _key(self, name: str) -> dict[str, Any]:
        ok = self.agent.key(name)
        payload = {"ok": ok, "command": "key", "key": name}
        self._emit(payload, f"按键 {name} -> {ok}")
        return payload

    def _capture(self, args: list[str]) -> dict[str, Any]:
        import cv2

        from rtas.tools.window_capture import default_temp_dir, unique_path

        image = self.agent.screenshot()
        if args:
            path = Path(args[0])
        else:
            # 不带路径时照 CLI 的规矩存进 <project>/__temp__：截图只有留在内存里等于没截
            path = unique_path(default_temp_dir())
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"截图保存失败: {path}")

        payload = {
            "ok": True,
            "command": "capture",
            "path": str(path),
            "size": [int(image.shape[1]), int(image.shape[0])],
        }
        self._emit(payload, f"截图 {image.shape[1]}x{image.shape[0]} -> {path}")
        return payload

    # -- 会话生命周期 -----------------------------------------------------
    def close(self) -> None:
        """释放会话内惰性创建的检测器。"""
        if self._cli_commands is not None:
            self._cli_commands.close()
            self._cli_commands = None
        if self._detector is not None:
            self._detector.shutdown()
            self._detector = None

    # -- 主循环 -----------------------------------------------------------
    @staticmethod
    def _ensure_utf8(stream: TextIO | None) -> None:
        """把 stdin/stdout 切到 UTF-8，避免中文命令在管道里乱码。"""
        if stream is None:
            return
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            return
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 已关闭的流
            pass

    def run(self, stdin: TextIO | None = None) -> int:
        stdin = stdin or sys.stdin
        self._ensure_utf8(stdin)
        self._ensure_utf8(self.out)
        if not self.quiet:
            self.out.write("RTAS 交互模式（输入 help 查看命令，quit 退出）\n")
            self.out.write("命令与 CLI 一致，例如: detect -i shot.png -f json\n")
            if not self.agent.is_admin():
                self.out.write("提示: 操作游戏建议以管理员身份运行本终端\n")
            self.out.flush()

        while True:
            if not self.quiet:
                self.out.write("rtas> ")
                self.out.flush()
            line = stdin.readline()
            if not line:
                break
            if not line.strip():
                continue
            self._emitted = False
            result = self.execute(line)
            if not self._emitted:
                self._emit(result)
            if result.get("exit"):
                break
        return 0


__all__ = ["InteractiveShell", "HELP_TEXT"]
