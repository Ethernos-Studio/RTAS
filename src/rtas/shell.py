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

quiet 模式协议：一行一条命令，返回一行 JSON（``ok`` 字段表示成功）。
日志默认写 stderr，不会污染 stdout 通道。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, TextIO

from .agent import MENU_TARGETS, GameAgent

logger = logging.getLogger(__name__)

HELP_TEXT = """\
可用命令:
  status                 显示窗口/权限/引擎状态
  nav                    判断主界面还是子菜单
  detect                 检测是否在战斗中
  ocr [top]              识别画面文字（可选只识别上半部分）
  goto <菜单>            进入指定界面（首页/编队/干员/终端/...）
  click <x> <y>          在截图坐标点击（自动换算到客户区）
  key <按键>             发送按键（esc/enter/f1...）
  capture [路径]         截图并保存
  sleep <秒>             等待
  help                   显示帮助
  quit / exit            退出
"""


class InteractiveShell:
    """一次性加载资源的交互式会话。"""

    def __init__(
        self,
        agent: GameAgent,
        *,
        quiet: bool = False,
        stdout: TextIO | None = None,
    ) -> None:
        self.agent = agent
        self.quiet = quiet
        self.out = stdout or sys.stdout
        self._detector = None
        self._emitted = False

    # -- 输出 -------------------------------------------------------------
    def _emit(self, payload: dict[str, Any], text: str = "") -> None:
        self._emitted = True
        if self.quiet:
            self.out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            self.out.write((text or payload.get("message", "")) + "\n")
        self.out.flush()

    # -- 命令解析 ---------------------------------------------------------
    def execute(self, line: str) -> dict[str, Any]:
        parts = line.strip().split()
        if not parts:
            return {"ok": True, "message": ""}
        cmd, args = parts[0].lower(), parts[1:]

        try:
            if cmd in ("quit", "exit", "q"):
                return {"ok": True, "command": cmd, "message": "bye", "exit": True}
            if cmd in ("help", "?", "h"):
                return {"ok": True, "command": cmd, "message": HELP_TEXT}
            if cmd == "status":
                return self._status()
            if cmd == "nav":
                return self._nav()
            if cmd == "detect":
                return self._detect()
            if cmd == "ocr":
                return self._ocr(args)
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
            if cmd == "capture":
                return self._capture(args)
            if cmd == "sleep":
                seconds = float(args[0]) if args else 1.0
                time.sleep(seconds)
                return {"ok": True, "command": cmd, "message": f"sleep {seconds}s"}
            raise ValueError(f"未知命令: {cmd}（输入 help 查看帮助）")
        except Exception as exc:  # noqa: BLE001 - 交互模式需要容错
            logger.debug(f"命令执行失败: {exc}")
            return {"ok": False, "command": cmd, "error": str(exc)}

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

        image = self.agent.screenshot()
        path = args[0] if args else None
        if path:
            cv2.imwrite(path, image)
        payload = {
            "ok": True,
            "command": "capture",
            "path": path,
            "size": [int(image.shape[1]), int(image.shape[0])],
        }
        self._emit(payload, f"截图 {image.shape[1]}x{image.shape[0]}" + (f" -> {path}" if path else ""))
        return payload

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
