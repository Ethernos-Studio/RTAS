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
# @brief 游戏操作代理（截图 + 识别 + 后台点击）

"""游戏操作代理。

把「截图 → 识别 → 点击」串成高层动作，输入走 ``rtas.platform`` 的
后台点击（不抢鼠标），因此可以在后台完成「打开顶部导航栏 → 点首页」这类操作。

典型用法::

    from rtas.agent import GameAgent

    with GameAgent() as agent:
        agent.goto("首页")        # 回到主界面
        agent.goto("编队")        # 进入编队界面
        agent.click(960, 540)     # 点击任意坐标
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from .platform import ClickMethod, UnsupportedPlatform, WindowRef, get_backend
from .vision.nav_detector import NavigationState, NavMenuButtonDetector
from .vision.ocr import OCRBackend, OCRText, create_backend

logger = logging.getLogger(__name__)

#: 主界面/顶部导航栏里的菜单名
MENU_TARGETS = (
    "首页", "终端", "干员", "编队", "公开招募", "干员寻访",
    "采购中心", "档案", "基建", "任务", "仓库",
)

_HOME_ALIASES = {"首页", "home", "主界面", "主页"}


class GameAgent:
    """《明日方舟》操作代理。"""

    def __init__(
        self,
        keywords: Sequence[str] | None = None,
        hwnd: int | None = None,
        *,
        method: ClickMethod | str = ClickMethod.WINDOW,
        dry_run: bool = False,
        ocr_engine: str | None = None,
        gpu: bool = False,
        action_delay: float = 1.0,
        nav_threshold: float = 0.85,
    ) -> None:
        self.keywords = keywords
        self.hwnd = hwnd
        self.method = ClickMethod(method)
        self.dry_run = dry_run
        self.action_delay = action_delay
        self._backend = get_backend()
        self._window: WindowRef | None = None
        self._frame_size: tuple[int, int] | None = None
        self._ocr: OCRBackend | None = None
        self._ocr_engine = ocr_engine
        self._gpu = gpu
        self._nav = NavMenuButtonDetector(threshold=nav_threshold)

    # -- 窗口与截图 --------------------------------------------------------
    @property
    def window(self) -> WindowRef:
        if self._window is None:
            self._window = self._backend.find_window(self.keywords, self.hwnd)
            logger.info(
                f"目标窗口: {self._window.title} ({self._window.width}x{self._window.height})"
            )
        return self._window

    def refresh_window(self) -> WindowRef:
        self._window = None
        return self.window

    def screenshot(self) -> np.ndarray:
        """截取窗口客户区（BGR）。

        WGC 抓到的画面可能比真实客户区略小（实测 1264×711 vs 1280×720），
        所以这里记录画面尺寸，OCR 得到的坐标会按比例换算回客户区坐标。
        """
        image = self._backend.capture(self.window)
        height, width = image.shape[:2]
        self._frame_size = (int(width), int(height))
        return image

    def to_client(self, x: float, y: float) -> tuple[int, int]:
        """把截图坐标换算成客户区坐标。"""
        if self._frame_size is None:
            return int(round(x)), int(round(y))
        frame_w, frame_h = self._frame_size
        client_w, client_h = self.window.width, self.window.height
        if frame_w <= 0 or frame_h <= 0 or (frame_w, frame_h) == (client_w, client_h):
            return int(round(x)), int(round(y))
        return (
            int(round(x * client_w / frame_w)),
            int(round(y * client_h / frame_h)),
        )

    def is_admin(self) -> bool:
        return self._backend.is_admin()

    # -- OCR ---------------------------------------------------------------
    @property
    def ocr(self) -> OCRBackend:
        if self._ocr is None:
            self._ocr = create_backend(
                self._ocr_engine, languages=["ch_sim", "en"], gpu=self._gpu
            )
            self._ocr.initialize()
        return self._ocr

    def read_text(
        self,
        image: np.ndarray | None = None,
        roi: tuple[float, float, float, float] | None = None,
    ) -> list[OCRText]:
        """识别文字；``roi`` 为相对坐标 (x1, y1, x2, y2)。"""
        image = self.screenshot() if image is None else image
        offset = (0, 0)
        if roi:
            h, w = image.shape[:2]
            x1, y1 = int(w * roi[0]), int(h * roi[1])
            x2, y2 = int(w * roi[2]), int(h * roi[3])
            image = image[y1:y2, x1:x2]
            offset = (x1, y1)

        items = self.ocr.recognize(image)
        if offset == (0, 0):
            return items
        shifted: list[OCRText] = []
        for item in items:
            bbox = item.bbox
            if bbox:
                bbox = [[p[0] + offset[0], p[1] + offset[1]] for p in bbox]
            shifted.append(OCRText(item.text, item.confidence, bbox))
        return shifted

    def find_text(
        self,
        target: str,
        image: np.ndarray | None = None,
        *,
        roi: tuple[float, float, float, float] | None = None,
        min_similarity: float = 0.5,
    ) -> OCRText | None:
        """在截图中查找一段文字，返回匹配度最高的一条。"""
        best: tuple[float, OCRText] | None = None
        for item in self.read_text(image, roi):
            text = item.text.strip()
            if not text:
                continue
            if target in text or text in target:
                score = 1.0
            else:
                score = SequenceMatcher(None, target, text).ratio()
            if score >= min_similarity and item.center and (best is None or score > best[0]):
                best = (score, item)
        return best[1] if best else None

    # -- 导航与点击 --------------------------------------------------------
    def nav(self, image: np.ndarray | None = None):
        """判断当前在主界面还是子菜单。"""
        image = self.screenshot() if image is None else image
        return self._nav.detect_state(image)

    def click(self, x: int, y: int, *, dry_run: bool | None = None) -> bool:
        """在客户区坐标点击（默认后台点击，不抢鼠标）。

        ``x``/``y`` 使用客户区坐标系（与截图同源，内部会自动换算）。
        """
        if dry_run if dry_run is not None else self.dry_run:
            logger.info(f"[dry-run] click ({x}, {y}) by {self.method.value}")
            return True
        ok = self._backend.click(self.window, int(x), int(y), self.method)
        logger.info(f"点击 ({x}, {y}) method={self.method.value} ok={ok}")
        return ok

    def click_frame(self, x: float, y: float, *, dry_run: bool | None = None) -> bool:
        """按**截图坐标**点击（自动换算成客户区坐标）。"""
        return self.click(*self.to_client(x, y), dry_run=dry_run)

    def click_text(self, target: str, *, roi=None, dry_run: bool | None = None) -> bool:
        hit = self.find_text(target, roi=roi)
        if hit is None or hit.center is None:
            return False
        return self.click_frame(hit.center[0], hit.center[1], dry_run=dry_run)

    def key(self, name: str) -> bool:
        if self.dry_run:
            logger.info(f"[dry-run] key {name}")
            return True
        return self._backend.key(self.window, name, self.method)

    # -- 高层动作 ----------------------------------------------------------
    def goto(self, target: str, *, attempts: int = 2) -> dict[str, Any]:
        """进入指定界面：子菜单先打开顶部导航栏，再点目标项。

        Args:
            target: 菜单名（见 :data:`MENU_TARGETS`）或任意界面文字
            attempts: 找不到目标时重试次数（等待页面切换）

        Returns:
            ``{"ok": bool, "target": str, "steps": [...]}``
        """
        target = target.strip()
        steps: list[dict[str, Any]] = []
        image = self.screenshot()
        state, match = self.nav(image)

        if target in _HOME_ALIASES and state is NavigationState.MAIN_MENU:
            return {"ok": True, "target": target, "steps": [{"action": "already_main_menu"}]}

        if state is NavigationState.SUB_MENU and match.found and match.center:
            self.click_frame(*match.center)
            steps.append({"action": "open_nav_bar", "center": list(match.center)})
            if self.dry_run:
                # dry-run 不会真正展开导航栏，无法继续验证目标位置
                steps.append({"action": "click_target", "text": target, "center": None})
                return {"ok": True, "target": target, "steps": steps, "dry_run": True}
            time.sleep(self.action_delay)
            image = self.screenshot()

        hit = None
        for attempt in range(max(1, attempts)):
            # 顶部导航栏展开后集中在画面上半部分
            hit = self.find_text(target, image, roi=(0.0, 0.0, 1.0, 0.45))
            if hit is None:
                hit = self.find_text(target, image)
            if hit is not None or attempt == attempts - 1:
                break
            time.sleep(self.action_delay)
            image = self.screenshot()

        if hit is None or hit.center is None:
            return {"ok": False, "target": target, "steps": steps,
                    "error": f"未在画面中找到「{target}」"}

        self.click_frame(*hit.center)
        steps.append({
            "action": "click_target",
            "text": hit.text,
            "center": list(hit.center),
        })
        return {"ok": True, "target": target, "steps": steps}

    def close(self) -> None:
        if self._ocr is not None:
            self._ocr.shutdown()
            self._ocr = None

    def __enter__(self) -> GameAgent:
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False


__all__ = ["GameAgent", "MENU_TARGETS", "UnsupportedPlatform"]
