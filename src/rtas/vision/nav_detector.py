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
# @brief 主界面/子菜单判定（导航按钮模板匹配）

"""主界面与子菜单判定。

明日方舟在子菜单（终端、干员、采购中心、基建……）里，左上角会显示一个
「网状球体 + 房子」的按钮（返回箭头 ``<`` 之后、``i`` 信息按钮之前）。
它**不是**「回到主界面」按钮，而是「打开顶部导航栏」按钮：点击后弹出
包含 首页 / 编队 / 干员 / 档案 / 终端 / 基建 / 公开招募 / 干员寻访 / 采购中心
的导航条，再点其中的「首页」才回到主界面。

MAA 的通用导航思路正是利用这一点：任意子菜单都能通过「点导航按钮 →
点首页」两步回到主界面，不必为每个界面单独写识别。

因此本模块的判定语义是：

- 主界面：没有这个按钮（顶部导航栏本身可见）→ :attr:`NavigationState.MAIN_MENU`
- 子菜单：存在这个按钮 → :attr:`NavigationState.SUB_MENU`，
  返回的中心坐标就是「打开导航栏」的点击位置

模板取自游戏 UI 资源 ``arts/ui/.../btn_topmenu_home`` 里的房子图标
（``templates/ui/home_btn_house.png``），只保留房子本体的高亮区域，
配合多尺度灰度模板匹配，能适应 720p~4K 以及不同主题/背景的截图
（实测 1264x759 采购中心截图相似度 0.96，主界面最高 0.61）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).with_name("templates") / "ui"
DEFAULT_TEMPLATE = TEMPLATE_DIR / "home_btn_house.png"


class NavigationState(Enum):
    """截图所处的导航层级。"""

    MAIN_MENU = "main_menu"   # 主界面：顶部导航栏可见，没有左上角导航按钮
    SUB_MENU = "sub_menu"     # 任意子菜单：左上角有「打开导航栏」按钮
    UNKNOWN = "unknown"       # 无法判断（例如对局中、加载中）


@dataclass(frozen=True)
class NavMenuButtonMatch:
    """左上角「打开导航栏」按钮的匹配结果。"""

    found: bool
    confidence: float = 0.0
    center: tuple[int, int] | None = None
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    scale: float = 0.0

    @property
    def state(self) -> NavigationState:
        if self.found:
            return NavigationState.SUB_MENU
        return NavigationState.UNKNOWN if self.confidence <= 0 else NavigationState.MAIN_MENU

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "state": self.state.value,
            "confidence": round(self.confidence, 4),
            "center": list(self.center) if self.center else None,
            "bbox": list(self.bbox) if self.bbox else None,
            "scale": round(self.scale, 3),
        }


class NavMenuButtonDetector:
    """在截图左上角寻找「打开顶部导航栏」按钮。"""

    def __init__(
        self,
        template_path: str | Path | None = None,
        threshold: float = 0.85,
        roi: tuple[float, float, float, float] = (0.0, 0.0, 0.40, 0.25),
        min_scale: float = 0.45,
        max_scale: float = 1.8,
    ) -> None:
        self.template_path = Path(template_path) if template_path else DEFAULT_TEMPLATE
        self.threshold = float(threshold)
        self.roi = roi
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self._template: np.ndarray | None = None

    # -- 模板 -------------------------------------------------------------
    @property
    def template(self) -> np.ndarray:
        if self._template is None:
            img = cv2.imread(str(self.template_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"导航按钮模板不存在: {self.template_path}")
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            self._template = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._template

    # -- 匹配 -------------------------------------------------------------
    def _match_at(self, region: np.ndarray, scales: np.ndarray):
        template = self.template
        best_score, best_scale, best_loc, best_size = -2.0, 0.0, (0, 0), (0, 0)
        rh, rw = region.shape
        for scale in scales:
            tw = int(round(template.shape[1] * float(scale)))
            th = int(round(template.shape[0] * float(scale)))
            if tw < 10 or th < 8 or tw >= rw or th >= rh:
                continue
            resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(region, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = float(max_val)
                best_scale = float(scale)
                best_loc = max_loc
                best_size = (tw, th)
        return best_score, best_scale, best_loc, best_size

    def detect(self, image: np.ndarray) -> NavMenuButtonMatch:
        """检测「打开顶部导航栏」按钮。

        Args:
            image: BGR 截图

        Returns:
            :class:`NavMenuButtonMatch`；``found=False`` 表示当前大概率在主界面
        """
        if image is None or image.size == 0:
            return NavMenuButtonMatch(False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        x1 = int(w * self.roi[0])
        y1 = int(h * self.roi[1])
        x2 = max(x1 + 1, int(w * self.roi[2]))
        y2 = max(y1 + 1, int(h * self.roi[3]))
        region = gray[y1:y2, x1:x2]

        # 粗搜 + 精搜，兼顾分辨率跨度与耗时
        coarse = np.arange(self.min_scale, self.max_scale + 1e-6, 0.2)
        score, scale, loc, size = self._match_at(region, coarse)
        if scale > 0:
            fine = np.arange(
                max(self.min_scale, scale - 0.15),
                min(self.max_scale, scale + 0.15) + 1e-6,
                0.05,
            )
            f_score, f_scale, f_loc, f_size = self._match_at(region, fine)
            if f_score > score:
                score, scale, loc, size = f_score, f_scale, f_loc, f_size

        if scale <= 0:
            return NavMenuButtonMatch(False)

        bx, by = loc[0] + x1, loc[1] + y1
        bw, bh = size
        found = score >= self.threshold
        match = NavMenuButtonMatch(
            found=found,
            confidence=score,
            center=(bx + bw // 2, by + bh // 2),
            bbox=(bx, by, bw, bh),
            scale=scale,
        )
        if found:
            logger.info(
                f"检测到导航按钮: center={match.center} conf={score:.3f} scale={scale:.2f}"
            )
        else:
            logger.debug(f"未检测到导航按钮: best={score:.3f} scale={scale:.2f}")
        return match

    def detect_state(self, image: np.ndarray) -> tuple[NavigationState, NavMenuButtonMatch]:
        """同时返回导航状态与匹配细节。"""
        match = self.detect(image)
        if match.found:
            return NavigationState.SUB_MENU, match
        if match.confidence > 0:
            return NavigationState.MAIN_MENU, match
        return NavigationState.UNKNOWN, match

    def visualize(
        self,
        image: np.ndarray,
        match: NavMenuButtonMatch,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        """在原图上标注检测结果。"""
        vis = image.copy()
        if match.bbox:
            x, y, w, h = match.bbox
            color = (0, 200, 0) if match.found else (0, 165, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label = f"nav {match.confidence:.2f}"
            cv2.putText(
                vis, label, (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )
        state_text = match.state.value
        cv2.putText(
            vis, state_text, (10, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )
        if output_path:
            cv2.imwrite(str(output_path), vis)
        return vis


def detect_navigation(image: np.ndarray, **kwargs) -> tuple[NavigationState, NavMenuButtonMatch]:
    """便捷函数：判断截图位于主界面还是子菜单。"""
    return NavMenuButtonDetector(**kwargs).detect_state(image)


# 向后兼容：早期版本把这个按钮误称为「主页按钮」
HomeButtonDetector = NavMenuButtonDetector
HomeButtonMatch = NavMenuButtonMatch
