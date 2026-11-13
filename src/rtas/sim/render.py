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
# @brief 推演画面渲染：静态底图 + 动态敌人/干员 + 视频导出

"""推演画面渲染。

只用 PIL 画图（自带中文字体回退），不依赖 OpenCV 的 GUI：

- :class:`MapRenderer` 预渲染静态底图（地块 / 红门 / 蓝门 / 路线），
  之后每帧只需复制底图再画干员和敌人；
- :func:`export_video` 支持 ``.mp4``（OpenCV VideoWriter）与 ``.gif``（PIL）；
- :func:`save_snapshot` 输出单帧 PNG/JPG。

坐标约定：``row`` 从地图数据第 0 行开始向下，``col`` 从左向右。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from PIL import Image, ImageDraw, ImageFont

from .model import Frame, SimLevel, TileKind

logger = logging.getLogger(__name__)

#: 默认格子像素
DEFAULT_CELL = 64
#: 顶部状态栏高度
HUD_HEIGHT = 54

_COLORS: dict[str, tuple[int, int, int]] = {
    "background": (16, 17, 20),
    "grid": (38, 40, 46),
    "forbidden": (22, 22, 25),
    "wall": (52, 54, 60),
    "road": (57, 59, 66),
    "floor": (76, 80, 90),
    "highland": (98, 103, 115),
    "hole": (10, 10, 12),
    "start": (188, 62, 62),
    "end": (62, 112, 200),
    "start_dark": (120, 32, 32),
    "end_dark": (34, 64, 132),
    "hud": (24, 25, 30),
    "hud_text": (226, 229, 235),
    "hud_dim": (150, 155, 165),
    "operator": (66, 128, 214),
    "operator_idle": (86, 90, 100),
    "text": (240, 242, 246),
    "shadow": (0, 0, 0),
}

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover - 字体损坏时回退
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - 老版本 Pillow
        return ImageFont.load_default()


class MapRenderer:
    """把一帧推演状态画成图片。"""

    def __init__(
        self,
        level: SimLevel,
        *,
        cell: int = DEFAULT_CELL,
        show_routes: bool = True,
        show_grid: bool = True,
        show_names: bool = True,
    ) -> None:
        if not level.grid:
            raise ValueError("关卡没有地块数据，无法渲染")
        self.level = level
        self.cell = max(24, int(cell))
        self.show_routes = show_routes
        self.show_grid = show_grid
        self.show_names = show_names
        self.width = level.cols * self.cell
        self.height = level.rows * self.cell + HUD_HEIGHT
        self._static = self._build_static()

    # -- 静态底图 ---------------------------------------------------------
    def _build_static(self) -> Image.Image:
        level = self.level
        cell = self.cell
        image = Image.new("RGBA", (self.width, self.height), _COLORS["background"] + (255,))
        draw = ImageDraw.Draw(image)

        for row, line in enumerate(level.grid):
            for col, tile in enumerate(line):
                x0 = col * cell
                y0 = HUD_HEIGHT + row * cell
                self._draw_tile(draw, tile, x0, y0)

        if self.show_routes:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            route_draw = ImageDraw.Draw(overlay)
            for route in level.routes.values():
                points = [
                    (col * cell + cell / 2, HUD_HEIGHT + row * cell + cell / 2)
                    for row, col in route.points
                ]
                if len(points) >= 2:
                    route_draw.line(points, fill=(255, 255, 255, 26), width=max(2, cell // 16),
                                    joint="curve")
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)

        self._draw_gate_markers(draw)
        draw.rectangle([0, 0, self.width, HUD_HEIGHT], fill=_COLORS["hud"] + (255,))
        draw.line([(0, HUD_HEIGHT), (self.width, HUD_HEIGHT)],
                  fill=(60, 63, 70, 255), width=1)
        return image

    def _draw_tile(self, draw: ImageDraw.ImageDraw, tile: Any, x0: int, y0: int) -> None:
        cell = self.cell
        x1, y1 = x0 + cell - 1, y0 + cell - 1
        if tile.kind is TileKind.FORBIDDEN:
            color = _COLORS["forbidden"]
        elif tile.kind is TileKind.WALL:
            color = _COLORS["wall"]
        elif tile.kind is TileKind.HOLE:
            color = _COLORS["hole"]
        elif tile.kind is TileKind.ROAD:
            color = _COLORS["road"]
        elif tile.kind is TileKind.FLOOR:
            color = _COLORS["highland"] if tile.height else _COLORS["floor"]
        elif tile.kind is TileKind.START:
            color = _COLORS["start_dark"]
        else:  # END
            color = _COLORS["end_dark"]

        draw.rectangle([x0, y0, x1, y1], fill=color + (255,))

        if tile.kind is TileKind.HOLE:
            pad = cell // 4
            draw.line([(x0 + pad, y0 + pad), (x1 - pad, y1 - pad)], fill=(70, 70, 76, 255), width=2)
            draw.line([(x1 - pad, y0 + pad), (x0 + pad, y1 - pad)], fill=(70, 70, 76, 255), width=2)

        if self.show_grid:
            draw.rectangle([x0, y0, x1, y1], outline=_COLORS["grid"] + (255,), width=1)

        # 可部署格：角落一个小方块，近战/远程用不同颜色
        if tile.buildable.value != "none" and tile.kind is not TileKind.FORBIDDEN:
            size = max(4, cell // 10)
            pad = max(3, cell // 12)
            if tile.buildable.value == "all":
                mark = (240, 214, 120, 210)
            elif tile.buildable.value == "ranged":
                mark = (120, 200, 240, 200)
            else:
                mark = (170, 230, 170, 200)
            draw.rectangle([x0 + pad, y0 + pad, x0 + pad + size, y0 + pad + size],
                           fill=mark)

    def _draw_gate_markers(self, draw: ImageDraw.ImageDraw) -> None:
        cell = self.cell
        for row, line in enumerate(self.level.grid):
            for col, tile in enumerate(line):
                if tile.kind not in (TileKind.START, TileKind.END):
                    continue
                x0 = col * cell
                y0 = HUD_HEIGHT + row * cell
                color = _COLORS["start"] if tile.kind is TileKind.START else _COLORS["end"]
                draw.rounded_rectangle(
                    [x0 + 6, y0 + 6, x0 + cell - 7, y0 + cell - 7],
                    radius=max(6, cell // 5), fill=color + (255,),
                    outline=(255, 255, 255, 90), width=2,
                )
                font = _font(max(11, cell // 4))
                text = "红门" if tile.kind is TileKind.START else "蓝门"
                draw.text((x0 + cell / 2, y0 + cell / 2), text, font=font,
                          fill=(255, 255, 255, 235), anchor="mm")

    # -- 单帧 -------------------------------------------------------------
    def render(self, frame: Frame) -> Image.Image:
        """渲染一帧，返回 RGB 图片。"""
        image = self._static.copy()
        draw = ImageDraw.Draw(image)

        for operator in frame.operators:
            self._draw_operator(draw, operator)

        show_enemy_names = self.show_names and len(frame.enemies) <= 10
        for enemy in frame.enemies:
            self._draw_enemy(draw, enemy, show_name=show_enemy_names)

        self._draw_hud(draw, frame)
        return image.convert("RGB")

    def _tile_center(self, row: float, col: float) -> tuple[float, float]:
        cell = self.cell
        return (col * cell + cell / 2, HUD_HEIGHT + row * cell + cell / 2)

    def _draw_operator(self, draw: ImageDraw.ImageDraw, operator: Any) -> None:
        cell = self.cell
        x, y = self._tile_center(operator.row, operator.col)
        pad = max(6, cell // 6)
        color = _COLORS["operator"] if operator.active else _COLORS["operator_idle"]
        alpha = 235 if operator.active else 120
        box = [x - cell / 2 + pad, y - cell / 2 + pad, x + cell / 2 - pad, y + cell / 2 - pad]
        draw.rounded_rectangle(box, radius=max(6, cell // 8), fill=color + (alpha,),
                               outline=(240, 245, 255, 200 if operator.active else 90), width=2)

        if self.show_names:
            font = _font(max(10, cell // 5))
            label = operator.name if len(operator.name) <= 4 else operator.name[:4]
            draw.text((x, y - cell / 8), label, font=font, fill=(255, 255, 255, 240),
                      anchor="mm")

        # 阻挡占用：底部小格
        capacity = max(0, int(operator.capacity))
        if capacity:
            size = max(3, cell // 14)
            gap = size + 2
            total = capacity * gap - 2
            start_x = x - total / 2
            base_y = y + cell / 4
            for slot in range(capacity):
                filled = slot < operator.blocked
                draw.rectangle(
                    [start_x + slot * gap, base_y,
                     start_x + slot * gap + size, base_y + size],
                    fill=(250, 220, 120, 235) if filled else (255, 255, 255, 60),
                )

        if operator.kills:
            font = _font(max(9, cell // 6))
            draw.text((x + cell / 3, y - cell / 3), str(operator.kills), font=font,
                      fill=(255, 210, 120, 235), anchor="mm")

    def _draw_enemy(self, draw: ImageDraw.ImageDraw, enemy: Any,
                    show_name: bool = True) -> None:
        cell = self.cell
        x, y = self._tile_center(enemy.row, enemy.col)
        ratio = 0.0 if enemy.max_hp <= 0 else max(0.0, min(1.0, enemy.hp / enemy.max_hp))
        if ratio > 0.6:
            color = (96, 200, 120)
        elif ratio > 0.3:
            color = (232, 200, 84)
        else:
            color = (232, 88, 84)

        radius = max(6, cell // 5)
        if enemy.blocked_by is not None:
            draw.ellipse([x - radius - 4, y - radius - 4, x + radius + 4, y + radius + 4],
                         outline=(255, 255, 255, 220), width=2)
        if enemy.flying:
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=(0, 0, 0, 0), outline=(140, 220, 240, 230), width=2)

        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=color + (255,), outline=(20, 20, 22, 220), width=2)

        # 血条
        bar_w = radius * 2.2
        bar_h = max(3, cell // 16)
        bar_x = x - bar_w / 2
        bar_y = y - radius - bar_h - 5
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                       fill=(30, 30, 34, 220))
        draw.rectangle([bar_x, bar_y, bar_x + bar_w * ratio, bar_y + bar_h],
                       fill=color + (255,))

        if show_name and cell >= 48:
            font = _font(max(9, cell // 7))
            draw.text((x, y + radius + 2), enemy.name[:6], font=font,
                      fill=(225, 228, 235, 215), anchor="ma")

    def _draw_hud(self, draw: ImageDraw.ImageDraw, frame: Frame) -> None:
        font = _font(15)
        small = _font(12)
        blocked = sum(1 for enemy in frame.enemies if enemy.blocked_by is not None)
        active_ops = sum(1 for operator in frame.operators if operator.active)

        draw.text((12, 8), f"t = {frame.time:6.2f}s", font=font, fill=_COLORS["hud_text"])
        draw.text((140, 8), f"生命 {frame.life}/{frame.max_life}", font=font,
                  fill=_COLORS["hud_text"])
        draw.text((250, 8), f"击杀 {frame.killed}", font=font, fill=(150, 225, 170))
        draw.text((340, 8), f"漏怪 {frame.leaked}", font=font, fill=(235, 150, 150))
        draw.text((430, 8), f"场上 {len(frame.enemies)}（阻挡 {blocked}）", font=font,
                  fill=_COLORS["hud_text"])
        draw.text((12, 31), f"干员 {active_ops}/{len(frame.operators)}    "
                             f"生成 {frame.spawned}", font=small, fill=_COLORS["hud_dim"])
        draw.text((self.width - 12, 31), self.level.level_id, font=small,
                  fill=_COLORS["hud_dim"], anchor="ra")


# =============================================================================
# 导出
# =============================================================================

def save_snapshot(
    level: SimLevel,
    frame: Frame,
    path: str | Path,
    *,
    renderer: Optional[MapRenderer] = None,
    cell: int = DEFAULT_CELL,
) -> Path:
    """把一帧保存为图片（PNG/JPG）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    renderer = renderer or MapRenderer(level, cell=cell)
    image = renderer.render(frame)
    if target.suffix.lower() in (".jpg", ".jpeg"):
        image = image.convert("RGB")
    image.save(target)
    return target


def export_video(
    level: SimLevel,
    frames: Iterable[Frame],
    path: str | Path,
    *,
    fps: float = 20.0,
    renderer: Optional[MapRenderer] = None,
    cell: int = DEFAULT_CELL,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Path:
    """把逐帧状态导出为视频。

    ``.gif`` 用 PIL 保存；其它后缀（``.mp4`` 等）用 OpenCV 的 ``mp4v`` 编码。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    renderer = renderer or MapRenderer(level, cell=cell)
    frame_list = list(frames)
    total = len(frame_list)
    if total == 0:
        raise ValueError("没有可导出的帧")

    if target.suffix.lower() == ".gif":
        images = [renderer.render(frame) for frame in frame_list]
        duration = max(20, int(round(1000.0 / max(fps, 1.0))))
        images[0].save(target, save_all=True, append_images=images[1:],
                       duration=duration, loop=0)
        return target

    import cv2
    import numpy as np

    probe = renderer.render(frame_list[0])
    size = probe.size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, max(fps, 1.0), size)
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建视频文件：{target}（当前环境缺少可用的视频编码器，"
            "可改用 .gif 输出）"
        )
    try:
        for index, frame in enumerate(frame_list):
            image = probe if index == 0 else renderer.render(frame)
            writer.write(np.asarray(image.convert("RGB"))[:, :, ::-1])
            if progress is not None:
                progress(index + 1, total)
    finally:
        writer.release()
    return target


__all__ = ["DEFAULT_CELL", "MapRenderer", "export_video", "save_snapshot"]
