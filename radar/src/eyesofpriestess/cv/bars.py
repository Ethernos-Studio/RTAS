"""血条 / 技能条：用颜色 + 几何的 CV 找，不走模型。

为什么不用 YOLO 学：这些条是**半透明黑底 + 纯色填充**的规则图形，
颜色阈值 + 连通域就足够稳；模型学它反而会被"同类不同色"骗
（干员开技能血条会变红、技能条绿→橙），标注成本还翻倍。

产出两种条：

* ``red`` / ``blue``：血条填充色（红=敌、蓝=友）；
* ``green``：技能条（充能/就绪）。

注意：颜色只是**线索**，不是归属。干员开技能时血条会变红，
所以下游要用"这条血条挂在谁身上"来判敌我，不能只看颜色。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

#: 血条/技能条填充色的 HSV 范围（OpenCV 的 H 是 0~179）
#:
#: 这些数值是**实测**出来的（tools/inspect_bars.py 打真实战斗帧）：
#: 条是半透明黑底 + 半透明彩色填充，颜色并不鲜艳——
#: 敌人红条中位色相 10~26、饱和度 66~88、亮度 60~95，条高 4~6px、宽 37~86px。
#: 按"纯红/纯蓝"的阈值（sat>120）会一个都抓不到。
RED_RANGES = ((0, 25), (170, 179))
BLUE_RANGE = (90, 140)
GREEN_RANGE = (35, 89)


@dataclass
class Bar:
    """一根条（像素坐标外接框）。"""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: str            # red / blue / green
    area: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_xywh(self) -> list[int]:
        return [self.x1, self.y1, self.width, self.height]


def _color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    red = np.zeros(hue.shape, dtype=bool)
    for low, high in RED_RANGES:
        red |= (hue >= low) & (hue <= high)

    return {
        "red": (red & (sat > 55) & (val > 50)),
        "blue": ((hue >= BLUE_RANGE[0]) & (hue <= BLUE_RANGE[1]) & (sat > 45) & (val > 45)),
        "green": ((hue >= GREEN_RANGE[0]) & (hue <= GREEN_RANGE[1]) & (sat > 45) & (val > 45)),
    }


def _has_dark_plate(value: np.ndarray, x: int, y: int, w: int, h: int, max_dark: float) -> bool:
    """条必须压在"半透明黑底"上：四边至少两边是暗的。

    地板上的彩色花纹同样细长、同样有颜色，但周围是亮的；真正的条四边紧贴着暗色 trough。
    """
    height, width = value.shape
    sides = []
    if y > 2:
        sides.append(float(value[y - 2:y - 1, x:x + w].mean()))
    if y + h + 2 < height:
        sides.append(float(value[y + h + 1:y + h + 2, x:x + w].mean()))
    if x > 2:
        sides.append(float(value[y:y + h, x - 2:x - 1].mean()))
    if x + w + 2 < width:
        sides.append(float(value[y:y + h, x + w + 1:x + w + 2].mean()))
    dark = sum(1 for side in sides if side <= max_dark)
    return dark >= 2


def find_bars(
    image: np.ndarray,
    *,
    min_width: int = 14,
    min_height: int = 2,
    max_height: int = 12,
    min_aspect: float = 2.5,
    min_area: int = 20,
    max_dark: float = 80.0,
) -> list[Bar]:
    """找出画面里所有像"条"的东西，按颜色分类。

    Args:
        image: BGR 原图
        min_width/min_height/max_height: 尺寸过滤（太细碎、太方的都不是条）
        min_aspect: 宽高比下限（条一定是横的）
        min_area: 填充像素数下限（滤掉噪点）
        max_dark: 条周围黑底的平均亮度上限（黑底是半透明的，不是纯黑）
    """
    if image.size == 0:
        return []
    height, width = image.shape[:2]
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
    kernel = np.ones((1, 5), dtype=np.uint8)
    bars: list[Bar] = []

    for kind, mask in _color_masks(image).items():
        binary = mask.astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            if w < min_width or w < 8 or h < min_height or h > max_height:
                continue
            if w / max(h, 1) < min_aspect or area < min_area:
                continue
            if x + w > width or y + h > height:
                continue
            # 条下面通常还有 1~2px 黑底；用框下方一条薄带校验"是不是贴在暗底上"
            if not _has_dark_plate(value, x, y, w, h, max_dark):
                continue
            bars.append(Bar(x1=x, y1=y, x2=x + w, y2=y + h, kind=kind, area=area))

    bars.sort(key=lambda bar: (bar.y1, bar.x1))
    return bars


def summarize(bars: list[Bar]) -> dict[str, int]:
    """按颜色计数，方便日志/报告。"""
    counts: dict[str, int] = {}
    for bar in bars:
        counts[bar.kind] = counts.get(bar.kind, 0) + 1
    return counts


def draw(image: np.ndarray, bars: list[Bar], *, thickness: int = 1) -> np.ndarray:
    """把找到的条画出来（调试用）。"""
    output = image.copy()
    colors = {"red": (0, 0, 255), "blue": (255, 128, 0), "green": (0, 200, 0)}
    for bar in bars:
        cv2.rectangle(output, (bar.x1, bar.y1), (bar.x2, bar.y2),
                      colors.get(bar.kind, (255, 255, 255)), thickness)
    return output
