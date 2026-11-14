"""友军（干员）抑制：模型从没见过"干员不是敌人"的负样本，所以会对着干员报 enemy。

两条路，按可靠性排序：

1. **RTAS 直接给干员框**（权威）：干员是玩家放的，RTAS 知道每个部署位的屏幕位置，
   把框传进来即可，这里只做去重；
2. **靠血条颜色兜底**（独立运行时的近似）：干员脚下那条基础血条是**蓝色**，
   敌人是红色。在检测框水平范围内找蓝色条，命中就压掉。

接口保持简单：传进 ``ally_boxes``（一帧的干员框），返回过滤后的 units。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cv.bars import Bar, find_bars

#: 干员血条的基础颜色（技能期间会变红，所以只能当"兜底"证据）
ALLY_BAR_KINDS = ("blue", "green")


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


@dataclass
class AllyFilter:
    """按干员框 / 友方血条压掉 enemy 误报。"""

    iou_threshold: float = 0.35
    bar_search_pad: int = 26        # 在框下方多少像素内找血条
    min_bar_width: int = 18
    use_bars: bool = True
    dropped: int = 0

    def ally_boxes_from_bars(self, image: np.ndarray, *, bars: list[Bar] | None = None) -> list[list[float]]:
        """把画面里的友方血条扩成"干员大致位置"的框。"""
        bars = bars if bars is not None else find_bars(image, min_width=self.min_bar_width)
        boxes = []
        for bar in bars:
            if bar.kind not in ALLY_BAR_KINDS:
                continue
            # 血条一般在地面单位脚下方：向上扩 4~5 倍条高当作身体
            body = bar.height * 4 + 12
            boxes.append([bar.x1 - 8, bar.y1 - body, bar.x2 + 8, bar.y1 + 4])
        return boxes

    def filter(
        self,
        image: np.ndarray,
        units: list[tuple[list[float], int]],
        ally_boxes: list[list[float]] | None = None,
        *,
        bars: list[Bar] | None = None,
    ) -> list[tuple[list[float], int]]:
        """过滤 units；``ally_boxes`` 为空时自动用血条推断。"""
        if ally_boxes is None:
            ally_boxes = self.ally_boxes_from_bars(image, bars=bars) if self.use_bars else []
        if not ally_boxes:
            return units

        kept: list[tuple[list[float], int]] = []
        for box, track_id in units:
            center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            drop = False
            for ally in ally_boxes:
                if iou(box, ally) >= self.iou_threshold:
                    drop = True
                    break
                if ally[0] <= center[0] <= ally[2] and ally[1] <= center[1] <= ally[3] + 8:
                    drop = True
                    break
            if drop:
                self.dropped += 1
                continue
            kept.append((box, track_id))
        return kept
