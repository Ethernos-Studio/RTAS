"""EYESOFPRIESTESS（EYESOFPRIESTESS）数据管线与检测模型。

设计约束（详见仓库根目录 ``HANDOFF.md`` / ``eyesofpriestess_plan.md``）：

* 全程纯像素坐标，不感知"格子"，不做任何透视矫正；
* 标注最小单位是点，append-only JSONL，字段锁定；
* ``ignore`` 是一等公民：被遮挡/模糊的目标保留坐标，不删帧；
* train/val 按**视频**切分，绝不按帧随机；
* 检测只有 5 个粗类，兵种身份走 embedding 图库（本期不做）。
"""

from .schema import Click, IGNORE, POINT, REASONS, iter_clicks, load_clicks, write_clicks

__all__ = [
    "Click",
    "POINT",
    "IGNORE",
    "REASONS",
    "iter_clicks",
    "load_clicks",
    "write_clicks",
]

__version__ = "0.1.0"
