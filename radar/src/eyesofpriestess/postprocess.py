"""检测后处理：压掉"把两个敌人框成一个"的合并框，以及重复框。

塔防里敌人经常贴在一起走，YOLO 会给出三种结果：

1. 两个框各自正确（理想）；
2. 一个大框把两只一起框住（回归合并）；
3. 一个大框 + 两个各自的小框（合并框没被 NMS 压掉，因为和它的 IoU 不够高）。

NMS 管不了 2/3（调低 IoU 会把正常相邻的两个也压掉）。这里做一件 NMS 不做的事：
**如果一个框里面完整套着 ≥2 个别的框，那它就是合并框，丢掉**。
"""

from __future__ import annotations

from dataclasses import dataclass


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dedupe(boxes_with_conf: list[tuple[list[float], float]], *, iou_threshold: float = 0.5
           ) -> list[tuple[list[float], float]]:
    """重复框抑制：IoU 很高时只留置信度最高的那个。

    NMS 调成 0.8 是为了不合并相邻的敌人，代价是同一个目标可能出现两个框
    （一大一小，IoU 0.5~0.7 都能活下来）。对 VLM 来说"同一只虫子算两个"比漏检更糟，
    所以这里补一道去重。
    """
    if len(boxes_with_conf) < 2:
        return boxes_with_conf
    ordered = sorted(boxes_with_conf, key=lambda item: -item[1])
    kept: list[tuple[list[float], float]] = []
    for box, conf in ordered:
        if any(_iou(box, other) >= iou_threshold for other, _ in kept):
            continue
        kept.append((box, conf))
    return kept


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def containment(inner: list[float], outer: list[float]) -> float:
    """inner 有多大比例落在 outer 里面（0~1）。"""
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / max(_area(inner), 1e-6)


@dataclass
class MergeSuppressor:
    """丢掉"包住 ≥2 个框"的合并框。

    额外一道保险（``min_child_coverage``）：被包住的子框加起来要能覆盖父框的大部分面积，
    否则说明父框里还有子框没盖到的敌人——这时丢掉父框就变成漏检了。
    """

    contain: float = 0.85          # 子框有这么多落在父框里才算"被包住"
    min_children: int = 2          # 至少包住几个子框才判定为合并框
    min_area_ratio: float = 1.6    # 父框面积至少是子框的多少倍，避免误伤正常的大目标
    min_child_coverage: float = 0.65   # 子框合计至少覆盖父框这么多面积才敢丢
    dropped: int = 0

    def __call__(self, boxes: list[list[float]]) -> list[list[float]]:
        if len(boxes) < 2:
            return boxes
        keep: list[list[float]] = []
        for index, box in enumerate(boxes):
            children: list[list[float]] = []
            for other_index, other in enumerate(boxes):
                if other_index == index:
                    continue
                if _area(box) < _area(other) * self.min_area_ratio:
                    continue
                if containment(other, box) >= self.contain:
                    children.append(other)
            if len(children) >= self.min_children:
                # 子框必须几乎撑满父框，才敢认定"父框只是把这两个框合起来"
                ux1 = min(child[0] for child in children)
                uy1 = min(child[1] for child in children)
                ux2 = max(child[2] for child in children)
                uy2 = max(child[3] for child in children)
                if (ux2 - ux1) >= self.min_child_coverage * (box[2] - box[0]) and \
                        (uy2 - uy1) >= self.min_child_coverage * (box[3] - box[1]):
                    self.dropped += 1
                    continue
            keep.append(box)
        return keep
