"""极简 IoU 跟踪器：给批量检测补上 track_id。

为什么不直接用 ultralytics 的 ``model.track()``：它是**逐帧**调用，没法批量，
实测在 960 输入下只有 ~20 FPS；而批量 ``predict`` 能到 31 FPS。这里把检测批量化，
跟踪用几行 IoU 贪心匹配补上——对"同一帧里敌人不重叠"的塔防画面足够用。

生产版应该换成 ByteTrack（它有 Kalman 预测和低分框二次关联），但那是"跟踪质量"的事，
不影响这条链路的接口：``SimpleTracker.update(boxes) -> [(box, track_id), ...]``。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


@dataclass
class SimpleTracker:
    """贪心 IoU 跟踪：上一帧的框按最大重叠配对，配不上的开新 ID。"""

    iou_threshold: float = 0.3
    max_missing: int = 30           # 连续多少帧没配上就丢掉这个 ID
    next_id: int = 1
    tracks: dict[int, dict] = field(default_factory=dict)

    def update(self, boxes: list[list[float]]) -> list[tuple[list[float], int]]:
        """喂当前帧的框（xyxy），返回 [(box, track_id), ...]。"""
        assigned: dict[int, int] = {}          # box_index -> track_id
        candidates: list[tuple[float, int, int]] = []
        for box_index, box in enumerate(boxes):
            for track_id, track in self.tracks.items():
                value = iou(box, track["box"])
                if value >= self.iou_threshold:
                    candidates.append((value, box_index, track_id))

        used_tracks: set[int] = set()
        for value, box_index, track_id in sorted(candidates, reverse=True):
            if box_index in assigned or track_id in used_tracks:
                continue
            assigned[box_index] = track_id
            used_tracks.add(track_id)

        result: list[tuple[list[float], int]] = []
        for box_index, box in enumerate(boxes):
            track_id = assigned.get(box_index)
            if track_id is None:
                track_id = self.next_id
                self.next_id += 1
            self.tracks[track_id] = {"box": box, "missing": 0}
            result.append((box, track_id))

        for track_id in list(self.tracks):
            if track_id in used_tracks or track_id in assigned.values():
                continue
            self.tracks[track_id]["missing"] += 1
            if self.tracks[track_id]["missing"] > self.max_missing:
                self.tracks.pop(track_id, None)
        return result
