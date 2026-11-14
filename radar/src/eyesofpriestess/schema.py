"""点选标注的唯一数据契约。

一行一个点，append-only JSONL，字段锁定（见 HANDOFF §2.3）::

    {"frame":"v000_1-7_f000123.jpg","x":512,"y":384,"cls":"源石虫",
     "type":"point","reason":null,"op":"dhjs0000","uid":"...","ts":"..."}

    {"frame":"v000_1-7_f000123.jpg","x":640,"y":150,"cls":null,
     "type":"ignore","reason":"occluded","op":"...","uid":"...","ts":"..."}

所有环节（LS 转换器、SAM 出框、YOLO 数据集、身份图库）都只读这一种格式，
所以这里的字段名与取值是**接口**，改动前先改 HANDOFF。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

#: 正常目标
POINT = "point"
#: 无法判定的目标（遮挡/模糊），保留坐标但不进训练集
IGNORE = "ignore"
#: ignore 的两种原因，必须用这两个值
REASONS = ("occluded", "blurry")

FIELDS = ("frame", "x", "y", "cls", "type", "reason", "op", "uid", "ts")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_uid() -> str:
    return uuid.uuid4().hex[:16]


@dataclass
class Click:
    """一个点选标注。"""

    frame: str
    x: float
    y: float
    cls: Optional[str] = None
    type: str = POINT
    reason: Optional[str] = None
    op: str = ""
    uid: str = field(default_factory=_new_uid)
    ts: str = field(default_factory=_now)

    def validate(self) -> "Click":
        """校验并返回自身；不合法直接抛 ``ValueError``（宁可炸，不要脏数据）。"""
        if not self.frame:
            raise ValueError("click.frame 不能为空")
        if not isinstance(self.x, (int, float)) or not isinstance(self.y, (int, float)):
            raise ValueError(f"坐标必须是数字: x={self.x!r} y={self.y!r}")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"坐标不能为负: ({self.x}, {self.y})")
        if self.type == POINT:
            if not self.cls:
                raise ValueError(f"point 必须带 cls: {self.frame} ({self.x},{self.y})")
            self.reason = None
        elif self.type == IGNORE:
            if self.reason not in REASONS:
                raise ValueError(f"ignore 的 reason 必须是 {REASONS}: {self.reason!r}")
            self.cls = None
        else:
            raise ValueError(f"type 只能是 point/ignore: {self.type!r}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "x": int(round(self.x)),
            "y": int(round(self.y)),
            "cls": self.cls,
            "type": self.type,
            "reason": self.reason,
            "op": self.op,
            "uid": self.uid,
            "ts": self.ts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Click":
        record = {
            "frame": data["frame"],
            "x": float(data["x"]),
            "y": float(data["y"]),
            "cls": data.get("cls"),
            "type": data.get("type", POINT),
            "reason": data.get("reason"),
            "op": str(data.get("op") or ""),
            "uid": str(data.get("uid") or _new_uid()),
            "ts": str(data.get("ts") or _now()),
        }
        return cls(**record).validate()

    @property
    def is_ignore(self) -> bool:
        return self.type == IGNORE


def write_clicks(path: Path, clicks: Iterable[Click], *, append: bool = True) -> int:
    """写 JSONL（默认追加，符合 append-only 约定），返回写入条数。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for click in clicks:
            handle.write(json.dumps(click.validate().to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_clicks(path: Path) -> Iterator[Click]:
    """逐行读取 JSONL；坏行跳过并打印，避免一条脏数据拖死整批。"""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Click.from_dict(json.loads(line))
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                print(f"[warn] {path.name}:{lineno} 解析失败，已跳过: {exc}")


def load_clicks(source: Path) -> dict[str, list[Click]]:
    """按帧聚合标注，帧名 -> 点击列表（帧内按 ts 排序）。

    ``source`` 可以是单个 ``.jsonl``，也可以是目录（读目录下所有 ``*.jsonl``）。
    """
    source = Path(source)
    files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    frames: dict[str, list[Click]] = {}
    for file in files:
        for click in iter_clicks(file):
            frames.setdefault(click.frame, []).append(click)
    for clicks in frames.values():
        clicks.sort(key=lambda click: click.ts)
    return frames
