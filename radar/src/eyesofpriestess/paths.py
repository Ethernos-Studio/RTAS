"""目录约定（HANDOFF §4）与项目根定位。

::

    raw/        原始录像
    frames/     extract 输出（帧 + manifest.jsonl）
    clicks/     标注（LS 经转换器写出，一帧一个 jsonl）
    dataset/    build 输出（YOLO 格式 + crops + report.json）
    runs/       训练产物
    tasks/      Label Studio 导入用的 tasks.json
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """往上找带 ``pyproject.toml`` 的目录；找不到就用当前目录。"""
    start = Path(start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


@dataclass(frozen=True)
class Layout:
    """项目内所有落盘位置。用绝对路径，避免 cwd 变来变去。"""

    root: Path

    @classmethod
    def discover(cls, root: Path | str | None = None) -> "Layout":
        return cls(root=Path(root).resolve() if root else find_root())

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def manifest(self) -> Path:
        return self.frames / "manifest.jsonl"

    @property
    def clicks(self) -> Path:
        return self.root / "clicks"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks"

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    def ensure(self) -> "Layout":
        for directory in (self.raw, self.frames, self.clicks, self.tasks, self.weights):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def __str__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Layout({self.root})"
