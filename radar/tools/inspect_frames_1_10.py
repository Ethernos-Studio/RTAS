"""看看 1-10 抽出来的帧里有没有"关卡标题卡 / 编队预览"这类不该标的帧。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.frames import load_manifest  # noqa: E402


def main() -> int:
    man = load_manifest(ROOT / "frames" / "manifest.jsonl")
    rows = [(frame, entry) for frame, entry in man.items() if entry.video == "v000_1-10"]
    rows.sort(key=lambda item: item[1].frame_idx)
    print(f"{'frame':<28}{'edge':>8}{'bright':>8}")
    for frame, entry in rows[:5] + rows[-2:]:
        print(f"{frame:<28}{entry.edge_ratio:8.3f}{entry.brightness:8.1f}")

    edges = sorted(entry.edge_ratio for _frame, entry in rows)
    print(f"edge 分位: 10%={edges[len(edges) // 10]:.3f} "
          f"50%={edges[len(edges) // 2]:.3f} "
          f"90%={edges[int(len(edges) * 0.9)]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
