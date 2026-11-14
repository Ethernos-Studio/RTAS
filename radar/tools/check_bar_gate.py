"""干员挡住的敌人也要能过门控：验证"同时有蓝条+橙条时保留"。

用带标注帧里"和干员重叠的敌人"是最难的情况；这里先量普通敌人误杀率，
再看有干员的帧上会不会把敌人丢掉。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.hpbar_gate import find_hp_bars, keep_as_enemy  # noqa: E402
from eyesofpriestess.schema import load_clicks  # noqa: E402


def main() -> int:
    clicks = load_clicks(ROOT / "clicks")
    stats: dict[str, int] = {}
    total = 0
    for frame in [f for f in clicks if f.startswith(("v001_1-8", "v002_1-9"))][:40]:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        height, width = image.shape[:2]
        for click in clicks[frame]:
            if click.is_ignore or not click.cls:
                continue
            half_w, half_h = max(8.0, width * 0.015), max(8.0, height * 0.03)
            box = [click.x - half_w, click.y - half_h, click.x + half_w, click.y + half_h]
            bars = find_hp_bars(image, box)
            key = "+".join(sorted(bars)) or "none"
            stats[key] = stats.get(key, 0) + 1
            total += 1
    print(f"真敌人 {total} 个，血条组合: {stats}")
    killed = stats.get("ally", 0)
    print(f"只看到蓝条（会被丢）: {killed} ({killed/max(1,total):.1%})")
    print(f"同时有橙条+蓝条（保留）: {stats.get('ally+enemy', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
