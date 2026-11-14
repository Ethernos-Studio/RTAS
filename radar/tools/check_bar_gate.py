"""在带标注的帧上验证血条门控：会不会把真敌人（红条）误杀。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.hpbar_gate import bar_color_at  # noqa: E402
from eyesofpriestess.schema import load_clicks  # noqa: E402


def main() -> int:
    clicks = load_clicks(ROOT / "clicks")
    stats: dict[str, int] = {}
    total = 0
    for frame, items in clicks.items():
        if not frame.startswith(("v001_1-8", "v002_1-9", "v000_1-10")):
            continue
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        height, width = image.shape[:2]
        for click in items:
            if click.is_ignore or not click.cls:
                continue
            half_w, half_h = max(8.0, width * 0.015), max(8.0, height * 0.03)
            box = [click.x - half_w, click.y - half_h, click.x + half_w, click.y + half_h]
            color = bar_color_at(image, box) or "none"
            stats[color] = stats.get(color, 0) + 1
            total += 1
    print(f"真敌人 {total} 个，血条颜色分布：{stats}")
    killed = stats.get("green", 0) + stats.get("blue", 0)
    print(f"会被误杀: {killed} ({killed/max(1,total):.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
