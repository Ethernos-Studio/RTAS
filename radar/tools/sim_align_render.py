"""把对齐结果画到帧上，肉眼验证 sim 位置和敌人是否重合。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"E:\RTAS\src")
sys.stdout.reconfigure(encoding="utf-8")

GAMEDATA = Path(r"E:\RTAS\ArknightsGameData\zh_CN\gamedata")


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "01-07"
    data = json.loads((ROOT / "out" / f"sim_align_{stage}.json").read_text(encoding="utf-8"))
    frame_name, time_s = data["frame"], data["time"]
    matrix = np.array(data["homography"], dtype=np.float64)

    from rtas.sim.engine import simulate
    from rtas.sim.loader import load_level
    from eyesofpriestess.schema import load_clicks

    level = load_level(GAMEDATA / "levels" / "obt" / "main" / f"level_main_{stage}.json")
    result = simulate(level, None, duration=max(60.0, time_s + 1), fps=2.0, dt=0.1)
    frame = result.frame_at(time_s)

    image = cv2.imread(str(ROOT / "frames" / frame_name))
    for enemy in frame.enemies:
        point = matrix @ np.array([enemy.col, enemy.row, 1.0])
        if abs(point[2]) < 1e-9:
            continue
        x, y = point[0] / point[2], point[1] / point[2]
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            continue
        cv2.circle(image, (int(x), int(y)), 6, (0, 0, 255), -1)
        label = enemy.enemy_id.replace("enemy_", "")[:18]
        cv2.putText(image, label, (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1)

    clicks = load_clicks(ROOT / "clicks")
    for click in clicks.get(frame_name, []):
        if not click.is_ignore:
            cv2.drawMarker(image, (int(click.x), int(click.y)), (0, 255, 0),
                           cv2.MARKER_CROSS, 14, 2)

    out = ROOT / "out" / f"_sim_align_{stage}.jpg"
    cv2.imwrite(str(out), image)
    print(f"{len(frame.enemies)} 个 sim 点 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
