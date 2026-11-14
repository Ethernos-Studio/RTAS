"""关卡标定：用几组「格子坐标 ↔ 屏幕像素」对应点，拟合这张图的地图→世界偏移。

为什么需要：资料里的相机参数只在**一张图**上验证过（且作者说 Y 可能不对），
所以格子坐标不能靠全局公式，只能每张图用真实对应点标一次——这也和我们
`cv/calib.py`（每关一个单应）的思路一致。

对应点怎么来（任选）：
* 在截图里手点 3~4 个**格子中心**，记下它们的 (col,row) 和像素 (x,y)；
* 或者用 LS 的任务，把已知格子的单位点出来。

输入 JSON::

    [
      {"col": 1, "row": 2, "x": 812, "y": 640},
      {"col": 3, "row": 2, "x": 1032, "y": 640},
      {"col": 1, "row": 4, "x": 812, "y": 800}
    ]

用法：
    uv run python tools/calibrate_stage.py --stage 1-7 \
        --points stage_points.json --out configs/stage_calib.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.camera import BattleCamera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="用对应点拟合每关的地图→世界偏移")
    parser.add_argument("--stage", required=True, help="关卡名（如 1-7）")
    parser.add_argument("--points", required=True, help="JSON：col/row/x/y 列表")
    parser.add_argument("--out", default="configs/stage_calib.json")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    points = json.loads(Path(args.points).read_text(encoding="utf-8-sig"))
    if not isinstance(points, list) or len(points) < 3:
        print("错误: 至少需要 3 组对应点", file=sys.stderr)
        return 1

    camera = BattleCamera(screen_w=args.width, screen_h=args.height)
    correspondences = [
        (float(item["col"]), float(item["row"]), float(item["x"]), float(item["y"]))
        for item in points
    ]
    a, b = camera.fit_offset(correspondences)

    residuals = []
    for col, row, x, y in correspondences:
        projected = camera.project_map(col, row, (a, b))
        if projected is None:
            residuals.append(float("inf"))
            continue
        residuals.append(((projected[0] - x) ** 2 + (projected[1] - y) ** 2) ** 0.5)

    worst = max(residuals)
    mean = sum(residuals) / len(residuals)
    print(f"{args.stage}: offset a={a:.3f} b={b:.3f}  重投影误差 平均 {mean:.1f}px / 最大 {worst:.1f}px")
    if worst > 40:
        print("警告: 误差偏大，可能相机参数不适用于这张图（资料只在一张图上验证过）")

    out = Path(args.out)
    data = {"stages": []}
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8")) or {"stages": []}
    data["stages"] = [item for item in data.get("stages", []) if item.get("stage") != args.stage]
    data["stages"].append({"stage": args.stage, "offset": [round(a, 4), round(b, 4)],
                           "residual_mean_px": round(mean, 2), "residual_max_px": round(worst, 2)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
