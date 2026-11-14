"""把"内存读出来的自动标注"转成我们的 clicks/*.jsonl。

数据来源：群友那套工具（`unitext` / `battlemon`）导出的 JSON / JSONL / CSV。
**只用它的结论（字段语义），不依赖它的代码和运行环境。**

设计要点：
* **宽容解析**：字段名做别名归一（enemy/name/key、map_x/x/col、screen_x/sx/px …），
  因为不同工具版本导出的字段名不一致；
* **只产"点"，不产框**：把屏幕中心点写成 clicks，交给 `radar build` 的 SAM 去精修框。
  这样能自动吃掉那 25~30px 的 y 偏差（脚底 vs Transform 原点）；
* **编号映射**：`enemy_1007_slime` -> `B1`，靠 `configs/enemy_ids.json`；认不出的落 `other_enemy`；
* **时间对齐**：dump 里若只有时间戳，就用 `frames/manifest.jsonl` 的最近帧（默认容差 0.1s）；
* **幂等 & 可追溯**：op 标记为 `auto:<工具名>`，uid 由内容哈希生成，重复导入不会翻倍。

用法：
    uv run python tools/auto_label_import.py --input dump.json --video v000_1-9 \
        --out clicks/auto_v000_1-9.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.frames import load_manifest  # noqa: E402
from eyesofpriestess.schema import Click, write_clicks  # noqa: E402

#: 不同工具版本可能用的字段名（小写后比对）
ALIASES = {
    "name": ("name", "key", "enemy", "enemy_id", "enemyid", "type", "code", "id"),
    "map_x": ("map_x", "mapx", "col", "grid_x", "tile_x", "x_map"),
    "map_y": ("map_y", "mapy", "row", "grid_y", "tile_y", "y_map"),
    "screen_x": ("screen_x", "screenx", "sx", "px", "x", "cx", "center_x"),
    "screen_y": ("screen_y", "screeny", "sy", "py", "y", "cy", "center_y"),
    "world_x": ("world_x", "worldx", "wx"),
    "world_y": ("world_y", "worldy", "wy"),
    "world_z": ("world_z", "worldz", "wz"),
    "time": ("t", "time", "ts", "timestamp", "sec", "seconds"),
    "frame": ("frame", "frame_name", "filename", "file"),
    "box": ("box", "bbox", "rect", "screen_box", "xyxy"),
}


def to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_entries(path: Path) -> list[dict]:
    """读 JSON / JSONL / CSV，统一成 dict 列表。"""
    text = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    if not text:
        return []
    if path.suffix.lower() == ".csv" or (text.count(",") > text.count("{") and not text.startswith("[")):
        return list(csv.DictReader(text.splitlines()))

    # 整个文件是一个 JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 逐行 JSONL
        return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("units", "enemies", "data", "items", "records", "frames"):
            if isinstance(data.get(key), list):
                return [item for item in data[key] if isinstance(item, dict)]
        return [data]
    return []


def pick(entry: dict, kind: str):
    """按别名取值（大小写不敏感）。"""
    lowered = {str(key).lower(): value for key, value in entry.items()}
    for alias in ALIASES[kind]:
        if alias in lowered and lowered[alias] not in (None, ""):
            return lowered[alias]
    return None


def build_code_map() -> tuple[dict[str, str], set[str]]:
    """enemy_id / 名字 -> 图鉴编号；同时返回可用编号集合。"""
    mapping = json.loads((ROOT / "configs" / "enemy_ids.json").read_text(encoding="utf-8"))
    by_id: dict[str, str] = {}
    codes: set[str] = set()
    for code, info in mapping.items():
        codes.add(code)
        enemy_id = str(info.get("enemy_id") or "")
        if enemy_id:
            by_id[enemy_id.lower()] = code
            by_id[enemy_id.split("_")[-1].lower()] = code      # 允许只写短名
        name = str(info.get("name") or "")
        if name:
            by_id[name] = code
    return by_id, codes


def resolve_code(raw, by_id: dict[str, str], codes: set[str]) -> str:
    """把工具里的敌人标识映射到图鉴编号，认不出就 other_enemy。"""
    if raw is None:
        return "other_enemy"
    text = str(raw).strip()
    if not text:
        return "other_enemy"
    if text in codes:
        return text
    key = text.lower()
    if key in by_id:
        return by_id[key]
    if key.split("_")[-1] in by_id:
        return by_id[key.split("_")[-1]]
    return "other_enemy"


def main() -> int:
    parser = argparse.ArgumentParser(description="自动标注 dump -> clicks JSONL")
    parser.add_argument("--input", required=True, help="JSON / JSONL / CSV")
    parser.add_argument("--video", required=True, help="这段 dump 属于哪个视频（如 v000_1-9）")
    parser.add_argument("--out", required=True, help="输出 clicks JSONL")
    parser.add_argument("--frames", help="帧目录（默认 frames/）")
    parser.add_argument("--time-tolerance", type=float, default=0.1, help="时间对齐容差（秒）")
    parser.add_argument("--screen-size", default="1920x1080", help="dump 里屏幕坐标的分辨率")
    parser.add_argument("--op", default="auto:unitext", help="写进 clicks 的 op 字段")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frames_dir = Path(args.frames) if args.frames else ROOT / "frames"
    manifest = load_manifest(frames_dir / "manifest.jsonl")
    screen_w, screen_h = (int(v) for v in args.screen_size.lower().split("x"))

    entries = load_entries(Path(args.input))
    by_id, codes = build_code_map()
    video_frames = {name: entry for name, entry in manifest.items() if entry.video == args.video}
    if not video_frames:
        print(f"错误: manifest 里没有视频 {args.video}", file=sys.stderr)
        return 1
    frame_list = sorted(video_frames.items(), key=lambda kv: kv[1].t)

    clicks: list[Click] = []
    stats = {"entries": len(entries), "used": 0, "no_screen": 0, "no_frame": 0,
             "clamped": 0, "unknown_code": 0}

    for entry in entries:
        name = pick(entry, "name")
        screen_x = to_float(pick(entry, "screen_x"))
        screen_y = to_float(pick(entry, "screen_y"))

        # 没有屏幕坐标：尝试用世界坐标（相机投影留给调用方，这里只记录跳过原因）
        if screen_x is None or screen_y is None:
            world_x = to_float(pick(entry, "world_x"))
            world_y = to_float(pick(entry, "world_y"))
            if world_x is not None and world_y is not None:
                # 兜底：世界坐标 -> 屏幕（用 docs 里的结论公式，列主序透视）
                from eyesofpriestess.cv.camera import BattleCamera

                camera = BattleCamera(screen_w=screen_w, screen_h=screen_h)
                projected = camera.project((world_x, world_y, 0.0))
                if projected is None:
                    stats["no_screen"] += 1
                    continue
                screen_x, screen_y = projected[0], projected[1]
            else:
                stats["no_screen"] += 1
                continue

        # 对齐到具体帧
        frame_name = None
        explicit = pick(entry, "frame")
        if explicit and str(explicit) in video_frames:
            frame_name = str(explicit)
        else:
            seconds = to_float(pick(entry, "time"))
            if seconds is not None:
                nearest = min(frame_list, key=lambda kv: abs(kv[1].t - seconds))
                if abs(nearest[1].t - seconds) <= args.time_tolerance:
                    frame_name = nearest[0]
        if frame_name is None:
            stats["no_frame"] += 1
            continue

        # 坐标要落在画面内
        if not (0 <= screen_x <= screen_w and 0 <= screen_y <= screen_h):
            stats["clamped"] += 1
            continue

        code = resolve_code(name, by_id, codes)
        if code == "other_enemy" and name:
            stats["unknown_code"] += 1

        digest = hashlib.sha1(f"{frame_name}:{screen_x:.0f}:{screen_y:.0f}:{code}".encode()).hexdigest()[:16]
        clicks.append(Click(
            frame=frame_name,
            x=round(screen_x, 1),
            y=round(screen_y, 1),
            cls=code,
            type="point",
            op=args.op,
            uid=digest,
        ))
        stats["used"] += 1

    out = Path(args.out)
    if args.dry_run:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        print(f"（dry-run）将写 {len(clicks)} 个点到 {out}")
        return 0

    write_clicks(out, clicks)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"写入 {len(clicks)} 个点 -> {out}")
    print("下一步: uv run radar build   # SAM 会把点精修成框（顺带吃掉 25~30px 的 y 偏差）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
