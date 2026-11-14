"""从解包出来的 2712 张敌人图鉴图里，挑出我们类别表用到的那些。

数据来源：
* 图鉴图：ArkUnpacker 解 `refs/arts/enm_art_*.ab` 得到的 `enemy_<id>_<nick>.png`（158x158，官方画质）
* 编号映射：官方 `enemy_handbook_table.json` 里的 `enemyIndex`（B1/O1/U17 这种码）
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

HANDBOOK = ROOT / "out" / "enemy_handbook_table.json"
ART_DIR = ROOT / "out" / "enm_art"
OUT_DIR = ROOT / "reference" / "enemies"
MAP_FILE = ROOT / "configs" / "enemy_ids.json"


def main() -> int:
    catalog = json.loads((ROOT / "configs" / "enemies.json").read_text(encoding="utf-8"))
    codes = [code for code, item in catalog["enemies"].items() if item.get("cls") in ("enemy", "ally")]

    handbook = json.loads(HANDBOOK.read_text(encoding="utf-8-sig"))["enemyData"]
    by_index: dict[str, tuple[str, str]] = {}
    for _key, value in handbook.items():
        index = (value.get("enemyIndex") or "").strip()
        if index:
            by_index[index] = (value.get("enemyId") or "", value.get("name") or "")

    icons = {path.stem: path for path in ART_DIR.rglob("enemy_*.png")}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, dict[str, str]] = {}
    missing = []
    for code in codes:
        entry = by_index.get(code)
        if not entry:
            missing.append(f"{code}(无编号)")
            continue
        enemy_id, name = entry
        source = icons.get(enemy_id)
        if source is None:
            missing.append(f"{code}({enemy_id} 无图)")
            continue
        target = OUT_DIR / f"{code.replace('/', '_')}.png"
        shutil.copyfile(source, target)
        mapping[code] = {"enemy_id": enemy_id, "name": name, "icon": target.name}
        print(f"{code:<4} -> {enemy_id:<24} {name:<10} {source.stat().st_size // 1024} KB")

    MAP_FILE.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n参考图 {len(mapping)} 张 -> {OUT_DIR}")
    print(f"编号映射 -> {MAP_FILE}")
    if missing:
        print("缺:", "、".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
