"""把官方名字写进类别表（VLM 的文字里要显示"猎狗"而不是"O1"）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    catalog_path = ROOT / "configs" / "enemies.json"
    mapping_path = ROOT / "configs" / "enemy_ids.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    changed = 0
    for code, info in mapping.items():
        entry = catalog["enemies"].get(code)
        if not entry:
            continue
        if entry.get("name") != info["name"]:
            entry["name"] = info["name"]
            changed += 1
        entry["enemy_id"] = info["enemy_id"]

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"更新 {changed} 个名字")
    for code, entry in catalog["enemies"].items():
        print(f"  {code:<4} {entry.get('name',''):<12} {entry.get('enemy_id','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
