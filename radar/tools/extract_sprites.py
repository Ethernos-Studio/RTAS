"""用 UnityPy 直接从 .ab 里取 **Sprite**（裁好的图），而不是整张 atlas 贴图。

之前用 ArkUnpacker 的 --image 导出的是 Texture2D（一整页碎片），
敌人立绘是 Sprite：一张贴图里按 rect 裁出来的一块。UnityPy 能直接拿到裁好的图。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    import UnityPy

    ab = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / ".." / ".." / "Arknights bilibili" / "games" / "Arknights"
        / "Arknights_Data" / "StreamingAssets" / "AB" / "Windows" / "refs" / "arts" / "enm_art_0.ab"
    )
    out = ROOT / "out" / "sprites"
    out.mkdir(parents=True, exist_ok=True)

    env = UnityPy.load(str(ab))
    counts: dict[str, int] = {}
    saved = 0
    for obj in env.objects:
        counts[obj.type.name] = counts.get(obj.type.name, 0) + 1
        if obj.type.name != "Sprite":
            continue
        data = obj.read()
        name = getattr(data, "m_Name", "") or f"sprite_{saved}"
        try:
            image = data.image  # 裁好的 PIL Image
        except Exception as exc:  # noqa: BLE001
            print(f"跳过 {name}: {type(exc).__name__}")
            continue
        image.save(out / f"{name}.png")
        saved += 1
        if saved <= 8:
            print(f"{name}: {image.size}")
    print("对象类型统计:", counts)
    print(f"导出 Sprite {saved} 张 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
