"""从游戏资源渲染每个编号的"干净全身参考图"。

流程（每一步都是踩过坑才定下来的）：
1. ArkUnpacker 解 `refs/arts/enm_art_*.ab` -> `.skel` + `.atlas` + 贴图（含 [alpha] 通道图）
2. ArkUnpacker `-m cb` 把 RGB 和 A 通道合成 RGBA —— 不合成的话渲染出来全透明
3. Node + Spine **3.8** 官方运行时渲染 setup pose / Idle 一帧 -> PNG
   （4.x 运行时读 3.8 的 skel 会报 string table null）

用法（先确保 out/spine_all 和 tools/spine_render/node_modules 就绪）：
    uv run python tools/render_reference.py [--size 512]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

ARK_UNPACKER = Path(
    r"E:\Arknights bilibili\games\Arknights\Arknights_Data\StreamingAssets\AB"
    r"\ArkUnpacker-v5.2.0.exe"
)
SPINE_DIR = ROOT / "out" / "spine_all"
CB_DIR = ROOT / "out" / "spine_cb_all"
NODE_DIR = ROOT / "tools" / "spine_render"
RENDERER = NODE_DIR / "render38.js"
OUT_DIR = ROOT / "reference" / "enemies"


def node_exe() -> Path:
    """fnm 装出来的 node（本机 PATH 里没有）。"""
    candidates = list((Path(os.environ["APPDATA"]) / "fnm" / "node-versions").glob("v*/installation/node.exe"))
    if not candidates:
        raise SystemExit("找不到 node.exe（fnm 的 node-versions 目录为空）")
    return sorted(candidates)[-1]


def find_model(enemy_id: str) -> Path | None:
    """找 BattleFront 下这个敌人的模型目录。"""
    for path in SPINE_DIR.rglob(f"BattleFront/{enemy_id}"):
        if (path / f"{enemy_id}.skel").exists():
            return path
    # 少数模型没有 BattleFront 目录
    for path in SPINE_DIR.rglob(f"{enemy_id}"):
        if (path / f"{enemy_id}.skel").exists():
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--codes", nargs="*", help="只渲染这些编号")
    args = parser.parse_args()

    mapping = json.loads((ROOT / "configs" / "enemy_ids.json").read_text(encoding="utf-8"))
    codes = args.codes or list(mapping)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CB_DIR.mkdir(parents=True, exist_ok=True)
    node = node_exe()

    ok, failed = [], []
    for code in codes:
        info = mapping.get(code)
        if not info:
            failed.append(f"{code}(类别表里没有)")
            continue
        enemy_id = info["enemy_id"]
        model = find_model(enemy_id)
        if model is None:
            failed.append(f"{code}({enemy_id} 没有模型文件)")
            continue

        # 1) 合并 alpha
        target = CB_DIR / enemy_id
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [str(ARK_UNPACKER), "-m", "cb", "-i", str(model), "-o", str(target), "-l", "1"],
            capture_output=True, text=True, errors="ignore", check=False,
        )
        # 2) 拷 atlas/skel（cb 只输出图）
        for suffix in (".atlas", ".skel"):
            source = model / f"{enemy_id}{suffix}"
            if source.exists():
                shutil.copyfile(source, target / source.name)

        # 3) 渲染
        out_png = OUT_DIR / f"{code}.png"
        result = subprocess.run(
            [str(node), str(RENDERER),
             str(target / f"{enemy_id}.atlas"),
             str(target / f"{enemy_id}.skel"),
             str(out_png), str(args.size), str(args.size)],
            capture_output=True, text=True, errors="ignore", check=False,
        )
        if result.returncode != 0 or not out_png.exists():
            failed.append(f"{code}(渲染失败: {(result.stderr or result.stdout).strip()[:80]})")
            continue
        ok.append(code)
        print(f"{code:<4} {enemy_id:<26} -> {out_png.name}")

    print(f"\n成功 {len(ok)} / 失败 {len(failed)}")
    for item in failed:
        print("  失败:", item)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
