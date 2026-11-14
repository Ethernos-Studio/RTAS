"""从游戏窗口抓一段帧存成图片（给"负样本/背景图"和标注用）。

用法：
    # 站在主界面或刚进关的部署阶段（画面里只有干员、没有敌人）抓 40 帧
    uv run python tools/live_capture.py --seconds 8 --fps 5 --out raw_live/menu

这些帧会作为**背景图**进训练集（空标签），专门教模型"这些东西不是敌人"，
正好治现在把干员、立绘、召唤物当敌人的毛病。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="抓游戏窗口的帧存图")
    parser.add_argument("--out", default="raw_live/capture")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--window-title", help="窗口标题关键字")
    parser.add_argument("--wgc", default=str(Path(r"E:\RTAS\src\rtas\tools\_vendor\wgc\wgc_python.dll")))
    args = parser.parse_args()

    from live_overlay import PrintWindowGrabber, WgcGrabber, find_game_window

    info = find_game_window(args.window_title)
    if info is None:
        print("找不到游戏窗口", file=sys.stderr)
        return 2
    print(f"窗口: {info.title} {info.client}")

    grabber = None
    try:
        grabber = WgcGrabber(Path(args.wgc), info.title, info.class_name)
        if grabber.grab() is None:
            grabber = None
    except Exception as exc:  # noqa: BLE001
        print(f"WGC 不可用: {exc}", file=sys.stderr)
    if grabber is None:
        grabber = PrintWindowGrabber(info.hwnd)
        print("用 PrintWindow 抓屏")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / max(0.5, args.fps)
    deadline = time.time() + args.seconds
    saved = 0
    while time.time() < deadline:
        frame = grabber.grab()
        if frame is not None and frame.size:
            saved += 1
            cv2.imwrite(str(out / f"cap_{saved:04d}.jpg"), frame)
        time.sleep(interval)
    grabber.close()
    print(f"存了 {saved} 帧 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
