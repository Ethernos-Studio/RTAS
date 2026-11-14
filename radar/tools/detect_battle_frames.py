"""判断一帧是不是战斗画面：在顶部整条里找"敌人 x/y"那个橙色图标。

之前两版都不行：
* 中心白字占比 → 雪地/全屏火焰全被当成标题卡（R8-6 误报 161/169）；
* 固定位置整条模板匹配 → 不同录像的 HUD 横向位置不一样（1-8 在 620~1300，R8-6 在 960~1400），
  固定裁剪直接全灭。

所以改成：**只取橙色图标做模板，在顶部整条里滑窗找**，找得到就是战斗画面。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

ICON_TEMPLATE = ROOT / "configs" / "hud_icon.png"
BAND = (0, 90)          # 只看顶部 90px
SCORE_OK = 0.50         # 低于它的排进"先检查"清单（不是硬过滤：亮背景下的战斗帧也会掉到 0.5）


def grab_icon(frame_path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    image = cv2.imread(str(frame_path))
    x1, y1, x2, y2 = box
    icon = image[y1:y2, x1:x2]
    cv2.imwrite(str(ICON_TEMPLATE), icon)
    return icon


def hud_score(image: np.ndarray, icon: np.ndarray) -> float:
    y1, y2 = BAND
    band = image[y1:y2]
    # 模板比带子矮，滑窗匹配（TM_CCOEFF_NORMED）
    if band.shape[0] < icon.shape[0] or band.shape[1] < icon.shape[1]:
        return 0.0
    result = cv2.matchTemplate(band, icon, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def main() -> int:
    icon_path = ICON_TEMPLATE
    if not icon_path.exists():
        icon = grab_icon(ROOT / "frames" / "v000_R8-6_f002000.jpg", (990, 6, 1085, 52))
        print(f"生成图标模板: {icon_path} (来自 R8-6 战斗帧)")
    else:
        icon = cv2.imread(str(icon_path))
    print(f"模板尺寸: {icon.shape[1]}x{icon.shape[0]}")

    targets = sys.argv[1:] or ["v000_1-10", "v000_R8-6", "v001_1-8", "v002_1-9", "v000_1-7"]
    for video in targets:
        frames = sorted((ROOT / "frames").glob(f"{video}_*.jpg"))
        weak = []
        for path in frames:
            image = cv2.imread(str(path))
            if image is None:
                continue
            score = hud_score(image, icon)
            if score < SCORE_OK:
                weak.append((path.name, score))
        print(f"{video}: {len(frames)} 帧 → 非战斗 {len(weak)} 帧")
        for name, score in weak[:10]:
            print(f"    Skip? {name}  score={score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
