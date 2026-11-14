"""把检出的每根"条"放大裁出来看看到底是什么（别信统计数字）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.bars import find_bars  # noqa: E402

FRAME = ROOT / "out" / "_battle_frame.jpg"


def main() -> int:
    image = cv2.imread(str(FRAME))
    bars = find_bars(image, min_width=14)
    print(f"共 {len(bars)} 根：", {kind: sum(1 for b in bars if b.kind == kind)
                                  for kind in ("red", "green", "blue")})

    tiles = []
    for index, bar in enumerate(bars, 1):
        x1, y1 = max(0, bar.x1 - 14), max(0, bar.y1 - 26)
        x2, y2 = min(image.shape[1], bar.x2 + 14), min(image.shape[0], bar.y2 + 14)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        scale = max(1, int(96 / max(crop.shape[1], 1)))
        crop = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
        crop = cv2.resize(crop, (96, 72))
        cv2.putText(crop, f"{index}:{bar.kind[:1]} {bar.width}x{bar.height}", (2, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        cv2.rectangle(crop, (0, 0), (95, 71), (80, 80, 80), 1)
        tiles.append(crop)
        print(f"  {index:2d} {bar.kind:<6} x={bar.x1:4d} y={bar.y1:4d} w={bar.width:3d} h={bar.height}")

    rows = []
    for i in range(0, len(tiles), 8):
        chunk = tiles[i:i + 8]
        while len(chunk) < 8:
            chunk.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(chunk))
    out = ROOT / "out" / "_bars_zoom.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
