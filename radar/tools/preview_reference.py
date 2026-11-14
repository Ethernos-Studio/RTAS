"""把 reference/enemies/*.png 拼成一张对照图，肉眼确认挑对了。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

COLS = 6


def main() -> int:
    files = sorted((ROOT / "reference" / "enemies").glob("*.png"))
    tiles = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (128, 128))
        canvas = np.full((154, 144, 3), 30, np.uint8)
        canvas[0:128, 8:136] = image
        cv2.putText(canvas, path.stem, (10, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
        tiles.append(canvas)

    blank = np.full_like(tiles[0], 30)
    while len(tiles) % COLS:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    out = ROOT / "out" / "_ref_gallery.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"{len(files)} 张 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
