"""把 find_ui_frames 标出的可疑帧拼成一张图，肉眼确认。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

NAMES = ["v000_1-10_f000520.jpg", "v000_1-10_f005780.jpg",
         "v000_1-10_f005800.jpg", "v000_1-10_f006420.jpg"]

tiles = []
for name in NAMES:
    image = cv2.imread(str(ROOT / "frames" / name))
    if image is None:
        continue
    cv2.putText(image, name[-12:-4], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    tiles.append(cv2.resize(image, (480, 270)))

while len(tiles) % 2:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
cv2.imwrite(str(ROOT / "out" / "_ui_frames.jpg"), np.vstack(rows))
print("wrote out/_ui_frames.jpg")
