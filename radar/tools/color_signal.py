"""看看单纯用颜色统计能不能分开 B1/B2（源石虫 vs 源石虫·α）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def color_signature(image: np.ndarray) -> np.ndarray:
    """主体颜色的统计量：色相直方图 + 饱和度/亮度均值。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat > 60) & (val > 50)          # 只看"有颜色"的像素（本体）
    if mask.sum() < 10:
        mask = np.ones_like(sat, dtype=bool)
    hist = cv2.calcHist([hue], [0], mask.astype(np.uint8), [16], [0, 180]).flatten()
    hist = hist / max(hist.sum(), 1)
    return np.concatenate([hist * 2.0, [sat[mask].mean() / 255.0, val[mask].mean() / 255.0]])


def main() -> int:
    splits = {"train": ["v000_1-7", "v002_1-9", "v000_R8-6"], "val": ["v001_1-8"]}
    codes = ["B1", "B2"]
    protos, val_samples = {}, []
    for code in codes:
        for path in sorted((ROOT / "dataset" / "crops" / code).glob("*.png")):
            video = path.stem.split("_f")[0]
            image = cv2.imread(str(path))
            if image is None:
                continue
            signature = color_signature(image)
            if video in splits["train"]:
                protos.setdefault(code, []).append(signature)
            elif video in splits["val"]:
                val_samples.append((code, signature))
    prototypes = {code: np.mean(np.stack(values), axis=0) for code, values in protos.items()}

    correct = total = 0
    confusion: dict[str, int] = {}
    for expected, signature in val_samples:
        scores = {code: float(signature @ proto / (np.linalg.norm(signature) * np.linalg.norm(proto) + 1e-9))
                  for code, proto in prototypes.items()}
        best = max(scores, key=scores.get)
        total += 1
        if best == expected:
            correct += 1
        else:
            confusion[f"{expected}->{best}"] = confusion.get(f"{expected}->{best}", 0) + 1
    print(f"纯颜色 top-1: {correct}/{total} = {correct / max(1, total):.1%}  混淆 {confusion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
