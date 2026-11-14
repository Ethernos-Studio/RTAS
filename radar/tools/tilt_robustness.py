"""镜头偏转对身份匹配的影响有多大？以及"先转正再识别"能救回多少？

用留出的 1-8 裁图做受控实验：人为把裁图旋转 ±θ 模拟镜头偏转，
再比较「直接识别」vs「已知倾角、转正后识别」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.embed import collect_crops, split_by_video, to_tensor  # noqa: E402

CODES = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
         "U1", "U3", "U7", "U9", "U17"]
HOLDOUT = {"v001_1-8"}


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 1e-3:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def main() -> int:
    checkpoint = torch.load(ROOT / "weights" / "embedder.pt", map_location="cpu")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(512, int(checkpoint["dim"])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def embed(image: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            vector = torch.nn.functional.normalize(model(torch.from_numpy(to_tensor(image)).unsqueeze(0)))
        return vector.squeeze(0).numpy()

    items = collect_crops(ROOT / "dataset" / "crops", CODES)
    train_items, val_items = split_by_video(items, HOLDOUT)
    train_codes = {item.code for item in train_items}
    val_items = [item for item in val_items if item.code in train_codes]

    prototypes: dict[str, np.ndarray] = {}
    for code in sorted({item.code for item in train_items}):
        vectors = []
        for item in train_items:
            if item.code != code:
                continue
            image = cv2.imread(str(item.path))
            if image is not None:
                vectors.append(embed(image))
        if vectors:
            mean = np.mean(np.stack(vectors), axis=0)
            prototypes[code] = mean / max(np.linalg.norm(mean), 1e-6)
    labels = sorted(prototypes)
    matrix = np.stack([prototypes[code] for code in labels])
    print(f"原型 {len(labels)} 类 / 验证 {len(val_items)} 张")

    rng = np.random.default_rng(0)
    for tilt in (0, 2, 4, 6, 8):
        hits_raw = hits_fixed = total = 0
        for item in val_items:
            image = cv2.imread(str(item.path))
            if image is None:
                continue
            angle = float(rng.uniform(-tilt, tilt)) if tilt else 0.0
            disturbed = rotate(image, angle)
            total += 1
            if labels[int(np.argmax(matrix @ embed(disturbed)))] == item.code:
                hits_raw += 1
            # 已知倾角 -> 反向转正
            if labels[int(np.argmax(matrix @ embed(rotate(disturbed, -angle))))] == item.code:
                hits_fixed += 1
        print(f"模拟偏转 ±{tilt:>2}°: 直接识别 {hits_raw / max(1, total):5.1%} | "
              f"转正后 {hits_fixed / max(1, total):5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
