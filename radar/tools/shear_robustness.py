"""量化"部署时镜头俯仰（shear）"对身份匹配的影响，以及反向补偿能救回多少。

实测（tools/compare_geometry.py）：
* 正常帧 -> 部署帧：水平尺度 0.97~1.00、垂直尺度 0.95~1.06、竖直线倾斜 3.3~4.2°
* 两个部署帧之间：完全相同（1.0000 / 1.0000 / 0.00°）—— 印证"选中干员后角度是固定的"
所以模拟时用 shear ≈ 0.06（tan 3.4°），而不是平面内旋转。
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


def shear(image: np.ndarray, amount: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1.0, 0.0, 0.0], [amount, 1.0, -amount * width / 2]])
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

    prototypes = {}
    for code in sorted({item.code for item in train_items}):
        vectors = [embed(cv2.imread(str(item.path))) for item in train_items
                   if item.code == code and cv2.imread(str(item.path)) is not None]
        if vectors:
            mean = np.mean(np.stack(vectors), axis=0)
            prototypes[code] = mean / max(np.linalg.norm(mean), 1e-6)
    labels = sorted(prototypes)
    matrix = np.stack([prototypes[code] for code in labels])

    for amount in (0.0, 0.03, 0.06, 0.09):
        hits_raw = hits_fixed = total = 0
        for item in val_items:
            image = cv2.imread(str(item.path))
            if image is None:
                continue
            disturbed = shear(image, amount)
            total += 1
            if labels[int(np.argmax(matrix @ embed(disturbed)))] == item.code:
                hits_raw += 1
            if labels[int(np.argmax(matrix @ embed(shear(disturbed, -amount))))] == item.code:
                hits_fixed += 1
        print(f"shear {amount:.2f}（竖线倾斜 {np.degrees(np.arctan(amount)):.1f}°）: "
              f"直接识别 {hits_raw/max(1,total):5.1%} | 反向补偿后 {hits_fixed/max(1,total):5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
