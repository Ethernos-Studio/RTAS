"""第二轮实验：预处理层面的改进（保比例 letterbox / 椭圆中心掩码 / 掩掉背景色）。"""

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

from eyesofpriestess.embed import MEAN, STD, collect_crops, split_by_video  # noqa: E402

CODES = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
         "U1", "U3", "U7", "U9", "U17"]
HOLDOUT = {"v001_1-8"}
SIZE = 224


def _norm(rgb: np.ndarray) -> np.ndarray:
    return np.transpose((rgb.astype(np.float32) / 255.0 - MEAN) / STD, (2, 0, 1))


def letterbox(image: np.ndarray) -> np.ndarray:
    """保比例缩放到 224，四周补黑（不拉伸）。"""
    height, width = image.shape[:2]
    scale = SIZE / max(height, width)
    resized = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    top = (SIZE - resized.shape[0]) // 2
    left = (SIZE - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return _norm(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def squash(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return _norm(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def ellipse(image: np.ndarray) -> np.ndarray:
    """压暗四角：sprite 一般居中，四角基本是地图背景。"""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(mask, (width // 2, height // 2), (int(width * 0.46), int(height * 0.46)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)[:, :, None]
    return squash((image.astype(np.float32) * mask).astype(np.uint8))


def drop_background(image: np.ndarray) -> np.ndarray:
    """把"和四边颜色接近"的像素当背景抹黑（地图是低饱和的大色块）。"""
    height, width = image.shape[:2]
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]]).reshape(-1, 3).astype(np.float32)
    background = border.mean(axis=0)
    distance = np.linalg.norm(image.astype(np.float32) - background, axis=2)
    mask = (distance > 28).astype(np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[:, :, None]
    return squash((image.astype(np.float32) * mask).astype(np.uint8))


TRANSFORMS = {
    "squash(当前)": squash,
    "letterbox": letterbox,
    "椭圆掩码": ellipse,
    "去背景色": drop_background,
}


def main() -> int:
    items = collect_crops(ROOT / "dataset" / "crops", CODES)
    train_items, val_items = split_by_video(items, HOLDOUT)
    train_codes = {item.code for item in train_items}
    val_items = [item for item in val_items if item.code in train_codes]

    checkpoint = torch.load(ROOT / "weights" / "embedder.pt", map_location="cpu")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(512, int(checkpoint["dim"])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results: dict[str, float] = {}
    for name, transform in TRANSFORMS.items():
        cache: dict[Path, np.ndarray] = {}
        for item in train_items + val_items:
            image = cv2.imread(str(item.path))
            if image is None:
                continue
            tensor = torch.from_numpy(transform(image)).unsqueeze(0)
            with torch.no_grad():
                vector = torch.nn.functional.normalize(model(tensor)).squeeze(0).numpy()
            cache[item.path] = vector

        protos, labels = {}, []
        for code in sorted({item.code for item in train_items if item.path in cache}):
            vectors = [cache[item.path] for item in train_items if item.code == code and item.path in cache]
            if not vectors:
                continue
            mean = np.mean(np.stack(vectors), axis=0)
            protos[code] = mean / max(np.linalg.norm(mean), 1e-6)
            labels.append(code)
        matrix = np.stack([protos[code] for code in labels])

        hits = total = 0
        confusion: dict[str, int] = {}
        for item in val_items:
            vector = cache.get(item.path)
            if vector is None:
                continue
            total += 1
            best = labels[int(np.argmax(matrix @ vector))]
            if best == item.code:
                hits += 1
            else:
                key = f"{item.code}->{best}"
                confusion[key] = confusion.get(key, 0) + 1
        score = hits / max(1, total)
        results[name] = score
        print(f"  {name:<14} top-1 {score:6.1%}  {sorted(confusion.items(), key=lambda kv: -kv[1])[:3]}")

    print("\n== 排名 ==")
    for name, score in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {score:6.1%}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
