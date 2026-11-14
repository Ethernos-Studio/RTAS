"""身份匹配方案对比实验：同一划分下试多种特征/融合策略，挑最好的。

划分：留出整段 `v001_1-8` 当验证（同帧相邻裁图不跨集），其余视频训练。
指标：top-1（只统计训练里出现过的编号，没见过的编号不能用原型匹配）。

方案：
  A emb-imagenet   ImageNet resnet18 特征 + 原型均值
  B emb-trained    训练后的 ArcFace embedding + 原型均值
  C color          色相直方图 + 饱和度/亮度均值
  D fusion         A/B 的余弦 + λ × 颜色余弦（λ 扫一遍）
  E emb+flip       embedding 对水平翻转做 TTA
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
from eyesofpriestess.identity import Encoder  # noqa: E402

CODES = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
         "U1", "U3", "U7", "U9", "U17"]
HOLDOUT = {"v001_1-8"}


def color_signature(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (sat > 60) & (val > 50)
    if mask.sum() < 10:
        mask = np.ones_like(sat, dtype=bool)
    hist = cv2.calcHist([hue], [0], mask.astype(np.uint8), [16], [0, 180]).flatten()
    hist = hist / max(hist.sum(), 1)
    signature = np.concatenate([hist * 2.0, [sat[mask].mean() / 255.0, val[mask].mean() / 255.0]])
    return signature / max(np.linalg.norm(signature), 1e-6)


def trained_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(512, int(checkpoint["dim"])))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> int:
    items = collect_crops(ROOT / "dataset" / "crops", CODES)
    train_items, val_items = split_by_video(items, HOLDOUT)
    train_codes = {item.code for item in train_items}
    val_items = [item for item in val_items if item.code in train_codes]
    print(f"训练 {len(train_items)} / 验证 {len(val_items)}（{sorted({i.code for i in val_items})}）")

    baseline = Encoder("resnet18")
    arc = trained_model(ROOT / "weights" / "embedder.pt")

    def emb_imagenet(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        return baseline.embed([image])[0] if image is not None else np.zeros(512, np.float32)

    def emb_trained(path: Path, flip: bool = False) -> np.ndarray:
        image = cv2.imread(str(path))
        if image is None:
            return np.zeros(256, np.float32)
        variants = [image] + ([cv2.flip(image, 1)] if flip else [])
        with torch.no_grad():
            tensor = torch.stack([torch.from_numpy(to_tensor(v)) for v in variants])
            out = torch.nn.functional.normalize(arc(tensor)).mean(0)
        return (out / out.norm()).numpy()

    def color(path: Path) -> np.ndarray:
        image = cv2.imread(str(path))
        return color_signature(image) if image is not None else np.zeros(18, np.float32)

    feats = {
        "emb-imagenet": lambda p: emb_imagenet(p),
        "emb-trained": lambda p: emb_trained(p),
        "emb-trained-flip": lambda p: emb_trained(p, flip=True),
        "color": color,
    }

    cache: dict[str, dict[Path, np.ndarray]] = {}
    for name, fn in feats.items():
        print(f"  提取特征: {name}")
        cache[name] = {item.path: fn(item.path) for item in train_items + val_items}

    def evaluate(name: str, train_key: str, val_key: str | None = None) -> tuple[float, dict[str, int]]:
        val_key = val_key or train_key
        protos, labels = {}, []
        for code in sorted({item.code for item in train_items}):
            vectors = [cache[train_key][item.path] for item in train_items if item.code == code]
            if not vectors:
                continue
            mean = np.mean(np.stack(vectors), axis=0)
            protos[code] = mean / max(np.linalg.norm(mean), 1e-6)
            labels.append(code)
        matrix = np.stack([protos[code] for code in labels])
        hits = 0
        confusion: dict[str, int] = {}
        for item in val_items:
            vector = cache[val_key][item.path]
            scores = matrix @ vector
            best = labels[int(np.argmax(scores))]
            if best == item.code:
                hits += 1
            else:
                key = f"{item.code}->{best}"
                confusion[key] = confusion.get(key, 0) + 1
        total = max(1, len(val_items))
        print(f"  {name:<28} top-1 {hits / total:6.1%}  {sorted(confusion.items(), key=lambda kv: -kv[1])[:3]}")
        return hits / total, confusion

    results: dict[str, float] = {}
    for name in feats:
        results[name], _ = evaluate(name, name)

    # 融合：embedding 余弦 + λ × 颜色余弦
    best_lam, best_score = 0.0, 0.0
    for source in ("emb-imagenet", "emb-trained", "emb-trained-flip"):
        for lam in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
            protos_emb, protos_color, labels = {}, {}, []
            for code in sorted({item.code for item in train_items}):
                vecs = [cache[source][item.path] for item in train_items if item.code == code]
                cols = [cache["color"][item.path] for item in train_items if item.code == code]
                if not vecs:
                    continue
                mean = np.mean(np.stack(vecs), axis=0)
                protos_emb[code] = mean / max(np.linalg.norm(mean), 1e-6)
                cmean = np.mean(np.stack(cols), axis=0)
                protos_color[code] = cmean / max(np.linalg.norm(cmean), 1e-6)
                labels.append(code)

            hits = 0
            for item in val_items:
                scores = np.array([
                    float(protos_emb[code] @ cache[source][item.path])
                    + lam * float(protos_color[code] @ cache["color"][item.path])
                    for code in labels
                ])
                if labels[int(np.argmax(scores))] == item.code:
                    hits += 1
            score = hits / max(1, len(val_items))
            tag = f"fusion[{source}, λ={lam}]"
            print(f"  {tag:<28} top-1 {score:6.1%}")
            results[tag] = score
            if score > best_score:
                best_score, best_lam = score, lam

    print("\n== 排名 ==")
    for name, score in sorted(results.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  {score:6.1%}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
