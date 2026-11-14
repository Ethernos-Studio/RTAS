"""敌人身份 embedding：在战斗裁图上做度量学习（ArcFace），替掉 ImageNet 特征。

为什么需要：ImageNet 预训练的 resnet18 对 40px 像素小人几乎没概念，
直接拿它做最近邻，top-1 只有 ~55%，主要混在"源石虫 / 源石虫·α"这种差分上。
这里用 2321 张人工标注裁图（+ 17 张官方渲染图当类锚点）微调一个 ArcFace 头，
让同编号的裁图在特征空间里聚起来。

关键约束：
* **按视频划分训练/验证**（同一帧的相邻裁图几乎一样，按裁图随机切会泄漏）；
* 训练完存 `weights/embedder.pt`，`gallery-build --embedder` 和
  `identity-check --embedder`、`predict --embedder` 都认它。
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
EMBED_DIM = 256


@dataclass
class CropItem:
    path: Path
    code: str
    video: str


def collect_crops(crops_dir: Path, codes: list[str]) -> list[CropItem]:
    """收集裁图；只看类别表里的编号（旧名字如 01/02 会被自然过滤掉）。"""
    items: list[CropItem] = []
    for folder in sorted(Path(crops_dir).iterdir()):
        if not folder.is_dir() or folder.name not in codes:
            continue
        for path in sorted(folder.glob("*.png")):
            video = path.stem.split("_f")[0]
            items.append(CropItem(path=path, code=folder.name, video=video))
    return items


def split_by_video(items: list[CropItem], val_videos: set[str]) -> tuple[list[CropItem], list[CropItem]]:
    train = [item for item in items if item.video not in val_videos]
    val = [item for item in items if item.video in val_videos]
    return train, val


def _augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """轻量增强：随机裁剪缩放 + 亮度/对比度抖动 + 偶尔水平翻转。"""
    height, width = image.shape[:2]
    scale = rng.uniform(0.75, 1.0)
    target_h = max(8, int(height * scale))
    target_w = max(8, int(width * scale))
    top = rng.randint(0, max(0, height - target_h))
    left = rng.randint(0, max(0, width - target_w))
    crop = image[top:top + target_h, left:left + target_w]
    if rng.random() < 0.3:
        crop = cv2.flip(crop, 1)
    if rng.random() < 0.5:
        alpha = rng.uniform(0.85, 1.15)
        beta = rng.randint(-15, 15)
        crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return crop


def letterbox(image: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """保比例缩放到 size×size 并居中补黑。

    实验（tools/identity_experiments2.py，留出整段 1-8）：拉伸 78.1% -> letterbox 86.2%。
    敌人的"近大远小"只改变尺寸、不改变形状，所以千万别把长宽比揉掉。
    """
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / max(height, width)
    resized = cv2.resize(image, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def to_tensor(image: np.ndarray) -> np.ndarray:
    """BGR 图 -> (3,224,224) 归一化数组（letterbox，不拉伸）。"""
    square = letterbox(image)
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1))


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    backbone_lr: float = 1e-4
    weight_decay: float = 5e-4
    margin: float = 0.1
    scale: float = 30.0
    reference_dir: Path | None = None
    seed: int = 0


def train_embedder(
    items: list[CropItem],
    val_items: list[CropItem],
    codes: list[str],
    out_path: Path,
    config: TrainConfig | None = None,
) -> dict:
    """训练 ArcFace embedding，返回评估报告。"""
    import torch
    from torch import nn
    from torchvision import models

    config = config or TrainConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    code_to_index = {code: index for index, code in enumerate(codes)}

    # 收集样本（路径 + 标签），参考渲染图带重复权重以充当类锚点
    samples: list[tuple[Path, int]] = []
    for item in items:
        index = code_to_index.get(item.code)
        if index is not None:
            samples.append((item.path, index))
    if config.reference_dir and Path(config.reference_dir).exists():
        for path in sorted(Path(config.reference_dir).glob("*.png")):
            index = code_to_index.get(path.stem)
            if index is not None:
                samples.extend([(path, index)] * 8)     # 干净的官方图多喂几遍
    if not samples:
        raise SystemExit("没有可用的训练裁图")

    class CropDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            path, label = samples[index]
            image = cv2.imread(str(path))
            if image is None or image.size == 0:
                image = np.zeros((32, 32, 3), dtype=np.uint8)
            image = _augment(image, random)
            return torch.from_numpy(to_tensor(image)), label

    loader = torch.utils.data.DataLoader(
        CropDataset(), batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=False
    )

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    head = nn.Linear(512, EMBED_DIM)

    class ArcFace(nn.Module):
        def __init__(self, dim: int, classes: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(classes, dim) * 0.01)

        def forward(self, features, labels=None):
            cosine = nn.functional.normalize(features) @ nn.functional.normalize(self.weight).T
            if labels is None:
                return cosine
            rows = torch.arange(len(labels), device=cosine.device)
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            target = theta[rows, labels] + config.margin
            logits = cosine.clone()
            logits[rows, labels] = torch.cos(target)
            return logits * config.scale

    model = nn.Sequential(backbone, nn.Flatten(), head).to(device)
    arc = ArcFace(EMBED_DIM, len(codes)).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model[0].parameters(), "lr": config.backbone_lr},
            {"params": list(model[1:].parameters()) + list(arc.parameters()), "lr": config.lr},
        ],
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            logits = arc(features, labels)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
        scheduler.step()
        entry = {"epoch": epoch, "loss": round(loss_sum / max(1, len(loader)), 4),
                 "train_acc": round(correct / max(1, total), 4)}
        history.append(entry)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:2d}  loss={entry['loss']:.3f}  train_acc={entry['train_acc']:.3f}")

    report = evaluate_embedder(model, items, val_items, codes, device, config.reference_dir)
    report["history"] = history

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": codes,
        "arch": "resnet18",
        "size": INPUT_SIZE,
        "dim": EMBED_DIM,
    }, out_path)
    return report


def _embed_paths(model, paths: list[Path], device: str):
    import torch

    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            chunk = []
            for path in paths[start:start + 64]:
                image = cv2.imread(str(path))
                if image is None or image.size == 0:
                    image = np.zeros((32, 32, 3), dtype=np.uint8)
                chunk.append(torch.from_numpy(to_tensor(image)))
            tensor = torch.stack(chunk).to(device)
            outputs.append(torch.nn.functional.normalize(model(tensor)).cpu())
    return torch.cat(outputs) if outputs else None


def build_prototypes(model, items: list[CropItem], codes: list[str],
                     device: str, reference_dir: Path | None = None) -> tuple:
    """用训练集裁图（+ 官方渲染图）算每个编号的原型向量，和线上图库同一套做法。"""
    import torch

    by_code: dict[str, list[Path]] = {}
    for item in items:
        by_code.setdefault(item.code, []).append(item.path)
    if reference_dir and Path(reference_dir).exists():
        for path in sorted(Path(reference_dir).glob("*.png")):
            by_code.setdefault(path.stem, []).append(path)

    prototypes, labels = [], []
    for code in codes:
        paths = by_code.get(code) or []
        if not paths:
            continue
        features = _embed_paths(model, paths, device)
        vector = features.mean(dim=0)
        prototypes.append(torch.nn.functional.normalize(vector, dim=0))
        labels.append(code)
    return torch.stack(prototypes), labels


def evaluate_embedder(model, train_items: list[CropItem], val_items: list[CropItem],
                      codes: list[str], device: str,
                      reference_dir: Path | None = None) -> dict:
    """验证：训练集原型 vs 验证裁图（按视频隔离），算 top-1 / top-5 与主要混淆。"""
    import torch

    prototypes, proto_labels = build_prototypes(model, train_items, codes, device, reference_dir)
    if not val_items or prototypes is None:
        return {"count": 0, "top1": 0.0, "top5": 0.0, "confusion": []}

    features = _embed_paths(model, [item.path for item in val_items], device)
    truth = [item.code for item in val_items]
    similarity = features @ prototypes.T

    top1 = top5 = 0
    confusion: dict[tuple[str, str], int] = {}
    for index, row in enumerate(similarity):
        order = torch.argsort(row, descending=True).tolist()
        ranked = [proto_labels[j] for j in order]
        expected = truth[index]
        if ranked and ranked[0] == expected:
            top1 += 1
        else:
            key = (expected, ranked[0] if ranked else "?")
            confusion[key] = confusion.get(key, 0) + 1
        if expected in ranked[:5]:
            top5 += 1

    total = len(truth)
    worst = sorted(confusion.items(), key=lambda item: -item[1])[:6]
    return {
        "count": total,
        "top1": round(top1 / max(1, total), 4),
        "top5": round(top5 / max(1, total), 4),
        "confusion": [f"{a}->{b}×{n}" for (a, b), n in worst],
    }
