"""同一划分下对比：ImageNet 特征 vs 训练后的 embedding（按视频隔离）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.embed import collect_crops, split_by_video, to_tensor  # noqa: E402
from eyesofpriestess.identity import Encoder, Gallery  # noqa: E402


def build_trained(ckpt: Path):
    checkpoint = torch.load(ckpt, map_location="cpu")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(512, checkpoint["dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["classes"]


def embed_paths(fn, paths: list[Path]):
    outputs = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        outputs.append(fn(image))
    return torch.stack(outputs)


def accuracy(features: dict[str, list], val_items, codes) -> tuple[float, int, list[str]]:
    """用"每类训练特征均值"当原型，算验证集 top-1。"""
    protos, names = [], []
    for code in codes:
        vectors = features.get(code)
        if not vectors:
            continue
        mean = torch.stack(vectors).mean(0)
        protos.append(mean / mean.norm())
        names.append(code)
    if not protos:
        return 0.0, 0, []
    matrix = torch.stack(protos)
    hits = total = 0
    confusion: dict[tuple[str, str], int] = {}
    for item in val_items:
        vector = feat_of(item)
        if vector is None:
            continue
        similarity = matrix @ vector
        best = names[int(torch.argmax(similarity))]
        total += 1
        if best == item.code:
            hits += 1
        else:
            confusion[(item.code, best)] = confusion.get((item.code, best), 0) + 1
    worst = sorted(confusion.items(), key=lambda kv: -kv[1])[:5]
    return hits / max(1, total), total, [f"{a}->{b}×{n}" for (a, b), n in worst]


def main() -> int:
    val_video = sys.argv[1] if len(sys.argv) > 1 else "v000_1-10"
    codes = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
             "U1", "U3", "U7", "U9", "U17"]
    items = collect_crops(ROOT / "dataset" / "crops", codes)
    train_items, val_items = split_by_video(items, {val_video})
    train_codes = sorted({item.code for item in train_items})
    val_codes = sorted({item.code for item in val_items})
    print(f"训练 {len(train_items)} 张（{len(train_codes)} 类）/ 验证 {len(val_items)} 张（{val_codes}）")
    shared = [code for code in val_codes if code in train_codes]
    print("两边都有的编号:", shared)
    if not shared:
        print("没有共享编号，无法比较")
        return 1

    # 基线：ImageNet resnet18
    baseline_encoder = Encoder("resnet18")
    global feat_of
    base_feats: dict[str, list] = {}
    by_code: dict[str, list[Path]] = {}
    for item in train_items:
        by_code.setdefault(item.code, []).append(item.path)
    for code, paths in by_code.items():
        vectors = baseline_encoder.embed([cv2.imread(str(p)) for p in paths if cv2.imread(str(p)) is not None])
        base_feats[code] = [torch.from_numpy(v) for v in vectors]

    val_feats_base = {}
    for item in val_items:
        image = cv2.imread(str(item.path))
        if image is None:
            continue
        val_feats_base[item.path] = torch.from_numpy(baseline_encoder.embed([image])[0])

    def feat_of_base(item):
        return val_feats_base.get(item.path)

    global feat_of
    feat_of = feat_of_base
    base_acc, base_total, base_conf = accuracy(base_feats, [i for i in val_items if i.code in shared], shared)
    print(f"基线(ImageNet): top-1 {base_acc:.1%} / {base_total}  {base_conf}")

    # 训练后的 embedding
    model, _classes = build_trained(ROOT / "weights" / "embedder.pt")

    def trained_fn(image):
        with torch.no_grad():
            return torch.nn.functional.normalize(model(torch.from_numpy(to_tensor(image)).unsqueeze(0))).squeeze(0)

    trained_feats: dict[str, list] = {}
    for code, paths in by_code.items():
        vectors = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is not None:
                vectors.append(trained_fn(image))
        trained_feats[code] = vectors

    val_feats_trained = {}
    for item in val_items:
        image = cv2.imread(str(item.path))
        if image is not None:
            val_feats_trained[item.path] = trained_fn(image)

    def feat_of_trained(item):
        return val_feats_trained.get(item.path)

    feat_of = feat_of_trained
    train_acc, train_total, train_conf = accuracy(
        trained_feats, [i for i in val_items if i.code in shared], shared
    )
    print(f"训练后: top-1 {train_acc:.1%} / {train_total}  {train_conf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
