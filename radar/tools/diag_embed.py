"""诊断 embedding 是否塌了：看各编号原型之间的余弦相似度。"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.embed import collect_crops, to_tensor  # noqa: E402

CODES = ["U7", "1", "O1", "B2", "U3", "A9"]


def load_model(ckpt: Path):
    checkpoint = torch.load(ckpt, map_location="cpu")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    model = nn.Sequential(backbone, nn.Flatten(), nn.Linear(512, checkpoint["dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main() -> int:
    model, checkpoint = load_model(ROOT / "weights" / "embedder.pt")
    items = collect_crops(ROOT / "dataset" / "crops", CODES)
    features: dict[str, list] = {}
    for item in items:
        image = cv2.imread(str(item.path))
        if image is None:
            continue
        with torch.no_grad():
            vector = torch.nn.functional.normalize(model(torch.from_numpy(to_tensor(image)).unsqueeze(0))).squeeze(0)
        features.setdefault(item.code, []).append(vector)

    print("每类样本:", {code: len(v) for code, v in features.items()})
    protos = {}
    for code, vectors in features.items():
        mean = torch.stack(vectors).mean(0)
        protos[code] = mean / mean.norm()
    for a, b in itertools.combinations(sorted(protos), 2):
        print(f"{a:>3} vs {b:<3}: cos={float(protos[a] @ protos[b]):.3f}")
    norms = [float(v.norm()) for code, values in features.items() for v in values[:5]]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
