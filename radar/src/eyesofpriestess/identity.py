"""敌人身份：embedding 图库匹配（不训分类器）。

设计依据（HANDOFF §3.2 / plan §3.2）：细粒度兵种识别不走分类器，走**图库最近邻**——
新敌人上线只需要往图库里加几张参考图，零训练接入。

数据来源就是 ``radar build`` 顺手产出的 ``dataset/crops/<图鉴编号>/``：
每个检测框裁出来的全身图，按编号分目录。这里做的事：

1. 用一个通用视觉编码器（torchvision 预训练）把裁剪图变成向量；
2. 每个编号取所有样本的**单位化均值**当原型（对遮挡/受击帧更稳）；
3. 推理时把检测框的向量与所有原型算余弦相似度，取最高的编号；
4. top1 与 top2 太接近 → 报 ``ambiguous``，交给人工/专家队列（plan §3.2）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

#: ImageNet 归一化（torchvision 预训练模型的输入约定）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224

#: 相似度低于它就算"认不出"
DEFAULT_MIN_SCORE = 0.55
#: top1 领先 top2 不足这个差值就报 ambiguous
DEFAULT_MIN_MARGIN = 0.03


@dataclass
class Match:
    """一次身份匹配的结果。"""

    code: str                     # 图鉴编号；认不出时是 ""
    score: float = 0.0            # 与原型余弦相似度
    runner_up: str = ""           # 次优编号
    runner_up_score: float = 0.0
    ambiguous: bool = False       # top1/top2 太接近

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "score": round(self.score, 3),
            "runner_up": self.runner_up,
            "runner_up_score": round(self.runner_up_score, 3),
            "ambiguous": self.ambiguous,
        }


class Encoder:
    """torchvision 预训练骨干，输出 L2 单位化的特征向量。"""

    def __init__(self, arch: str = "resnet18", device: str | None = None) -> None:
        self.arch = arch
        self.device = device
        self._model = None
        self._torch = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import models
        except ImportError as exc:  # pragma: no cover - 环境问题
            raise RuntimeError(
                "身份匹配需要 torchvision：`uv sync --extra train`"
            ) from exc

        self._torch = torch
        weights = "DEFAULT"
        if self.arch == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights[weights])
            model.fc = torch.nn.Identity()          # 取 512 维池化特征
        elif self.arch == "convnext_tiny":
            model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights[weights])
            model.classifier[2] = torch.nn.Identity()
        else:
            raise ValueError(f"不支持的骨干: {self.arch}")

        if self.device:
            model = model.to(self.device)
        model.eval()
        self._model = model

    @property
    def device_name(self) -> str:
        self._ensure()
        return str(next(self._model.parameters()).device)

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """一批 BGR 裁剪图 -> (n, dim) 单位向量。"""
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure()
        torch = self._torch
        batch = np.stack([_prepare(crop) for crop in crops]).astype(np.float32)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
        if self.device:
            tensor = tensor.to(self.device)
        with torch.no_grad():
            features = self._model(tensor)
        array = features.detach().cpu().numpy().reshape(len(crops), -1).astype(np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-6)


def _prepare(crop: np.ndarray) -> np.ndarray:
    """裁剪图 -> 224x224 RGB 归一化数组（BGR 输入）。"""
    image = crop
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


@dataclass
class Gallery:
    """每个图鉴编号若干原型向量（覆盖动作/差分/透视大小差异）。

    单个均值原型不够用：同一编号有行走/攻击/受击等动作，还有差分立绘，
    远景又比近景小一圈（只缩放、不透视变形）。所以每类先做 k-means，
    保留 k 个原型，匹配时取"最像的那一个原型"。
    """

    codes: list[str] = field(default_factory=list)          # 与 prototypes 行一一对应
    prototypes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    samples: dict[str, int] = field(default_factory=dict)
    _index: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._index:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        index: dict[str, list[int]] = {}
        for row, code in enumerate(self.codes):
            index.setdefault(code, []).append(row)
        self._index = index

    @property
    def unique_codes(self) -> list[str]:
        return list(self._index)

    # -- 构建 -------------------------------------------------------------
    @classmethod
    def from_crops(
        cls,
        crops_dir: Path,
        encoder: Encoder,
        *,
        max_per_code: int = 240,
        min_crops: int = 5,
        prototypes_per_code: int = 4,
        keep_codes: Iterable[str] | None = None,
        diversity: float = 0.96,
        reference_dir: Path | None = None,
    ) -> "Gallery":
        """从 ``dataset/crops/<编号>/*.png`` 建多原型图库。

        ``keep_codes`` 只保留这些编号（正常只放 enemy 类，别把 tile/ally 混进身份图库）。
        ``diversity``：同一编号里两两相似度超过它的裁图只留一张——战斗裁图里大量是
        "同一只怪同一姿势"的重复帧，全喂进去只会让原型塌成一团。
        """
        crops_dir = Path(crops_dir)
        allowed = {str(code) for code in keep_codes} if keep_codes is not None else None
        codes: list[str] = []
        prototypes: list[np.ndarray] = []
        samples: dict[str, int] = {}

        for folder in sorted(path for path in crops_dir.iterdir() if path.is_dir()):
            if allowed is not None and folder.name not in allowed:
                continue
            files = sorted(folder.glob("*.png"))[:max_per_code]
            if len(files) < min_crops:
                continue
            images = []
            for file in files:
                image = cv2.imread(str(file))
                if image is not None and image.size:
                    images.append(image)
            if len(images) < min_crops:
                continue

            features = encoder.embed(images)
            features, kept = _diversify(features, images, diversity)
            centres = _cluster(features, prototypes_per_code)
            for centre in centres:
                unit = centre / max(float(np.linalg.norm(centre)), 1e-6)
                codes.append(folder.name)
                prototypes.append(unit.astype(np.float32))
            samples[folder.name] = kept

        # 官方渲染图（tools/render_reference.py 从游戏 Spine 渲出来的干净全身图）
        # 单独作为原型加进去：它没有遮挡/受击变色，是最稳定的锚点
        if reference_dir is not None:
            reference_dir = Path(reference_dir)
            if reference_dir.exists():
                for image_path in sorted(reference_dir.glob("*.png")):
                    code = image_path.stem
                    if allowed is not None and code not in allowed:
                        continue
                    image = cv2.imread(str(image_path))
                    if image is None or not image.size:
                        continue
                    unit = encoder.embed([image])[0]
                    unit /= max(float(np.linalg.norm(unit)), 1e-6)
                    codes.append(code)
                    prototypes.append(unit.astype(np.float32))
                    samples[code] = samples.get(code, 0) + 1

        if not prototypes:
            raise RuntimeError(f"{crops_dir} 里没有足够的裁剪图（每类至少 {min_crops} 张）")
        return cls(codes=codes, prototypes=np.stack(prototypes), samples=samples)

    # -- 匹配 -------------------------------------------------------------
    def identify(
        self,
        embeddings: np.ndarray,
        *,
        min_score: float = DEFAULT_MIN_SCORE,
        min_margin: float = DEFAULT_MIN_MARGIN,
    ) -> list[Match]:
        """一批特征向量 -> 匹配结果（每个编号取最像的那个原型）。"""
        if not self.codes or embeddings.size == 0:
            return [Match(code="") for _ in range(len(embeddings))]

        raw = embeddings @ self.prototypes.T              # (n, P) 与每个原型的相似度
        unique = self.unique_codes
        per_code = np.stack([raw[:, self._index[code]].max(axis=1) for code in unique], axis=1)

        results: list[Match] = []
        for row in per_code:
            order = np.argsort(row)[::-1]
            best = int(order[0])
            second = int(order[1]) if len(order) > 1 else best
            match = Match(
                code=unique[best],
                score=float(row[best]),
                runner_up=unique[second] if len(unique) > 1 else "",
                runner_up_score=float(row[second]) if len(unique) > 1 else 0.0,
            )
            match.ambiguous = match.margin < min_margin
            if match.score < min_score:
                match.code = ""
            results.append(match)
        return results

    # -- 存取 -------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            codes=np.array(self.codes),
            prototypes=self.prototypes,
            samples=np.array(json.dumps(self.samples, ensure_ascii=False)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到图库: {path}（先跑 `radar gallery-build`）")
        data = np.load(path, allow_pickle=False)
        return cls(
            codes=[str(code) for code in data["codes"]],
            prototypes=data["prototypes"].astype(np.float32),
            samples=json.loads(str(data["samples"])),
        )

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for code in self.codes:
            counts[code] = counts.get(code, 0) + 1
        return "、".join(
            f"{code}×{self.samples.get(code, 0)}图/{counts[code]}原型" for code in self.unique_codes
        )


def _diversify(
    features: np.ndarray,
    images: Sequence[np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, int]:
    """按"清晰度 × 面积"排序，再贪心丢掉和已选样本太像的重复裁图。

    战斗裁图里同一只怪同一姿势会重复几十次；不筛的话 k-means 出来的原型全挤在一起。
    """
    if len(features) <= 2:
        return features, len(features)

    scores = []
    for index, image in enumerate(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        area = float(image.shape[0] * image.shape[1])
        scores.append((sharpness * (area ** 0.5), index))

    kept: list[int] = []
    for _score, index in sorted(scores, reverse=True):
        vector = features[index]
        if any(float(vector @ features[other]) > threshold for other in kept):
            continue
        kept.append(index)
        if len(kept) >= 120:
            break

    if not kept:
        return features, 0
    return features[kept], len(kept)


def _cluster(features: np.ndarray, k: int) -> np.ndarray:
    """在单位球面上做 k-means（cv2.kmeans），返回 k 个中心；样本少时退化成全部样本。"""
    n = len(features)
    k = max(1, min(k, n // 3 or 1, n))
    if k <= 1 or n <= k + 2:
        return features
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    data = features.astype(np.float32)
    _compactness, labels, centres = cv2.kmeans(
        data, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS
    )
    # 空簇丢掉
    used = set(int(label[0]) for label in labels)
    return np.stack([centres[index] for index in sorted(used)])


def crop_box(image: np.ndarray, box: Sequence[int], *, pad: float = 0.08) -> np.ndarray:
    """按检测框裁图（略微外扩，给身份匹配留点上下文）。"""
    height, width = image.shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    pad_x, pad_y = int(round(w * pad)), int(round(h * pad))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]
