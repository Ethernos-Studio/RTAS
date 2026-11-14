"""点击 -> 框：用 SAM（MobileSAM）把点升格成掩码/外接框。

设计要点：

* 点只用来"指认目标"，框是 SAM 的派生物——人永远不画框（HANDOFF §2.1）；
* 大图上 SAM 很慢：支持只在战场 ``region`` 里跑、或先 ``downscale`` 再映射回原坐标；
* 一个点对不上一个掩码时（极小目标 mask 失败）返回 ``None``，由上层记进 report，
  绝不拿错位的掩码去配类别。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

#: 掩码判为前景的阈值
MASK_THRESHOLD = 0.5


@dataclass
class Box:
    """原图像素坐标下的外接框（含面积，用于过滤噪声掩码）。"""

    x1: int
    y1: int
    x2: int
    y2: int
    area: int = 0

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def clamp(self, width: int, height: int) -> "Box":
        return Box(
            x1=max(0, min(self.x1, width - 1)),
            y1=max(0, min(self.y1, height - 1)),
            x2=max(0, min(self.x2, width)),
            y2=max(0, min(self.y2, height)),
            area=self.area,
        )


class SamBoxer:
    """ultralytics SAM 的薄封装；权重默认 ``mobile_sam.pt``（首次运行自动下载）。"""

    def __init__(self, weights: str | Path = "mobile_sam.pt", device: str | None = None) -> None:
        self.weights = str(weights)
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import SAM  # noqa: PLC0415 - 延迟导入，没装 ultralytics 也能用其它子命令
            except ImportError as exc:  # pragma: no cover - 环境问题
                raise RuntimeError(
                    "SAM 需要 ultralytics：`uv sync --extra train` 或 `pip install ultralytics`"
                ) from exc
            self._model = SAM(self.weights)
            if self.device:
                self._model.to(self.device)
        return self._model

    # -- 点 -> 框 ---------------------------------------------------------
    def boxes(
        self,
        image: np.ndarray,
        points: Sequence[tuple[float, float]],
        *,
        region: tuple[int, int, int, int] | None = None,
        downscale: float = 1.0,
    ) -> list[Box | None]:
        """给每个点出外接框；失败的槽位是 ``None``（顺序与输入严格对齐）。

        Args:
            image: BGR 原图
            points: 原图像素坐标的点
            region: 只在这个矩形（x0, y0, x1, y1，原图像素）里跑 SAM
            downscale: 在 region 内再降采样（0.5 = 边长减半），结果映射回原坐标
        """
        if not points:
            return []
        height, width = image.shape[:2]
        x_off, y_off = 0, 0
        work = image
        if region:
            rx0, ry0, rx1, ry1 = region
            rx0, ry0 = max(0, rx0), max(0, ry0)
            rx1, ry1 = min(width, rx1), min(height, ry1)
            if rx1 - rx0 < 8 or ry1 - ry0 < 8:
                raise ValueError(f"region 太小: {region}")
            work = image[ry0:ry1, rx0:rx1]
            x_off, y_off = rx0, ry0

        scale = 1.0
        if downscale and downscale != 1.0:
            scale = float(downscale)
            work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        local_points = [((px - x_off) * scale, (py - y_off) * scale) for px, py in points]
        masks = self._predict_masks(work, local_points)

        boxes: list[Box | None] = []
        for mask in masks:
            boxes.append(self._mask_to_box(mask, scale, x_off, y_off, width, height))
        return boxes

    def _predict_masks(self, image: np.ndarray, points: Sequence[tuple[float, float]]) -> list[np.ndarray | None]:
        """一次喂所有点；数量和输入对不上时退回逐点推理，保证顺序对齐。"""
        model = self.model
        results = model.predict(
            image,
            points=[list(point) for point in points],
            labels=[1] * len(points),
            verbose=False,
        )
        masks = self._extract(results)
        if len(masks) == len(points):
            return masks
        if len(masks) > len(points):
            return masks[: len(points)]

        # SAM 对极小/低对比目标可能少给 mask：逐点重跑，缺的补 None
        out: list[np.ndarray | None] = []
        for point in points:
            single = model.predict(
                image, points=[list(point)], labels=[1], verbose=False
            )
            found = self._extract(single)
            out.append(found[0] if found else None)
        return out

    @staticmethod
    def _extract(results: Any) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        if not results:
            return masks
        result = results[0]
        if getattr(result, "masks", None) is None:
            return masks
        data = result.masks.data
        array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        for mask in array:
            masks.append(np.asarray(mask))
        return masks

    @staticmethod
    def _mask_to_box(
        mask: np.ndarray | None,
        scale: float,
        x_off: int,
        y_off: int,
        width: int,
        height: int,
    ) -> Box | None:
        if mask is None or mask.size == 0:
            return None
        binary = mask > MASK_THRESHOLD
        rows = np.flatnonzero(binary.any(axis=1))
        cols = np.flatnonzero(binary.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return None
        inverse = 1.0 / scale if scale else 1.0
        box = Box(
            x1=int(round(cols[0] * inverse)) + x_off,
            y1=int(round(rows[0] * inverse)) + y_off,
            x2=int(round((cols[-1] + 1) * inverse)) + x_off,
            y2=int(round((rows[-1] + 1) * inverse)) + y_off,
            area=int(binary.sum()),
        )
        return box.clamp(width, height)
