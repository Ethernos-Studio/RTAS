"""镜头姿态标定：把"部署预览帧"和"正常帧"的几何关系存下来，用于把裁图转正。

机制（你确认过的）：选中干员后，镜头会缓动到一个**固定**的最大偏转角，
所以每个关卡只需要标定一次——拿一张正常视角帧和一张部署预览帧，
算出它们之间的单应 H，之后部署期间的裁图乘 H⁻¹ 就回到正常视角。

为什么用帧间单应而不是"解 3D 相机姿态"：
* 部署预览只改变相机的俯仰，地面还是同一个平面 —— 平面到平面的映射就是单应，
  单应能精确复现（本质矩阵/旋转矩阵那套是给多平面/非平面准备的，这里用不上）；
* 单应不需要知道内参（焦距/FOV），也不依赖游戏里看不到的相机参数；
* 计算量是 4 对点解 8 个未知数，离线一次，运行时只是一次 3×3 乘法。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2] if __file__.endswith("calib.py") else Path.cwd()
sys.stdout.reconfigure(encoding="utf-8")


def battlefield_mask(shape: tuple[int, int]) -> np.ndarray:
    """只用中间战场做配准，避开左侧面板和底部干员栏。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height * 0.10):int(height * 0.86), int(width * 0.22):int(width * 0.98)] = 255
    return mask


@dataclass
class StageCalibration:
    """一个关卡从"部署姿态"回到"正常姿态"的单应。"""

    stage: str
    matrix: list[list[float]]         # 3x3，把部署帧的像素映射回正常帧
    inliers: int = 0
    angle_deg: float = 0.0
    scale: float = 1.0

    def to_dict(self) -> dict:
        return {"stage": self.stage, "matrix": self.matrix, "inliers": self.inliers,
                "angle_deg": round(self.angle_deg, 3), "scale": round(self.scale, 4)}


def estimate(normal_path: Path, deploy_path: Path, *, stage: str) -> StageCalibration:
    """用 ORB + RANSAC 求"部署帧 -> 正常帧"的单应。"""
    normal = cv2.imread(str(normal_path))
    deploy = cv2.imread(str(deploy_path))
    if normal is None or deploy is None:
        raise FileNotFoundError("读不到标定帧")

    gray_normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
    gray_deploy = cv2.cvtColor(deploy, cv2.COLOR_BGR2GRAY)
    mask = battlefield_mask(gray_normal.shape)

    orb = cv2.ORB_create(nfeatures=6000, fastThreshold=10)
    kp_normal, des_normal = orb.detectAndCompute(gray_normal, mask)
    kp_deploy, des_deploy = orb.detectAndCompute(gray_deploy, mask)
    if des_normal is None or des_deploy is None:
        raise RuntimeError("特征太少，换一组标定帧")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_deploy, des_normal), key=lambda m: m.distance)[:1500]
    source = np.float32([kp_deploy[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    target = np.float32([kp_normal[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if matrix is None:
        raise RuntimeError("单应求解失败")

    linear = matrix[:2, :2]
    horizontal = linear @ np.array([1.0, 0.0])
    vertical = linear @ np.array([0.0, 1.0])
    angle = float(np.degrees(np.arctan2(vertical[1], vertical[0]))) - 90.0
    scale = float(np.sqrt(abs(np.linalg.det(linear))))
    return StageCalibration(
        stage=stage,
        matrix=matrix.tolist(),
        inliers=int(inliers.sum()) if inliers is not None else 0,
        angle_deg=angle,
        scale=scale,
    )


def load_all(path: Path) -> dict[str, np.ndarray]:
    """读标定文件 -> {关卡: 3x3 矩阵}。"""
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["stage"]: np.array(item["matrix"], dtype=np.float64)
            for item in data.get("stages", [])}


def save_all(path: Path, calibrations: list[StageCalibration]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stages": [item.to_dict() for item in calibrations]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rectify_frame(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """整帧从"部署姿态"转到"正常姿态"（输出同尺寸）。"""
    height, width = image.shape[:2]
    return cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def warp_box(box: list[float], matrix: np.ndarray) -> list[float]:
    """把框的四个角用同一个单应变换后取外接框。"""
    x, y, w, h = (float(v) for v in box)
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    x1, y1 = mapped.min(axis=0)
    x2, y2 = mapped.max(axis=0)
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def rectify_crop(image: np.ndarray, box: list[float], matrix: np.ndarray) -> np.ndarray:
    """把裁图从"部署姿态"转回"正常姿态"。

    box 是原图坐标；先在裁图坐标系里做同一变换（先平移再乘 H⁻¹），再取整。
    """
    inverse = np.linalg.inv(matrix)
    x, y, w, h = (float(v) for v in box)
    crop = image[int(y):int(y + h), int(x):int(x + w)]
    if crop.size == 0:
        return crop
    translate = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]], dtype=np.float64)
    warp = inverse @ translate
    return cv2.warpPerspective(crop, warp, (crop.shape[1], crop.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
