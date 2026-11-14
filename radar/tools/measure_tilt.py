"""用"和参考帧做配准"来估镜头偏角（比 Hough 找格线稳）。

Hough 那版在部署预览帧上会被白色攻击范围菱形带偏（1069 量到 -4°，1089 量到 +1.5°，
但这两帧其实是同一个机位）。改成：拿已知正常视角的帧当参考，ORB 特征匹配 +
RANSAC 估相似变换（旋转+缩放+平移），直接读出旋转角。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

REFERENCE = "v000_1-7_f000979.jpg"
TARGETS = ["v000_1-7_f001069.jpg", "v000_1-7_f001089.jpg"]


def battlefield_mask(shape: tuple[int, int]) -> np.ndarray:
    """只留中间战场区域，避开左侧面板和底部干员栏。"""
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height * 0.10):int(height * 0.86), int(width * 0.22):int(width * 0.98)] = 255
    return mask


def estimate_rotation(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, int]:
    """返回 (旋转角度, 缩放, 内点数)。"""
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=12)
    mask = battlefield_mask(gray_ref.shape)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, mask)
    keypoints_tgt, descriptors_tgt = orb.detectAndCompute(gray_tgt, mask)
    if descriptors_ref is None or descriptors_tgt is None:
        return 0.0, 1.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_ref, descriptors_tgt), key=lambda m: m.distance)
    if len(matches) < 10:
        return 0.0, 1.0, 0

    source = np.float32([keypoints_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destination = np.float32([keypoints_tgt[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
    if matrix is None:
        return 0.0, 1.0, 0
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return angle, scale, inlier_count


def main() -> int:
    reference = cv2.imread(str(ROOT / "frames" / REFERENCE))
    if reference is None:
        print("参考帧读不到")
        return 1
    print(f"参考: {REFERENCE} (正常视角 = 0°)")
    for name in (sys.argv[1:] or TARGETS):
        target = cv2.imread(str(ROOT / "frames" / name))
        if target is None:
            print(f"{name}: 读不到")
            continue
        angle, scale, inliers = estimate_rotation(reference, target)
        print(f"{name}: 旋转 {angle:+.2f}° | 缩放 ×{scale:.3f} | 内点 {inliers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
