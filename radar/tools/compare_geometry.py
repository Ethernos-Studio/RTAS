"""比较参考帧与部署预览帧的几何关系：单应矩阵 + 水平/垂直尺度 + 旋转。

部署时镜头可能不是"平面内旋转"，而是俯仰（pitch）——那会表现为
垂直方向被压缩，而不是角度变化。这里把两种都量出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(ROOT / "tools"))
from measure_tilt import battlefield_mask  # noqa: E402

PAIRS = [
    ("v000_1-7_f000979.jpg", "v000_1-7_f001069.jpg"),
    ("v000_1-7_f000979.jpg", "v000_1-7_f001089.jpg"),
    ("v000_1-7_f001069.jpg", "v000_1-7_f001089.jpg"),
]


def analyse(reference_path: Path, target_path: Path) -> None:
    reference = cv2.imread(str(reference_path))
    target = cv2.imread(str(target_path))
    if reference is None or target is None:
        print("读不到图")
        return
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_tgt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    mask = battlefield_mask(gray_ref.shape)

    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, mask)
    kp_tgt, des_tgt = orb.detectAndCompute(gray_tgt, mask)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_ref, des_tgt), key=lambda m: m.distance)[:1200]
    source = np.float32([kp_ref[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destination = np.float32([kp_tgt[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    homography, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 2.5)
    if homography is None:
        print(f"{reference_path.name} -> {target_path.name}: 单应失败")
        return
    linear = homography[:2, :2]
    # 水平方向 vs 垂直方向的局部尺度（在图像中心附近取单位向量）
    horizontal = linear @ np.array([1.0, 0.0])
    vertical = linear @ np.array([0.0, 1.0])
    angle_h = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))
    angle_v = float(np.degrees(np.arctan2(vertical[1], vertical[0]))) - 90.0
    print(f"{reference_path.name[-9:]} -> {target_path.name[-9:]}: "
          f"水平缩放 {np.linalg.norm(horizontal):.4f} 垂直缩放 {np.linalg.norm(vertical):.4f} | "
          f"旋转(水平线) {angle_h:+.2f}° (垂直线) {angle_v:+.2f}° | 内点 {int(inliers.sum())}")


def main() -> int:
    for reference, target in PAIRS:
        analyse(ROOT / "frames" / reference, ROOT / "frames" / target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
