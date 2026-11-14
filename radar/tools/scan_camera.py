"""扫一遍所有录像：镜头到底有没有转/缩放（相对每段视频的第一帧）。

如果旋转一直≈0、缩放≈1，说明"镜头偏转"在现有素材里可以忽略，
letterbox 的尺度不变性就够用；如果某些帧有明显转角，再考虑转正裁图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.frames import load_manifest  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
from measure_tilt import battlefield_mask  # noqa: E402


def relative_pose(reference_gray, reference_kp, reference_des, target: np.ndarray, mask):
    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    if descriptors is None or reference_des is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(reference_des, descriptors), key=lambda m: m.distance)[:800]
    if len(matches) < 12:
        return None
    source = np.float32([reference_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    destination = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    matrix, inliers = cv2.estimateAffinePartial2D(source, destination, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
    if matrix is None:
        return None
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return angle, scale


def main() -> int:
    manifest = load_manifest(ROOT / "frames" / "manifest.jsonl")
    by_video: dict[str, list[str]] = {}
    for frame, entry in manifest.items():
        by_video.setdefault(entry.video, []).append(frame)

    orb = cv2.ORB_create(nfeatures=3000, fastThreshold=12)
    for video, frames in sorted(by_video.items()):
        frames.sort()
        reference = cv2.imread(str(ROOT / "frames" / frames[0]))
        if reference is None:
            continue
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        mask = battlefield_mask(gray.shape)
        keypoints, descriptors = orb.detectAndCompute(gray, mask)

        angles, scales = [], []
        for name in frames[1:]:
            target = cv2.imread(str(ROOT / "frames" / name))
            if target is None:
                continue
            result = relative_pose(gray, keypoints, descriptors, target, mask)
            if result:
                angles.append(result[0])
                scales.append(result[1])
        if not angles:
            print(f"{video}: 配准失败")
            continue
        angles = np.array(angles)
        scales = np.array(scales)
        print(f"{video}: {len(angles)} 帧  "
              f"旋转 {angles.min():+.2f}°~{angles.max():+.2f}°（|中位| {abs(np.median(angles)):.2f}°）  "
              f"缩放 {scales.min():.3f}~{scales.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
