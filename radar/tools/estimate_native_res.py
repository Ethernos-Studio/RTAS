"""估计每段录像的"真实分辨率"：把帧缩到 720p 再放回 1080p，看差多少。

如果原始画面就是 720p 放大出来的，那么"缩回去再放大"几乎无损（PSNR 很高）；
如果是真 1080p，缩了再放大会明显丢细节（PSNR 低）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    groups: dict[str, list[Path]] = {}
    for path in sorted((ROOT / "frames").glob("*.jpg")):
        video = path.stem.split("_f")[0]
        groups.setdefault(video, []).append(path)

    print(f"{'视频':<12}{'帧数':>5}  {'720p 往返 PSNR':>14}  {'540p 往返 PSNR':>14}")
    for video, frames in groups.items():
        picks = frames[:: max(1, len(frames) // 6)][:6]
        scores_720, scores_540 = [], []
        for path in picks:
            image = cv2.imread(str(path))
            if image is None:
                continue
            height, width = image.shape[:2]
            for target_w, bucket in ((1280, scores_720), (960, scores_540)):
                target_h = int(round(height * target_w / width))
                small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
                back = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
                bucket.append(psnr(image, back))
        print(f"{video:<12}{len(frames):>5}  {np.mean(scores_720):>14.1f}  {np.mean(scores_540):>14.1f}")
    print("\nPSNR 越高说明原始画面越接近那个分辨率（越高越像被放大的）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
