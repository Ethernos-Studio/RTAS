"""量化"直播窗口分辨率 vs 训练分辨率"对检测的影响。

实战时抓到的画面是 1024x576（客户区），训练用的是 1920x1080，
同样的敌人在直播里只有一半的像素 -> 检测/身份都会掉。
这里用带标注的帧做对照：原图 vs 缩到 1024x576 vs 缩完再放大回 1920x1080。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    data = ROOT / "dataset"
    frames = sorted((data / "images/val").glob("*.jpg"))[:12]
    print(f"用 {len(frames)} 张带标注的验证帧做对照\n")

    modes = {
        "原图 1920x1080": lambda im: im,
        "缩到 1024x576": lambda im: cv2.resize(im, (1024, 576), interpolation=cv2.INTER_AREA),
        "缩到 1024x576 再放大回 1920": lambda im: cv2.resize(
            cv2.resize(im, (1024, 576), interpolation=cv2.INTER_AREA), (1920, 1080),
            interpolation=cv2.INTER_LINEAR),
    }

    for name, transform in modes.items():
        tp = fp = fn = 0
        confs = []
        for image_path in frames:
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            gt = []
            for line in (data / "labels/val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = (float(v) for v in line.split())
                if int(cls) != 0:
                    continue
                gt.append([(cx - w / 2) * width, (cy - h / 2) * height,
                           (cx + w / 2) * width, (cy + h / 2) * height])
            if not gt:
                continue

            processed = transform(image)
            scale = image.shape[1] / processed.shape[1]
            result = model.predict(processed, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
            preds = []
            for box in result.boxes or []:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = (float(v) * scale for v in box.xyxy[0].tolist())
                preds.append(([x1, y1, x2, y2], float(box.conf[0])))

            matched = set()
            for box, conf in preds:
                hit = next((index for index, g in enumerate(gt)
                            if index not in matched and iou(box, g) >= 0.5), -1)
                if hit >= 0:
                    matched.add(hit)
                    tp += 1
                    confs.append(conf)
                else:
                    fp += 1
            fn += len(gt) - len(matched)

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        print(f"{name:<28} TP {tp:3d} FP {fp:3d} FN {fn:3d} | P {precision:.2f} R {recall:.2f} "
              f"| 命中框平均置信度 {mean_conf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
