"""多尺度裁图融合能否救回身份识别（修正版：按框维度保留 n×256）。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.identity import Encoder, Gallery, crop_box  # noqa: E402
from eyesofpriestess.postprocess import MergeSuppressor  # noqa: E402
from eyesofpriestess.schema import load_clicks  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def scaled(box, factor):
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    return [cx - w * factor / 2, cy - h * factor / 2, w * factor, h * factor]


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    encoder = Encoder("resnet18", device="cuda", checkpoint=ROOT / "weights/embedder.pt")
    gallery = Gallery.load(ROOT / "weights/gallery.npz")
    clicks = load_clicks(ROOT / "clicks")

    for video in ("v001_1-8", "v002_1-9"):
        names = sorted(f for f in clicks if f.startswith(video))[:24]
        single_correct = multi_correct = total = 0
        for frame in names:
            image = cv2.imread(str(ROOT / "frames" / frame))
            if image is None:
                continue
            points = [(c.cls, c.x, c.y) for c in clicks[frame] if not c.is_ignore and c.cls]
            if not points:
                continue
            result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
            boxes = [[float(v) for v in box.xyxy[0].tolist()]
                     for box in (result.boxes or []) if int(box.cls[0]) == 0]
            boxes = MergeSuppressor()(boxes)
            if not boxes:
                total += len(points)
                continue

            fields = [b[0:2] + [b[2] - b[0], b[3] - b[1]] for b in boxes]
            embeddings = []
            for factor in (0.9, 1.0, 1.15):
                crops = [crop_box(image, scaled(box, factor), pad=0.0) for box in fields]
                embeddings.append(encoder.embed(crops))
            fused = np.mean(embeddings, axis=0)
            single = gallery.identify(embeddings[1])
            multi = gallery.identify(fused)

            for code, x, y in points:
                half_w, half_h = max(8.0, 1920 * 0.015), max(8.0, 1080 * 0.03)
                gt = [x - half_w, y - half_h, x + half_w, y + half_h]
                best, best_iou = -1, 0.3
                for box_index, box in enumerate(boxes):
                    value = iou(gt, box)
                    if value > best_iou:
                        best, best_iou = box_index, value
                total += 1
                if best < 0:
                    continue
                single_correct += 1 if single[best].code == code else 0
                multi_correct += 1 if multi[best].code == code else 0
        print(f"{video}: 单尺度 {single_correct/max(1,total):.1%} | 多尺度融合 "
              f"{multi_correct/max(1,total):.1%} (n={total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
