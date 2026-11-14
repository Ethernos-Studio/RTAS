"""对照：用**已发布的** gallery.npz 复现 identity-check 的 98.7%，确认刚才实验里的重建是否有 bug。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

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


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    encoder = Encoder("resnet18", device="cuda", checkpoint=ROOT / "weights/embedder.pt")
    gallery = Gallery.load(ROOT / "weights/gallery.npz")
    print("图库:", gallery.prototypes.shape, "编号数", len(gallery.unique_codes))
    clicks = load_clicks(ROOT / "clicks")

    frames = []
    for frame, items in clicks.items():
        points = [(c.cls, c.x, c.y) for c in items if not c.is_ignore and c.cls]
        if points:
            frames.append((frame, points))
    frames = frames[:40]

    correct = total = 0
    confusion: dict[str, int] = {}
    for frame, points in frames:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(boxes)
        matches = gallery.identify(encoder.embed(
            [crop_box(image, [b[0], b[1], b[2] - b[0], b[3] - b[1]], pad=0.0) for b in boxes])) if boxes else []
        for code, x, y in points:
            half_w, half_h = max(8.0, 1920 * 0.015), max(8.0, 1080 * 0.03)
            gt = [x - half_w, y - half_h, x + half_w, y + half_h]
            best, best_iou = -1, 0.3
            for index, box in enumerate(boxes):
                value = iou(gt, box)
                if value > best_iou:
                    best, best_iou = index, value
            total += 1
            if best >= 0 and matches[best].code == code:
                correct += 1
            elif best >= 0:
                key = f"{code}->{matches[best].code}"
                confusion[key] = confusion.get(key, 0) + 1
    print(f"命中率 {correct}/{total} = {correct/max(1,total):.1%}")
    print("主要混淆:", sorted(confusion.items(), key=lambda kv: -kv[1])[:6])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
