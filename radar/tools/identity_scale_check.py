"""身份匹配对分辨率的敏感度（用 clicks 里的真实编号当 GT）。

1-8 带标注帧：原图 vs 缩到 1024x576（直播窗口实际大小）vs 1280x720。
用户实战里 1-8 的源石虫被认成 U7，怀疑是"直播画面只有 1024x576，敌人只剩 20 像素"。
"""

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
    clicks = load_clicks(ROOT / "clicks")

    marks = []
    for frame, items in clicks.items():
        if not frame.startswith("v001_1-8"):
            continue
        points = [(click.cls, click.x, click.y) for click in items if not click.is_ignore and click.cls]
        if points:
            marks.append((frame, points))
    marks = marks[:30]
    print(f"用 {len(marks)} 张 1-8 带标注帧（GT 来自 clicks）\n")

    for name, scale in (("原图 1920", 1.0), ("缩到 1280x720", 1280 / 1920), ("缩到 1024x576", 1024 / 1920)):
        correct = total = 0
        scores = []
        confusion: dict[str, int] = {}
        for frame, points in marks:
            image = cv2.imread(str(ROOT / "frames" / frame))
            if image is None:
                continue
            if scale != 1.0:
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            result = model.predict(image, imgsz=1280, conf=0.45, iou=0.8, verbose=False)[0]
            boxes = [[float(v) for v in box.xyxy[0].tolist()]
                     for box in (result.boxes or []) if int(box.cls[0]) == 0]
            boxes = MergeSuppressor()(boxes)
            matches = gallery.identify(encoder.embed(
                [crop_box(image, [b[0], b[1], b[2] - b[0], b[3] - b[1]], pad=0.0) for b in boxes])) if boxes else []
            for code, px, py in points:
                gx, gy = px * scale, py * scale
                gt = [gx - 12 * scale, gy - 24 * scale, gx + 12 * scale, gy + 24 * scale]
                best, best_iou = -1, 0.25
                for index, box in enumerate(boxes):
                    value = iou(gt, box)
                    if value > best_iou:
                        best, best_iou = index, value
                total += 1
                if best < 0:
                    continue
                scores.append(matches[best].score)
                if matches[best].code == code:
                    correct += 1
                else:
                    key = f"{code}->{matches[best].code}"
                    confusion[key] = confusion.get(key, 0) + 1
        top = sorted(confusion.items(), key=lambda kv: -kv[1])[:4]
        print(f"{name:<14} top-1 {correct/max(1,total):6.1%} ({correct}/{total}) | "
              f"平均相似度 {sum(scores)/max(1,len(scores)):.2f} | 主要混淆 {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
