"""关键实验：把候选集缩小到"本关会出现的编号"，外观识别还准不准？

逻辑：sim 的出怪表能告诉我们"这一段时间内、这条路上可能出现哪几种怪"。
如果外观只在 2~7 个候选里做选择，即使 embedding 在直播尺度下变差，
准确率也应该大幅回升——这决定了"sim 约束 + 外观"这条路通不通。
"""

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

#: 各关图鉴（用户提供 / 我们的标注统计）
STAGE_CODES = {
    "v001_1-8": ["B1", "B2", "1", "17", "19", "21", "A9", "CS", "O2"],
    "v002_1-9": ["B1", "B2", "1", "A9", "19", "21", "O1"],
    "v000_1-7": ["B2", "O2"],
}
ALL_CODES = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
             "U1", "U3", "U7", "U9", "U17"]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def evaluate(frames, clicks, model, encoder, gallery, scale: float) -> tuple[int, int]:
    correct = total = 0
    for frame, points in frames:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        result = model.predict(image, imgsz=1280, conf=0.3, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(boxes)
        if not boxes:
            total += len(points)
            continue
        matches = gallery.identify(encoder.embed(
            [crop_box(image, [b[0], b[1], b[2] - b[0], b[3] - b[1]], pad=0.0) for b in boxes]))
        for code, x, y in points:
            gx, gy = x * scale, y * scale
            half_w, half_h = max(8.0, 1920 * 0.015) * scale, max(8.0, 1080 * 0.03) * scale
            gt = [gx - half_w, gy - half_h, gx + half_w, gy + half_h]
            best, best_iou = -1, 0.3
            for index, box in enumerate(boxes):
                value = iou(gt, box)
                if value > best_iou:
                    best, best_iou = index, value
            total += 1
            if best >= 0 and matches[best].code == code:
                correct += 1
    return correct, total


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    encoder = Encoder("resnet18", device="cuda", checkpoint=ROOT / "weights/embedder.pt")
    clicks = load_clicks(ROOT / "clicks")

    for video, codes in STAGE_CODES.items():
        frames = []
        for frame, items in clicks.items():
            if not frame.startswith(video):
                continue
            points = [(c.cls, c.x, c.y) for c in items if not c.is_ignore and c.cls]
            if points:
                frames.append((frame, points))
        frames = frames[:24]
        if not frames:
            continue
        print(f"\n== {video}  {len(frames)} 帧 ==")
        for name, gallery_codes in (("全 17 类", ALL_CODES), (f"限定 {len(codes)} 类", codes)):
            gallery = Gallery.from_crops(ROOT / "dataset" / "crops", encoder,
                                         max_per_code=240, prototypes_per_code=6,
                                         keep_codes=gallery_codes,
                                         reference_dir=ROOT / "reference" / "enemies")
            for scale_name, scale in (("1080p", 1.0), ("576p", 1024 / 1920)):
                correct, total = evaluate(frames, clicks, model, encoder, gallery, scale)
                print(f"  {name:<12} {scale_name}: {correct/max(1,total):6.1%} ({correct}/{total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
