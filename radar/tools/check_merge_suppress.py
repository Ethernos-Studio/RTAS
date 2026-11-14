"""量一下合并框抑制的效果：val 上的框数、TP/FP 变化。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.postprocess import MergeSuppressor  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def evaluate(units, gt) -> tuple[int, int, int]:
    matched: set[int] = set()
    tp = fp = 0
    for box in units:
        hit = -1
        for index, g in enumerate(gt):
            if index not in matched and iou(box, g) >= 0.5:
                hit = index
                break
        if hit >= 0:
            matched.add(hit)
            tp += 1
        else:
            fp += 1
    return tp, fp, len(gt) - len(matched)


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    data = ROOT / "dataset"
    suppressor = MergeSuppressor()

    before = [0, 0, 0]
    after = [0, 0, 0]
    boxes_before = boxes_after = 0

    for image_path in sorted((data / "images/val").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gt = []
        for line in (data / "labels/val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = (float(v) for v in line.split())
            gt.append([(cx - w / 2) * width, (cy - h / 2) * height,
                       (cx + w / 2) * width, (cy + h / 2) * height])

        result = model.predict(image, imgsz=960, conf=0.25, iou=0.8, verbose=False)[0]
        units = [list(map(float, box.xyxy[0].tolist()))
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes_before += len(units)
        kept = suppressor(units)
        boxes_after += len(kept)

        for target, boxes in ((before, units), (after, kept)):
            tp, fp, fn = evaluate(boxes, gt)
            target[0] += tp
            target[1] += fp
            target[2] += fn

    def report(name, values, boxes):
        tp, fp, fn = values
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        print(f"{name}: 框 {boxes}  TP {tp} FP {fp} FN {fn}  P {precision:.2f} R {recall:.2f}")

    report("原始      ", before, boxes_before)
    report("抑制合并框", after, boxes_after)
    print(f"共丢掉 {suppressor.dropped} 个合并框")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
