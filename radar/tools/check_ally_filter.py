"""量一下友军抑制的效果：在 val 上按血条兜底压掉干员误报。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.ally import AllyFilter  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    data = ROOT / "dataset"
    flt = AllyFilter()

    tp = fp = fn = 0
    tp_after = fp_after = 0
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
        units = []
        for box in result.boxes or []:
            if int(box.cls[0]) != 0:
                continue
            units.append((list(map(float, box.xyxy[0].tolist())), 0))

        matched: set[int] = set()
        for box, _ in units:
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
        fn += len(gt) - len(matched)

        kept = flt.filter(image, units)
        matched_after: set[int] = set()
        for box, _ in kept:
            hit = -1
            for index, g in enumerate(gt):
                if index not in matched_after and iou(box, g) >= 0.5:
                    hit = index
                    break
            if hit >= 0:
                matched_after.add(hit)
                tp_after += 1
            else:
                fp_after += 1

    print(f"原始: TP {tp} FP {fp} FN {fn}  → P {tp/(tp+fp):.2f} R {tp/(tp+fn):.2f}")
    print(f"抑制后: TP {tp_after} FP {fp_after}  → P {tp_after/max(1,tp_after+fp_after):.2f} "
          f"R {tp_after/max(1,tp_after+fn):.2f}（压掉 {flt.dropped} 个框）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
