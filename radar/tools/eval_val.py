"""在 val 上做 IoU 匹配的检出统计（TP/FP/FN），比"数框"靠谱。

用法：
    uv run python tools/eval_val.py [--weights runs/arknights/v1/weights/best.pt] [--conf 0.25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: "enemy", 1: "ally", 2: "tile"}


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs/arknights/v1/weights/best.pt"))
    parser.add_argument("--data", default=str(ROOT / "dataset"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    from ultralytics import YOLO

    data_dir = Path(args.data)
    model = YOLO(args.weights)
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in NAMES.values()}
    frames = sorted((data_dir / "images" / "val").glob("*.jpg"))

    import cv2

    for image_path in frames:
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        label_path = data_dir / "labels" / "val" / f"{image_path.stem}.txt"

        gt_boxes: list[tuple[str, list[float]]] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = (float(v) for v in line.split())
            gt_boxes.append(
                (NAMES.get(int(cls), str(int(cls))),
                 [(cx - w / 2) * width, (cy - h / 2) * height,
                  (cx + w / 2) * width, (cy + h / 2) * height])
            )

        result = model.predict(image, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        preds: list[tuple[str, list[float]]] = []
        for box in result.boxes or []:
            cls = int(box.cls[0])
            preds.append((NAMES.get(cls, str(cls)), list(map(float, box.xyxy[0].tolist()))))

        matched_gt: set[int] = set()
        for name, pred_box in preds:
            best_index, best_iou = -1, args.iou
            for index, (gt_name, gt_box) in enumerate(gt_boxes):
                if index in matched_gt or gt_name != name:
                    continue
                value = iou(pred_box, gt_box)
                if value >= best_iou:
                    best_index, best_iou = index, value
            if best_index >= 0:
                matched_gt.add(best_index)
                totals[name]["tp"] += 1
            else:
                totals[name]["fp"] += 1
        for index, (gt_name, _box) in enumerate(gt_boxes):
            if index not in matched_gt:
                totals[gt_name]["fn"] += 1

    report = {}
    for name, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        if tp + fp + fn == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        report[name] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    out = ROOT / "dataset" / "eval_val.json"
    out.write_text(json.dumps({"frames": len(frames), "iou": args.iou,
                               "conf": args.conf, "classes": report},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
