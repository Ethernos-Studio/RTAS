"""量化"合并框"有多严重：一个预测框里塞了 ≥2 个 GT，就是"两怪合一"。

用法：
    uv run python tools/merge_rate.py [--weights ...] [--imgsz 960 1280]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.postprocess import MergeSuppressor, containment  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return 0.0 if union <= 0 else inter / union


def load_gt(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, cx, cy, w, h = (float(v) for v in line.split())
        if int(cls) != 0:            # 只看敌人
            continue
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def merge_errors(pred_boxes: list[list[float]], gt: list[list[float]]) -> tuple[int, int, int]:
    """返回 (合并框个数, 被吞掉的 GT 数, 正常匹配的 GT 数)。"""
    merged = swallowed = 0
    matched: set[int] = set()
    for box in pred_boxes:
        inside = [index for index, g in enumerate(gt) if containment(g, box) >= 0.7]
        if len(inside) >= 2:
            merged += 1
            swallowed += len(inside)
            matched.update(inside)
        else:
            for index in inside:
                matched.add(index)
    return merged, swallowed, len(gt) - len(matched)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(ROOT / "runs/arknights/v1/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[960, 1280])
    parser.add_argument("--iou", type=float, default=0.8)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    data = ROOT / "dataset"
    images = sorted((data / "images/val").glob("*.jpg"))
    print(f"val {len(images)} 帧，NMS iou={args.iou}")

    for imgsz in args.imgsz:
        for suppress in (False, True):
            merged = swallowed = missed = total_gt = 0
            for image_path in images:
                image = cv2.imread(str(image_path))
                height, width = image.shape[:2]
                gt = load_gt(data / "labels/val" / f"{image_path.stem}.txt", width, height)
                result = model.predict(image, imgsz=imgsz, conf=0.25, iou=args.iou, verbose=False)[0]
                boxes = [list(map(float, box.xyxy[0].tolist()))
                         for box in (result.boxes or []) if int(box.cls[0]) == 0]
                if suppress:
                    boxes = MergeSuppressor()(boxes)
                m, s, miss = merge_errors(boxes, gt)
                merged += m
                swallowed += s
                missed += miss
                total_gt += len(gt)
            tag = " +合并抑制" if suppress else "         "
            print(f"  imgsz={imgsz:4d}{tag}: 合并框 {merged:2d} 个，吞掉 {swallowed:2d}/{total_gt} 个 GT，"
                  f"漏检 {missed:2d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
