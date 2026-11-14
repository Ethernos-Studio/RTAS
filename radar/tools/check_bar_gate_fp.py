"""血条门控对"误检框"（大多是干员/召唤物）的拦截率。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.hpbar_gate import bar_color_at  # noqa: E402
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
    clicks = load_clicks(ROOT / "clicks")
    frames = [f for f in clicks if f.startswith(("v001_1-8", "v002_1-9"))][:16]
    fp_stats: dict[str, int] = {}
    tp_stats: dict[str, int] = {}
    for frame in frames:
        image = cv2.imread(str(ROOT / "frames" / frame))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = []
        for click in clicks[frame]:
            if click.is_ignore or not click.cls:
                continue
            half_w, half_h = max(8.0, width * 0.015), max(8.0, height * 0.03)
            gt.append([click.x - half_w, click.y - half_h, click.x + half_w, click.y + half_h])

        result = model.predict(image, imgsz=1280, conf=0.25, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        for box in MergeSuppressor()(boxes):
            color = bar_color_at(image, box) or "none"
            is_tp = any(iou(box, g) >= 0.4 for g in gt)
            bucket = tp_stats if is_tp else fp_stats
            bucket[color] = bucket.get(color, 0) + 1

    fp_total = sum(fp_stats.values())
    tp_total = sum(tp_stats.values())
    fp_removed = fp_stats.get("green", 0) + fp_stats.get("blue", 0)
    tp_removed = tp_stats.get("green", 0) + tp_stats.get("blue", 0)
    print(f"误检框 {fp_total} 个，血条 {fp_stats}")
    print(f"  门控会拦掉 {fp_removed} = {fp_removed/max(1,fp_total):.1%}")
    print(f"命中框 {tp_total} 个，血条 {tp_stats}")
    print(f"  门控误杀 {tp_removed} = {tp_removed/max(1,tp_total):.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
