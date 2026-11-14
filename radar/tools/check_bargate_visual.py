"""在有干员的帧上检查门控：把被判为 ally 的框画出来，人工看一眼对不对。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.hpbar_gate import find_hp_bar  # noqa: E402
from eyesofpriestess.postprocess import MergeSuppressor  # noqa: E402


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    frames = ["v001_1-8_f000600.jpg", "v001_1-8_f000700.jpg", "v002_1-9_f001300.jpg",
              "v000_1-7_f000300.jpg"]
    out_dir = ROOT / "out" / "_bargate"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in frames:
        path = ROOT / "frames" / name
        image = cv2.imread(str(path))
        if image is None:
            print("缺帧", name)
            continue
        result = model.predict(image, imgsz=1280, conf=0.3, iou=0.8, verbose=False)[0]
        boxes = [[float(v) for v in box.xyxy[0].tolist()]
                 for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(boxes)
        counts: dict[str, int] = {}
        canvas = image.copy()
        for box in boxes:
            found = find_hp_bar(image, box)
            kind = found[0] if found else "none"
            counts[kind] = counts.get(kind, 0) + 1
            color = {"ally": (255, 200, 0), "enemy": (0, 0, 255), "none": (160, 160, 160)}[kind]
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, kind, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if found:
                cv2.line(canvas, (x1, found[1]), (x2, found[1]), (0, 255, 255), 1)
        cv2.imwrite(str(out_dir / name), canvas)
        print(f"{name}: {len(boxes)} 框 -> {counts}")
    print("图在", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
