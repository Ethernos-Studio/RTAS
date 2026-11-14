"""验证镜头标定：拿部署帧里的敌人裁图，用标定矩阵转正，和正常帧里的同类裁图对比。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.cv.calib import estimate, rectify_frame, warp_box  # noqa: E402

NORMAL = ROOT / "frames" / "v000_1-7_f000979.jpg"
DEPLOY = ROOT / "frames" / "v000_1-7_f001089.jpg"


def main() -> int:
    calibration = estimate(NORMAL, DEPLOY, stage="1-7")
    print(f"标定 1-7: 内点 {calibration.inliers} | 竖线倾角 {calibration.angle_deg:+.2f}° | "
          f"尺度 ×{calibration.scale:.3f}")
    matrix = np.array(calibration.matrix, dtype=np.float64)

    # 部署帧里挑几只敌人（用检测器出框），把裁图转正后拼图
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v1/weights/best.pt"))
    deploy = cv2.imread(str(DEPLOY))
    result = model.predict(deploy, imgsz=1280, conf=0.3, iou=0.8, verbose=False)[0]
    boxes = []
    for box in result.boxes or []:
        if int(box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        boxes.append([x1, y1, x2 - x1, y2 - y1])
    print(f"部署帧检测到 {len(boxes)} 个敌人")

    rectified_frame = rectify_frame(deploy, matrix)
    tiles = []
    for box in boxes[:6]:
        x, y, w, h = (int(v) for v in box)
        raw = deploy[max(0, y):y + h, max(0, x):x + w]
        warped = warp_box(box, matrix)
        wx, wy, ww, wh = (int(round(v)) for v in warped)
        rectified = rectified_frame[max(0, wy):wy + wh, max(0, wx):wx + ww]
        for image, tag in ((raw, "raw"), (rectified, "fixed")):
            if image.size == 0:
                continue
            tile = cv2.resize(image, (128, 128))
            cv2.putText(tile, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            tiles.append(tile)
    if tiles:
        rows = []
        for index in range(0, len(tiles), 4):
            chunk = tiles[index:index + 4]
            while len(chunk) < 4:
                chunk.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(chunk))
        out = ROOT / "out" / "_calib_check.jpg"
        cv2.imwrite(str(out), np.vstack(rows))
        print("对比图 ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
