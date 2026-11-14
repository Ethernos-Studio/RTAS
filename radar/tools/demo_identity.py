"""8 秒演示片：30 帧、检测 + 身份编号（图库匹配），每帧停留 0.27s。

用法：
    uv run python tools/demo_identity.py
产出：
    out/demo_identity_8s.mp4
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DATASET = ROOT / "dataset"
WEIGHTS = ROOT / "runs" / "arknights" / "v1" / "weights" / "best.pt"
GALLERY = ROOT / "weights" / "gallery.npz"
FRAME_COUNT = 30
HOLD = 8                       # 30 帧 × 8 = 240 帧 = 8 秒 @30fps
SIZE = (1280, 720)
FPS = 30

FONT_DIR = ("C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf")


def font(size: int):
    for path in FONT_DIR:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_HUD = font(22)


def main() -> int:
    from ultralytics import YOLO

    from eyesofpriestess.identity import Encoder, Gallery, crop_box

    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))
    gallery = Gallery.load(GALLERY)
    encoder = Encoder("resnet18")

    # 取同一段视频里连续的 30 帧：画面连续、看起来像"实时推理"
    images = sorted((ROOT / "frames").glob("v001_1-8_*.jpg"))[:FRAME_COUNT]
    if not images:
        images = sorted((DATASET / "images" / "val").glob("*.jpg"))[:FRAME_COUNT]
    frames: list[np.ndarray] = []
    total_boxes = 0
    total_unknown = 0
    total_ambiguous = 0

    for index, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        result = model.predict(image, imgsz=1280, conf=0.25, verbose=False)[0]
        boxes, crops = [], []
        for box in result.boxes or []:
            if int(box.cls[0]) != 0:          # 身份只对 enemy 做
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
            crops.append(crop_box(image, [x1, y1, x2 - x1, y2 - y1], pad=0.0))

        matches = gallery.identify(encoder.embed(crops)) if crops else []

        canvas = cv2.resize(image, SIZE)
        scale_x, scale_y = SIZE[0] / width, SIZE[1] / height
        unknown_here = 0
        for (x1, y1, x2, y2), match in zip(boxes, matches):
            p1 = (int(x1 * scale_x), int(y1 * scale_y))
            p2 = (int(x2 * scale_x), int(y2 * scale_y))
            if not match.code:
                unknown_here += 1
                color = (120, 120, 120)
                label = "unknown"
            else:
                color = (90, 220, 90)
                label = f"{match.code} {match.score:.2f}"
                if match.ambiguous:
                    label += " ?"
                    total_ambiguous += 1
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, label, (p1[0], max(18, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        total_boxes += len(boxes)
        total_unknown += unknown_here

        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (SIZE[0], 44)], fill=(16, 14, 12))
        draw.text((14, 9), "EYESOFPRIESTESS · 检测 + 身份编号（图库原型匹配，非分类器）",
                  font=F_HUD, fill=(255, 255, 255))
        draw.text((SIZE[0] - 140, 9), f"{index:02d}/{len(images)}", font=F_HUD, fill=(190, 190, 190))
        draw.rectangle([(0, SIZE[1] - 38), (SIZE[0], SIZE[1])], fill=(16, 14, 12))
        draw.text((14, SIZE[1] - 32),
                  f"本帧检出 {len(boxes)} 个敌人 · 灰色=认不出，问号=top1/top2 接近 · "
                  f"身份匹配为实验特性（top-1 48%，待接官方参考图）",
                  font=F_HUD, fill=(210, 210, 210))
        frames.append(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR))
        frames.extend([frames[-1]] * (HOLD - 1))

    writer = cv2.VideoWriter(str(OUT / "demo_identity_8s.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT / 'demo_identity_8s.mp4'}  {len(frames)} 帧 / {len(frames) / FPS:.1f}s")
    print(f"检出 {total_boxes} 个敌人；编号未知 {total_unknown}；歧义 {total_ambiguous}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
