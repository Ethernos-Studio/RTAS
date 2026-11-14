"""30fps 真·逐帧推理演示：取录像的一段 8 秒（240 帧），逐帧检测 + 身份编号，按 30fps 输出。

用法：
    uv run python tools/demo_30fps.py --src raw/1-8.mp4 --start 31.0 --seconds 8
产出：
    out/demo_30fps_8s.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
def embedder_for(gallery) -> Path | None:
    """图库是 256 维说明用训练过的 embedding 建的；512 维才是 ImageNet 特征。"""
    try:
        import numpy as np
        data = np.load(str(gallery), allow_pickle=False)
        dim = int(data["prototypes"].shape[1])
    except Exception:
        return None
    checkpoint = Path(gallery).parent / "embedder.pt"
    if dim == 512 or not checkpoint.exists():
        return None
    return checkpoint



OUT = ROOT / "out"
WEIGHTS = ROOT / "runs" / "arknights" / "v1" / "weights" / "best.pt"
GALLERY = ROOT / "weights" / "gallery.npz"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "raw" / "1-8.mp4"))
    parser.add_argument("--start", type=float, default=31.0, help="从第几秒开始")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.8,
                        help="NMS 阈值：调高才不会把挨在一起的两个敌人合并成一个")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--out", default=str(OUT / "demo_30fps_8s.mp4"))
    args = parser.parse_args()

    from ultralytics import YOLO

    from eyesofpriestess.identity import Encoder, Gallery, crop_box
    from eyesofpriestess.postprocess import MergeSuppressor

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {args.src}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    model = YOLO(str(WEIGHTS))
    gallery = Gallery.load(GALLERY)
    encoder = Encoder("resnet18", checkpoint=embedder_for(GALLERY))

    want = int(round(args.seconds * FPS))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)

    written = 0
    detections = 0
    while written < want:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               verbose=False)[0]

        boxes, crops = [], []
        merge = MergeSuppressor()
        raw = [[float(v) for v in box.xyxy[0].tolist()]
               for box in (result.boxes or []) if int(box.cls[0]) == 0]
        kept = merge(raw)
        kept_keys = {(round(b[0]), round(b[1])) for b in kept}
        for x1, y1, x2, y2 in raw:
            if (round(x1), round(y1)) not in kept_keys and len(raw) > len(kept):
                continue
            boxes.append((x1, y1, x2, y2))
            crops.append(crop_box(frame, [x1, y1, x2 - x1, y2 - y1], pad=0.0))
        matches = [] if args.no_identity else (gallery.identify(encoder.embed(crops)) if crops else [])
        detections += len(boxes)

        canvas = cv2.resize(frame, SIZE)
        scale_x, scale_y = SIZE[0] / width, SIZE[1] / height
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            p1 = (int(x1 * scale_x), int(y1 * scale_y))
            p2 = (int(x2 * scale_x), int(y2 * scale_y))
            match = matches[index] if index < len(matches) else None
            if match is None or not match.code:
                color, label = (120, 120, 120), "enemy"
            else:
                color = (90, 220, 90)
                label = f"{match.code} {match.score:.2f}" + (" ?" if match.ambiguous else "")
            cv2.rectangle(canvas, p1, p2, color, 2)
            cv2.putText(canvas, label, (p1[0], max(18, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        timestamp = args.start + written / source_fps
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (SIZE[0], 44)], fill=(16, 14, 12))
        draw.text((14, 9), f"EYESOFPRIESTESS · 30fps 逐帧推理（{Path(args.src).name}）",
                  font=F_HUD, fill=(255, 255, 255))
        draw.text((SIZE[0] - 240, 9), f"t={timestamp:05.1f}s  {written + 1:03d}/{want}",
                  font=F_HUD, fill=(190, 190, 190))
        draw.rectangle([(0, SIZE[1] - 38), (SIZE[0], SIZE[1])], fill=(16, 14, 12))
        draw.text((14, SIZE[1] - 32),
                  f"本帧敌人 {len(boxes)} · 检测=YOLOv8s@1280 · 编号=图库原型匹配（实验特性）",
                  font=F_HUD, fill=(210, 210, 210))
        canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        writer.write(canvas)
        written += 1

    cap.release()
    writer.release()
    print(f"wrote {args.out}  {written} 帧 / {written / FPS:.1f}s @ {FPS}fps，累计检出 {detections}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
