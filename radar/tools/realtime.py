"""实时推理管线（多线程流水线）：解码 / 检测 / 渲染各占一条线程，互不等待。

实测结论（GTX 1650 Laptop，1920x1080 输入）：
* 检测 960 批量 8：34~36 FPS，GPU 利用率 72~76% —— **显卡没跑满，瓶颈在 CPU 侧**；
* 检测 1280：20 FPS，GPU 76~86% —— 1280 才是真的算不过来；
* 解码 84 FPS、画框 138 FPS。

所以做法是：解码线程只负责喂帧，主线程批量推理，渲染线程负责画框写视频。
任何一环单独看都不是瓶颈，串起来（单线程）才会掉到 13~20 FPS。

用法：
    uv run python tools/realtime.py --src raw/1-8.mp4 --start 31 --seconds 8 --imgsz 960 --batch 8
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WEIGHTS = ROOT / "runs" / "arknights" / "v1" / "weights" / "best.pt"
GALLERY = ROOT / "weights" / "gallery.npz"
OUT = ROOT / "out"
FPS = 30.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "raw" / "1-8.mp4"))
    parser.add_argument("--start", type=float, default=31.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.75)
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--out", default=str(OUT / "demo_realtime.mp4"))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-render", action="store_true",
                        help="不画框（雷达引擎的正确形态：只吐 scene JSONL，overlay 由 RTAS 画）")
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    from ultralytics import YOLO

    from eyesofpriestess.frames import open_video
    from eyesofpriestess.identity import Encoder, Gallery, crop_box
    from eyesofpriestess.postprocess import MergeSuppressor
    from eyesofpriestess.tracking import SimpleTracker

    use_identity = not args.no_identity
    model = YOLO(str(WEIGHTS))
    gallery = Gallery.load(GALLERY) if use_identity else None
    encoder = Encoder("resnet18", device="cuda") if use_identity else None
    tracker = SimpleTracker(iou_threshold=0.25)
    labels: dict[int, object] = {}

    want = int(round(args.seconds * FPS))
    frame_queue: queue.Queue = queue.Queue(maxsize=64)
    render_queue: queue.Queue = queue.Queue(maxsize=64)
    stop = threading.Event()
    counters = {"decoded": 0, "detected_frames": 0, "detected": 0, "identified": 0, "rendered": 0}

    out_size = None
    writer = None
    if not args.no_video:
        OUT.mkdir(parents=True, exist_ok=True)

    def decoder() -> None:
        stream = open_video(Path(args.src), sample_fps=FPS, start=args.start)
        try:
            for _timestamp, frame in stream.frames:
                frame_queue.put(frame)
                counters["decoded"] += 1
                if counters["decoded"] >= want:
                    break
        finally:
            stream.close()
            frame_queue.put(None)

    def renderer() -> None:
        nonlocal out_size, writer
        while True:
            item = render_queue.get()
            if item is None:
                break
            frame, units = item
            if args.no_render:
                counters["rendered"] += 1
                continue
            height, width = frame.shape[:2]
            if out_size is None:
                out_size = (args.width, int(round(args.width * height / width)))
                if not args.no_video:
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, out_size)
            canvas = cv2.resize(frame, out_size)
            sx, sy = out_size[0] / width, out_size[1] / height
            for (x1, y1, x2, y2), track_id in units:
                match = labels.get(track_id)
                p1 = (int(x1 * sx), int(y1 * sy))
                p2 = (int(x2 * sx), int(y2 * sy))
                if match is None or not getattr(match, "code", ""):
                    color, text = (140, 140, 140), f"#{track_id}"
                else:
                    color = (90, 220, 90)
                    text = f"#{track_id} {match.code} {match.score:.2f}"
                cv2.rectangle(canvas, p1, p2, color, 2)
                cv2.putText(canvas, text, (p1[0], max(16, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(canvas, f"frame {counters['rendered'] + 1:03d}  tracks {len(units)}  "
                                f"imgsz {args.imgsz}  batch {args.batch}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if writer is not None:
                writer.write(canvas)
            counters["rendered"] += 1
        if writer is not None:
            writer.release()

    decoder_thread = threading.Thread(target=decoder, daemon=True)
    render_thread = threading.Thread(target=renderer, daemon=True)
    decoder_thread.start()
    render_thread.start()

    started = time.perf_counter()
    batch: list[np.ndarray] = []
    while True:
        item = frame_queue.get()
        if item is None:
            break
        batch.append(item)
        if len(batch) < args.batch:
            continue
        results = model.predict(batch, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                verbose=False, device="0")
        new_crops, new_ids = [], []
        for frame, result in zip(batch, results):
            boxes = []
            for box in result.boxes or []:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
            boxes = MergeSuppressor()(boxes)
            units = tracker.update(boxes)
            counters["detected"] += len(units)
            render_queue.put((frame, units))
            for box, track_id in units:
                if use_identity and track_id not in labels:
                    new_crops.append(crop_box(frame, [box[0], box[1], box[2] - box[0],
                                                      box[3] - box[1]], pad=0.0))
                    new_ids.append(track_id)
        counters["detected_frames"] += len(batch)
        if new_crops:
            counters["identified"] += len(new_crops)
            for track_id, match in zip(new_ids, gallery.identify(encoder.embed(new_crops))):
                labels[track_id] = match
        batch = []
        if counters["rendered"] >= want:
            break

    if batch:
        results = model.predict(batch, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                verbose=False, device="0")
        for frame, result in zip(batch, results):
            boxes = []
            for box in result.boxes or []:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
            render_queue.put((frame, tracker.update(boxes)))
    render_queue.put(None)
    render_thread.join(timeout=30)
    elapsed = time.perf_counter() - started

    print(f"输出 {counters['rendered']} 帧 / {elapsed:.2f}s = {counters['rendered'] / max(elapsed, 1e-6):.1f} FPS"
          f"（imgsz={args.imgsz} batch={args.batch} 检测 {counters['detected_frames']} 帧，"
          f"身份匹配 {counters['identified']} 次，tracks {len(labels)}）")
    print(f"累计检出 {counters['detected']} 个敌人；输出 {args.out if not args.no_video else '（未写视频）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
