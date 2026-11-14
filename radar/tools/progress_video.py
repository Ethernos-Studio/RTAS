"""生成一份进度报告 MP4（发群里用）：统计卡片 + v1 在未见关卡上的检测演示 + 指标 + 下一步。

用法：
    uv run python tools/progress_video.py
产出：
    out/progress_v1.mp4   (1280x720, 30fps, 约 18s)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
WEIGHTS = ROOT / "runs" / "arknights" / "v1" / "weights" / "best.pt"
DATASET = ROOT / "dataset"
SIZE = (1280, 720)
FPS = 30
NAMES = {0: "enemy", 1: "ally", 2: "tile"}
COLORS = {"enemy": (480, 480, 255), "ally": (255, 170, 120), "tile": (200, 220, 120)}

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(64)
FONT_SUB = load_font(30)
FONT_BODY = load_font(26)
FONT_SMALL = load_font(20)
FONT_HUD = load_font(22)


def gradient(width: int, height: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    """竖直渐变底（BGR）。"""
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top_arr = np.array(top, dtype=np.float32)[None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, :]
    column = top_arr * (1 - ramp) + bottom_arr * ramp          # (h, 3)
    return np.repeat(column[:, None, :], width, axis=1).astype(np.uint8)


def card(title: str, lines: list[str], *, subtitle: str = "", accent=(90, 200, 255)) -> np.ndarray:
    """一张文字卡片（背景渐变 + 标题 + 若干行）。"""
    frame = gradient(SIZE[0], SIZE[1], (26, 22, 20), (48, 40, 34))
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.rectangle([(0, 0), (12, SIZE[1])], fill=tuple(reversed(accent)))

    y = 70
    draw.text((60, y), title, font=FONT_TITLE, fill=(255, 255, 255))
    if subtitle:
        y += 92
        draw.text((62, y), subtitle, font=FONT_SUB, fill=tuple(reversed(accent)))
    y += 96
    for line in lines:
        draw.text((62, y), line, font=FONT_BODY, fill=(226, 226, 226))
        y += 46
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def hold(frame: np.ndarray, seconds: float) -> list[np.ndarray]:
    return [frame.copy() for _ in range(max(1, int(round(seconds * FPS))))]


def title_card(stats: dict) -> np.ndarray:
    lines = [
        f"素材：{stats['videos']} 段实战录像（1-7 / 1-8 / 1-9） · 抽帧 {stats['frames']} 张",
        f"标注：{stats['labeled_frames']} 帧 / {stats['points']} 个点（图鉴编号点选，框由 SAM 生成）",
        f"数据集：{stats['boxes']} 个框（enemy {stats['enemy']} / tile {stats['tile']}），按视频切分",
        "",
        "模型：YOLOv8s @1280 · freeze=10 · ByteTrack · 纯像素坐标",
    ]
    return card("EYESOFPRIESTESS · 进度", lines, subtitle="2026-11-14 · L1 雷达数据管线打通")


def metrics_card(report: dict) -> np.ndarray:
    lines = [
        "训练集：1-7 + 1-8（90 帧 / 420 个敌人框）",
        "验证集：1-9（17 帧 / 56 个敌人）—— 完全没参与训练",
        "",
        f"IoU0.5：精确率 {report['precision']:.0%}   召回 {report['recall']:.0%}   F1 {report['f1']:.2f}",
        f"（TP {report['tp']} / FP {report['fp']} / FN {report['fn']}）",
        f"ultralytics val：mAP50 {report['mp50']:.3f}   mAP50-95 {report['mp']:.3f}",
        "",
        "结论：位置准、召回还差一截；瓶颈是跨关卡数据，不是模型",
    ]
    return card("v1 指标（独立视频 1-9）", lines,
                subtitle="20 epoch / 6.7 分钟 / GTX 1650", accent=(120, 220, 160))


def next_card() -> np.ndarray:
    lines = [
        "1. 把 1-8 换作 val 视频，1-9 进训练（新增关卡不丢样本）",
        "2. 再录一条 1-10 做长期 val，三条训练",
        "3. 离线 copy-paste 增强（按透视尺度贴敌人）",
        "4. 主动学习：按低置信/漏检排序推荐下一批标注",
        "",
        "产出接口：scene.jsonl（track_id / cls / conf / box，纯像素）",
    ]
    return card("下一步", lines, subtitle="需要你：录一条新关卡 + 继续点选", accent=(255, 180, 120))


def load_stats() -> dict:
    report = json.loads((DATASET / "report.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / "frames" / "manifest.jsonl"
    videos = set()
    frames = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            videos.add(json.loads(line)["video"])
            frames += 1
    return {
        "videos": len(videos),
        "frames": frames,
        "labeled_frames": len(report["frames"]),
        "points": report["points"] + report["ignore"],
        "boxes": report["boxes"],
        "enemy": report["class_hist"].get("enemy", 0),
        "tile": report["class_hist"].get("tile", 0),
    }


def demo_frames(model) -> tuple[list[np.ndarray], dict]:
    """把 val（1-9）的帧按顺序过一遍模型，画框 + HUD。"""
    images = sorted((DATASET / "images" / "val").glob("*.jpg"))
    tiles: list[np.ndarray] = []
    gt_total = 0
    pred_total = 0

    for index, image_path in enumerate(images, 1):
        frame = cv2.imread(str(image_path))
        height, width = frame.shape[:2]
        height_img = cv2.resize(frame, SIZE)

        label_path = DATASET / "labels" / "val" / f"{image_path.stem}.txt"
        gt = 0
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls, cx, cy, w, h = (float(v) for v in line.split())
            x1 = int((cx - w / 2) * width * SIZE[0] / width)
            y1 = int((cy - h / 2) * height * SIZE[1] / height)
            x2 = int((cx + w / 2) * width * SIZE[0] / width)
            y2 = int((cy + h / 2) * height * SIZE[1] / height)
            cv2.rectangle(height_img, (x1, y1), (x2, y2), (120, 230, 120), 2)
            gt += 1

        result = model.predict(frame, imgsz=1280, conf=0.25, verbose=False)[0]
        pred = 0
        for box in result.boxes or []:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cls = int(box.cls[0])
            name = NAMES.get(cls, str(cls))
            scale_x, scale_y = SIZE[0] / width, SIZE[1] / height
            p1 = (int(x1 * scale_x), int(y1 * scale_y))
            p2 = (int(x2 * scale_x), int(y2 * scale_y))
            cv2.rectangle(height_img, p1, p2, COLORS.get(name, (255, 255, 255)), 2)
            cv2.putText(height_img, f"{name} {float(box.conf[0]):.2f}",
                        (p1[0], max(16, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        COLORS.get(name, (255, 255, 255)), 1)
            pred += 1
        gt_total += gt
        pred_total += pred

        canvas = cv2.cvtColor(np.array(Image.fromarray(cv2.cvtColor(height_img, cv2.COLOR_BGR2RGB))),
                              cv2.COLOR_RGB2BGR)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        draw.rectangle([(0, 0), (SIZE[0], 46)], fill=(18, 16, 14))
        draw.text((16, 10), f"v1 推理 · 未见过的关卡 1-9", font=FONT_HUD, fill=(255, 255, 255))
        draw.text((520, 10), f"绿框=人工 GT", font=FONT_HUD, fill=(150, 230, 150))
        draw.text((760, 10), f"红框=模型预测", font=FONT_HUD, fill=(120, 120, 255))
        draw.text((1010, 10), f"{index:02d}/{len(images)}", font=FONT_HUD, fill=(200, 200, 200))
        draw.rectangle([(0, SIZE[1] - 40), (SIZE[0], SIZE[1])], fill=(18, 16, 14))
        draw.text((16, SIZE[1] - 33), f"本帧 GT {gt} / 预测 {pred}", font=FONT_HUD, fill=(220, 220, 220))
        tiles.extend(hold(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), 0.45))

    return tiles, {"gt": gt_total, "pred": pred_total}


def main() -> int:
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    stats = load_stats()
    demo_report = json.loads((ROOT / "dataset" / "report.json").read_text(encoding="utf-8"))
    model = YOLO(str(WEIGHTS))

    demo, counts = demo_frames(model)
    eval_path = DATASET / "eval_val.json"
    if eval_path.exists():
        measured = json.loads(eval_path.read_text(encoding="utf-8"))["classes"]["enemy"]
    else:
        measured = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    metrics = {**measured, "mp50": 0.684, "mp": 0.458}

    frames: list[np.ndarray] = []
    frames += hold(title_card(stats), 4.0)
    frames += demo
    frames += hold(metrics_card(metrics), 4.5)
    frames += hold(next_card(), 4.5)
    frames += hold(card("谢谢 · 需要的是：新关卡录像 + 继续点选", [], subtitle="repo: radar/", accent=(255, 255, 255)), 2.5)

    writer = cv2.VideoWriter(str(OUT / "progress_v1.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f"wrote {OUT / 'progress_v1.mp4'}  {len(frames)} 帧 / {len(frames) / FPS:.1f}s")
    print(f"demo: GT {counts['gt']} / pred {counts['pred']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
