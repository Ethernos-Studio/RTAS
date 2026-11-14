"""1-10 上身份错乱（源石虫 B1 被认成 U7）的三种图库配置对照。

1-10 没标注，但它的图鉴是 B1/B2/O1/1/A9/19/21（用户提供），
所以可以数"检出的源石虫被认成什么"，不需要逐帧 GT。

对照：
  A 当前图库（每类最多 240 张，U7 有 121 张，样本严重不均）
  B 均衡图库（每类最多 40 张）
  C 只保留 1-10 该有的编号（模拟关卡先验）
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from eyesofpriestess.identity import Encoder, Gallery, crop_box  # noqa: E402
from eyesofpriestess.postprocess import MergeSuppressor  # noqa: E402

STAGE_CODES = ["B1", "B2", "O1", "1", "A9", "19", "21"]
ALL_CODES = ["B1", "B2", "O1", "O2", "1", "A3", "D1", "17", "19", "21", "A9", "CS",
             "U1", "U3", "U7", "U9", "U17"]


def main() -> int:
    from ultralytics import YOLO

    model = YOLO(str(ROOT / "runs/arknights/v3/weights/best.pt"))
    frames = sorted((ROOT / "frames").glob("v000_1-10_*.jpg"))[:12]
    print(f"用 {len(frames)} 张 1-10 帧对照\n")

    boxes_per_frame = []
    for path in frames:
        image = cv2.imread(str(path))
        result = model.predict(image, imgsz=1280, conf=0.45, iou=0.8, verbose=False)[0]
        raw = [[float(v) for v in box.xyxy[0].tolist()]
               for box in (result.boxes or []) if int(box.cls[0]) == 0]
        boxes = MergeSuppressor()(raw)
        crops = [crop_box(image, [b[0], b[1], b[2] - b[0], b[3] - b[1]], pad=0.0) for b in boxes]
        boxes_per_frame.append(crops)
    total = sum(len(c) for c in boxes_per_frame)
    print(f"共 {total} 个检出框")

    encoder = Encoder("resnet18", device="cuda", checkpoint=ROOT / "weights/embedder.pt")
    configs = {
        "A 当前图库": dict(prototypes=6, max_per_code=240),
        "B 均衡(每类40)": dict(prototypes=6, max_per_code=40),
        "C 限本关编号": dict(prototypes=6, max_per_code=240, codes=STAGE_CODES),
    }

    for name, cfg in configs.items():
        codes = cfg.get("codes", ALL_CODES)
        gallery = Gallery.from_crops(ROOT / "dataset" / "crops", encoder,
                                     max_per_code=cfg["max_per_code"],
                                     prototypes_per_code=cfg["prototypes"],
                                     keep_codes=codes,
                                     reference_dir=ROOT / "reference" / "enemies")
        counter: Counter = Counter()
        scores = []
        for crops in boxes_per_frame:
            if not crops:
                continue
            for match in gallery.identify(encoder.embed(crops)):
                counter[match.code or "?"] += 1
                scores.append(match.score)
        mean_score = sum(scores) / max(1, len(scores))
        print(f"{name:<16} 编号分布 {dict(counter.most_common(6))} | 平均相似度 {mean_score:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
