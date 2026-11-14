"""检测 + ByteTrack -> 场景图 JSONL（L1 雷达的数据源）。

输出一行一帧（RTAS 侧只消费这个格式）::

    {"t": 3.2, "frame_idx": 96,
     "units": [{"track_id": 12, "cls": "enemy", "conf": 0.71, "box": [x, y, w, h]}, ...]}

坐标是**截图像素**，不做任何映射（HANDOFF §2.1）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2

#: 图片目录没有帧率，按这个假 fps 折算时间戳
ASSUMED_FPS = 30.0


def _load_yolo(weights: str) -> Any:
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - 延迟导入
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            "预测需要 ultralytics：`uv sync --extra train` 或 `pip install ultralytics`"
        ) from exc
    return YOLO(weights)


def iter_frames(
    source: Path | str,
    *,
    start: float = 0.0,
    seconds: float | None = None,
) -> Iterator[tuple[int, float, Any]]:
    """按顺序给 ``(frame_idx, t, frame)``；视频按真实 fps，图片目录按 30fps 估。

    ``start`` / ``seconds`` 用来只跑视频的一小段（实时演示、快速验证用）。
    """
    source = Path(source)
    if source.is_dir():
        for index, path in enumerate(sorted(source.glob("*.jpg")), 1):
            timestamp = index / ASSUMED_FPS
            if timestamp < start or (seconds is not None and timestamp > start + seconds):
                continue
            frame = cv2.imread(str(path))
            if frame is not None:
                yield index, timestamp, frame
        return

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or ASSUMED_FPS
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    index = int(start * fps)
    limit = None if seconds is None else start + seconds
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index += 1
            timestamp = index / fps
            if limit is not None and timestamp > limit:
                break
            yield index, timestamp, frame
    finally:
        cap.release()


def predict_track(
    *,
    weights: Path | str,
    source: Path | str,
    out: Path,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.8,
    device: str | None = None,
    tracker: str = "bytetrack.yaml",
    gallery: Path | str | None = None,
    start: float = 0.0,
    seconds: float | None = None,
) -> dict[str, Any]:
    """逐帧检测 + 跟踪，写 scene JSONL；返回统计。

    ``iou`` 是 NMS 阈值：**调高**（0.7~0.8）才不会把挨在一起的两个敌人合并成一个；
    塔防里敌人经常叠着走，用默认 0.45 会明显少报。
    ``gallery`` 给了就把图鉴编号也写进 unit（``code`` / ``code_score``）。
    """
    model = _load_yolo(str(weights))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    identifier = None
    if gallery:
        from .identity import Encoder, Gallery

        loaded = Gallery.load(Path(gallery))
        # 图库维度决定用哪个编码器：512 = ImageNet 特征，256 = 训练过的 embedding
        dim = int(loaded.prototypes.shape[1]) if loaded.prototypes.size else 512
        checkpoint = Path(gallery).parent / "embedder.pt"
        if dim == 512 or not checkpoint.exists():
            checkpoint = None
        identifier = (loaded, Encoder("resnet18", device=device, checkpoint=checkpoint))

    frames_written = 0
    units_total = 0
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, timestamp, frame in iter_frames(source, start=start, seconds=seconds):
            units = _infer(model, frame, imgsz, conf, tracker, device, iou)
            if identifier is not None and units:
                _attach_identity(identifier, frame, units)
            record = {"t": round(float(timestamp), 2), "frame_idx": index, "units": units}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            frames_written += 1
            units_total += len(units)

    return {"frames": frames_written, "units": units_total, "out": str(out)}


def _attach_identity(identifier: tuple[Any, Any], frame: Any, units: list[dict[str, Any]]) -> None:
    """给 enemy 类补上 ``code`` / ``code_score``（图库最近邻，不训练分类器）。"""
    from .identity import crop_box

    gallery, encoder = identifier
    targets = [unit for unit in units if unit.get("cls") == "enemy"]
    if not targets:
        return
    crops = [crop_box(frame, unit["box"], pad=0.0) for unit in targets]
    for unit, match in zip(targets, gallery.identify(encoder.embed(crops))):
        unit["code"] = match.code or None
        unit["code_score"] = round(match.score, 3)
        unit["code_ambiguous"] = match.ambiguous


def _infer(
    model: Any,
    frame: Any,
    imgsz: int,
    conf: float,
    tracker: str,
    device: str | None,
    iou: float = 0.75,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = dict(
        imgsz=imgsz, persist=True, tracker=tracker, conf=conf, iou=iou, verbose=False
    )
    if device:
        kwargs["device"] = device
    result = model.track(frame, **kwargs)[0]
    units: list[dict[str, Any]] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return units
    names = result.names

    # 先做合并框抑制：一个大框套着两个各自的小框时，丢掉大的那个
    # （塔防里敌人贴在一起走，NMS 压不掉这种"合并框"）
    from .postprocess import MergeSuppressor, dedupe

    raw = []
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
        raw.append((x1, y1, x2, y2, int(box.cls), float(box.conf),
                    int(box.id[0]) if getattr(box, "id", None) is not None else -1))

    merged = MergeSuppressor()([[item[0], item[1], item[2], item[3]] for item in raw])
    merged_keys = {(round(box[0]), round(box[1]), round(box[2]), round(box[3])) for box in merged}

    # 同一个 key 可能有多条（同框重复），只留置信度最高的
    survivors = []
    for item in sorted(raw, key=lambda entry: -entry[5]):
        key = (round(item[0]), round(item[1]), round(item[2]), round(item[3]))
        if key in merged_keys:
            survivors.append(item)
            merged_keys.discard(key)

    # 再去重：同一个目标的两个框（IoU 0.6+）只留置信度高的
    unique = dedupe([([item[0], item[1], item[2], item[3]], item[5]) for item in survivors])
    kept_pairs = {(round(box[0]), round(box[1]), round(box[2]), round(box[3])) for box, _ in unique}

    for x1, y1, x2, y2, cls, conf, track_id in survivors:
        if (round(x1), round(y1), round(x2), round(y2)) not in kept_pairs:
            continue
        units.append(
            {
                "track_id": track_id,
                "cls": names[cls],
                "conf": round(conf, 3),
                "box": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
            }
        )
    return units


def iter_units(scene_path: Path) -> Iterator[dict[str, Any]]:
    """逐帧读场景图。"""
    for line in Path(scene_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def count_units(scene_path: Path) -> dict[str, int]:
    """按类统计场景图里的 unit 数（快速体检用）。"""
    counts: dict[str, int] = {}
    for record in iter_units(scene_path):
        for unit in record.get("units", []):
            cls = unit["cls"]
            counts[cls] = counts.get(cls, 0) + 1
    return counts
