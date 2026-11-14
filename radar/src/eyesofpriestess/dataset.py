"""clicks -> YOLO 数据集（点经 SAM 变框，ignore 留档，身份裁剪图入库）。

产出（HANDOFF §4 / nightly_sam2yolo.py 的结构，修掉了几个坑）::

    dataset/images/{train,val}/<帧>.jpg
    dataset/labels/{train,val}/<帧>.txt     cls cx cy w h（归一化）
    dataset/labels_ignore/<帧>.txt           ignore 点备查（cls=99）
    dataset/crops/<编号>/<帧>_<n>.png        身份图库素材（按图鉴编号分目录）
    dataset/data.yaml                       YOLO 训练配置（绝对路径）
    dataset/report.json                     统计 + 报警

必须守住的规则：

* **按视频切分** train/val（相邻帧几乎一样，按帧随机 = 验证集泄漏）；
  只有一个视频时退化成"按时间切"，并显式报警。
* ignore 不进 labels（YOLO 没有 ignore 语义），但绝不能丢：写进 labels_ignore。
* 点与掩码严格一一对应：对不上的点记 ``missing``，**不**拿错位的框配类别。
* 标签不在 catalog 里（笔误 / 已删除的类如血条）→ 跳过并记进 ``skipped_labels``，
  **不**改成 enemy：那等于往训练集里灌错样本。
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import yaml

from .catalog import Catalog
from .frames import ManifestEntry, load_manifest
from .schema import Click, load_clicks
from .sam import Box, SamBoxer

#: ignore 点的圆形占位框边长（相对整幅宽/高）
IGNORE_BOX_SIDE = 0.05


@dataclass
class BuildOptions:
    val_ratio: float = 0.2
    #: 固定这些视频当验证集（None = 按 val_ratio 随机挑，加录像会换 val，指标不可比）
    val_videos: tuple[str, ...] | None = None
    seed: int = 42
    min_box_area: int = 20
    min_box_side: int = 3
    #: 给这些 YOLO 类裁身份图库素材
    crop_classes: tuple[str, ...] = ("enemy",)
    #: 只在战场区域里跑 SAM（相对整幅的比例 x0,y0,x1,y1）；None = 整帧
    region: tuple[float, float, float, float] | None = None
    #: 在 region 内再降采样，提速用（0.5 = 边长减半）
    downscale: float = 1.0
    #: 出框数 / 点数 低于该比例就报警（mobile_sam 对极小目标会失败）
    min_box_ratio: float = 0.8


@dataclass
class BuildReport:
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)
    class_hist: Counter = field(default_factory=Counter)
    label_hist: Counter = field(default_factory=Counter)
    points: int = 0
    boxes: int = 0
    missing: int = 0
    ignore: int = 0
    skipped_labels: Counter = field(default_factory=Counter)
    val_videos: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        ratio = (self.boxes / self.points) if self.points else 1.0
        ignore_rate = (self.ignore / (self.points + self.ignore)) if (self.points + self.ignore) else 0.0
        return {
            "frames": self.frames,
            "class_hist": dict(self.class_hist),
            "label_hist": dict(self.label_hist),
            "skipped_labels": dict(self.skipped_labels),
            "points": self.points,
            "boxes": self.boxes,
            "missing": self.missing,
            "ignore": self.ignore,
            "box_ratio": round(ratio, 4),
            "ignore_rate": round(ignore_rate, 4),
            "val_videos": self.val_videos,
            "warnings": self.warnings,
        }


def _split_by_video(
    entries: dict[str, ManifestEntry], val_ratio: float, seed: int, warnings: list[str]
) -> tuple[set[str], dict[str, str]]:
    """返回 (val 视频集合, 帧 -> split)。只有一个视频时按时间切并报警。"""
    videos = sorted({entry.video for entry in entries.values()})
    split_of: dict[str, str] = {}

    if len(videos) >= 2:
        import random

        count = max(1, int(round(len(videos) * val_ratio)))
        count = min(count, len(videos) - 1)          # 至少留一个训练视频
        val_videos = set(random.Random(seed).sample(videos, count))
        for frame, entry in entries.items():
            split_of[frame] = "val" if entry.video in val_videos else "train"
        return val_videos, split_of

    warning = (
        f"只有 {len(videos)} 个视频（{videos[0] if videos else '-'}），"
        "按 HANDOFF 的规矩 val 应该来自不同的视频；这里临时按时间切，"
        "请尽快补录第二个视频后重建数据集。"
    )
    warnings.append(warning)
    ordered = sorted(entries.values(), key=lambda entry: (entry.video, entry.frame_idx))
    cut = int(len(ordered) * (1.0 - val_ratio))
    for index, entry in enumerate(ordered):
        split_of[entry.frame] = "val" if index >= cut else "train"
    return {videos[0]} if videos else set(), split_of


def _crop_region(
    region: tuple[float, float, float, float] | None, width: int, height: int
) -> tuple[int, int, int, int] | None:
    if not region:
        return None
    x0, y0, x1, y1 = region
    return (
        int(round(x0 * width)),
        int(round(y0 * height)),
        int(round(x1 * width)),
        int(round(y1 * height)),
    )


def _yolo_line(cls_id: int, box: Box, width: int, height: int) -> str:
    cx = (box.x1 + box.x2) / 2 / width
    cy = (box.y1 + box.y2) / 2 / height
    w = box.width / width
    h = box.height / height
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build(
    *,
    clicks: Path,
    frames: Path,
    manifest_path: Path,
    out: Path,
    catalog: Catalog,
    boxer: SamBoxer,
    options: BuildOptions | None = None,
) -> BuildReport:
    """跑完整个 SAM -> YOLO 数据集体检，返回报告（同时写 report.json / data.yaml）。"""
    options = options or BuildOptions()
    report = BuildReport()

    manifest = load_manifest(manifest_path)
    if not manifest:
        raise RuntimeError(f"manifest 是空的: {manifest_path}（先跑 `radar extract`）")

    by_frame = load_clicks(clicks)
    if not by_frame:
        raise RuntimeError(f"{clicks} 里没有点击标注（先 `radar import-ls`）")

    class_ids = {name: index for index, name in enumerate(catalog.classes)}
    known_frames = {
        frame: entry
        for frame, entry in manifest.items()
        if (frames / frame).exists() and frame in by_frame
    }
    for frame in by_frame:
        if frame not in known_frames:
            report.warnings.append(f"跳过 {frame}：manifest 里没有，或帧文件不存在")

    if options.val_videos:
        explicit = {video for video in options.val_videos if video in {e.video for e in known_frames.values()}}
        if explicit:
            val_videos = explicit
            split_of = {frame: ("val" if entry.video in val_videos else "train")
                        for frame, entry in known_frames.items()}
        else:
            report.warnings.append(f"--val-videos 里没有 manifest 中的视频: {options.val_videos}")
            val_videos, split_of = _split_by_video(known_frames, options.val_ratio, options.seed, report.warnings)
    else:
        val_videos, split_of = _split_by_video(known_frames, options.val_ratio, options.seed, report.warnings)
    report.val_videos = sorted(val_videos)

    # 先清掉上一次的产物：不清的话旧 split 的图会留在 val 里（既过时又泄漏）
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "labels_ignore"):
        target = out / sub
        if target.exists():
            for path in target.iterdir():
                if path.is_file():
                    path.unlink()
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "labels_ignore", "crops"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    for frame in sorted(known_frames):
        entry = known_frames[frame]
        image_path = frames / frame
        image = cv2.imread(str(image_path))
        if image is None:
            report.warnings.append(f"读不到图: {image_path}")
            continue
        height, width = image.shape[:2]

        clicks_for_frame: list[Click] = by_frame[frame]
        points = [click for click in clicks_for_frame if not click.is_ignore]
        ignores = [click for click in clicks_for_frame if click.is_ignore]

        # 标签不在 catalog 里（笔误、或已经从类别表里删掉的，比如血条）→ 跳过并记账，
        # 绝不能悄悄改成 enemy：那等于往训练集里灌错样本。
        usable: list[Click] = []
        for click in points:
            if catalog.cls_for(click.cls or "") is None:
                report.skipped_labels[click.cls or "?"] += 1
                continue
            usable.append(click)
        points = usable

        report.points += len(points)
        report.ignore += len(ignores)

        lines: list[str] = []
        if points:
            boxes = boxer.boxes(
                image,
                [(click.x, click.y) for click in points],
                region=_crop_region(options.region, width, height),
                downscale=options.downscale,
            )
            for click, box in zip(points, boxes):
                cls_name = catalog.cls_for(click.cls or "")
                if box is None:
                    report.missing += 1
                    continue
                if box.area < options.min_box_area:
                    report.missing += 1
                    continue
                if box.width < options.min_box_side or box.height < options.min_box_side:
                    report.missing += 1
                    continue
                cls_id = class_ids[cls_name]
                lines.append(_yolo_line(cls_id, box, width, height))
                report.class_hist[cls_name] += 1
                report.label_hist[click.cls or "unknown"] += 1
                report.boxes += 1

                if cls_name in options.crop_classes:
                    crop = image[box.y1:box.y2, box.x1:box.x2]
                    if crop.size:
                        crop_dir = out / "crops" / str(click.cls or "unknown")
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        index = len(list(crop_dir.glob(f"{Path(frame).stem}_*.png")))
                        cv2.imwrite(str(crop_dir / f"{Path(frame).stem}_{index:03d}.png"), crop)

        split = split_of[frame]
        shutil.copy2(str(image_path), str(out / "images" / split / frame))
        label_path = out / "labels" / split / f"{Path(frame).stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")

        if ignores:
            ignore_lines = [
                f"99 {click.x / width:.6f} {click.y / height:.6f} "
                f"{IGNORE_BOX_SIDE:.6f} {IGNORE_BOX_SIDE:.6f}"
                for click in ignores
            ]
            (out / "labels_ignore" / f"{Path(frame).stem}.txt").write_text(
                "\n".join(ignore_lines), encoding="utf-8"
            )

        report.frames[frame] = {
            "split": split,
            "video": entry.video,
            "points": len(points),
            "boxes": len(lines),
            "ignore": len(ignores),
        }

    # 出框率报警：mobile_sam 对 <30px 的目标经常整片 mask 失败（HANDOFF §6.4）
    if report.points:
        ratio = report.boxes / report.points
        if ratio < options.min_box_ratio:
            report.warnings.append(
                f"出框率只有 {ratio:.0%}（{report.boxes}/{report.points}），"
                "低于 80%：先确认点击落在目标上，或考虑对战场区域降采样/换 sam2 权重"
            )
    if report.skipped_labels:
        names = "、".join(f"{name}×{count}" for name, count in report.skipped_labels.most_common(6))
        report.warnings.append(
            f"有 {sum(report.skipped_labels.values())} 个点的标签不在类别表里，已跳过: {names}"
            "（血条不再作为检测类别；若是认错编号，改标签后重新导出）"
        )
    if not report.frames:
        raise RuntimeError("没有任何帧进入数据集：检查 clicks 与 manifest 的帧名是否对得上")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {index: name for name, index in class_ids.items()},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (out / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
