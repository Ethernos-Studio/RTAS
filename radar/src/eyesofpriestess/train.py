"""训练 / 导出 / 验证：ultralytics YOLO 的薄封装。

固定住 HANDOFF §2.8 定的那几件事，避免每次手打参数走样：

* ``imgsz=1280``（敌人最小约 40px，640 会直接漏检）；
* ``freeze=10`` 冻 backbone（小数据防过拟合）+ mosaic 拉满、后 20 轮关掉；
* 数据配置用 ``dataset/data.yaml``（绝对路径，换 cwd 不会炸）。

命令行之外的细节（学习率、优化器）保留可覆盖参数，默认值就是 train_yolo.sh 里那套。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

DATASET_YAML = "data.yaml"


def _load_yolo(weights: str) -> Any:
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - 延迟导入
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            "训练需要 ultralytics：`uv sync --extra train` 或 `pip install ultralytics`"
        ) from exc
    return YOLO(weights)


def check_dataset(dataset_dir: Path) -> dict[str, Any]:
    """训练前体检：train/val 目录非空、类别数与 data.yaml 对得上。"""
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / DATASET_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(f"缺 {data_yaml}（先跑 `radar build`）")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(config.get("path") or dataset_dir)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        relative = config.get(split) or f"images/{split}"
        images_dir = Path(relative) if Path(relative).is_absolute() else root / relative
        labels_dir = images_dir.parent.parent / "labels" / split
        counts[split] = {
            "images": len([p for p in images_dir.glob("*.*")]) if images_dir.exists() else 0,
            "labels": len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0,
        }
    if counts["train"]["images"] == 0:
        raise RuntimeError(f"{root}/images/train 是空的，没法训练")
    if counts["val"]["images"] == 0:
        raise RuntimeError(f"{root}/images/val 是空的，没法验证")
    return {
        "data": str(data_yaml),
        "root": str(root),
        "counts": counts,
        "names": config.get("names"),
    }


def train(
    dataset_dir: Path,
    *,
    weights: str = "yolov8s.pt",
    epochs: int = 200,
    imgsz: int = 1280,
    batch: int = 8,
    freeze: int = 10,
    patience: int = 60,
    project: Path | None = None,
    name: str = "v0",
    device: str | None = None,
    resume: bool = False,
    extra: dict[str, Any] | None = None,
) -> Any:
    """开训；返回 ultralytics 的 results（含 save_dir）。"""
    dataset_dir = Path(dataset_dir)
    info = check_dataset(dataset_dir)
    model = _load_yolo(weights)

    # ultralytics 会把相对 project 拼到它自己的 runs_dir 下面（可能是别的项目目录），
    # 所以这里一律转成绝对路径，保证产物落在 radar/runs/arknights/<name>
    project_path = (
        Path(project).resolve() if project else (dataset_dir.parent / "runs" / "arknights").resolve()
    )

    kwargs: dict[str, Any] = dict(
        data=str(Path(info["data"])),
        imgsz=imgsz,
        epochs=epochs,
        patience=patience,
        batch=batch,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        freeze=freeze,
        mosaic=1.0,
        close_mosaic=20,
        fliplr=0.5,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=str(project_path),
        name=name,
        exist_ok=True,
        seed=42,
        resume=resume,
        plots=True,
        verbose=True,
    )
    if device:
        kwargs["device"] = device
    if extra:
        kwargs.update(extra)
    model.train(**kwargs)
    return model


def export(weights: Path | str, *, fmt: str = "onnx", imgsz: int = 1280, half: bool = False) -> Path:
    """导出 ONNX（兜底）或 TensorRT engine；返回产物路径。"""
    model = _load_yolo(str(weights))
    produced = model.export(format=fmt, imgsz=imgsz, half=half)
    return Path(str(produced))


def validate(
    weights: Path | str,
    dataset_dir: Path,
    *,
    imgsz: int = 1280,
    split: str = "val",
    device: str | None = None,
) -> dict[str, Any]:
    """在 val 上跑一遍，返回 metrics 摘要（mAP50 / mAP50-95 / 每类）。"""
    info = check_dataset(dataset_dir)
    model = _load_yolo(str(weights))
    kwargs: dict[str, Any] = dict(data=str(Path(info["data"])), imgsz=imgsz, split=split, verbose=True)
    if device:
        kwargs["device"] = device
    metrics = model.val(**kwargs)
    box = metrics.box
    summary: dict[str, Any] = {
        # ultralytics 的字段名是 map50 / map / mp / mr（mAP 不是 mp50）
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(getattr(box, "mp", 0.0) or 0.0),
        "recall": float(getattr(box, "mr", 0.0) or 0.0),
    }
    maps = getattr(box, "maps", None)
    if maps is not None:
        summary["per_class_map50_95"] = [float(value) for value in maps]
    names = getattr(metrics, "names", None)
    if names:
        summary["names"] = names
    return summary


def copy_weights_to(weights: Path, destination_dir: Path) -> Path:
    """把 best.pt 复制到 weights/，便于固定版本号引用。"""
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / Path(weights).name
    shutil.copy2(weights, target)
    return target
