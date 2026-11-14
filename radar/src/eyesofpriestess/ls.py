"""Label Studio 这一段：生成任务、把导出转成 clicks。

LS 的坑（HANDOFF §6.1）：图片若走 LS 的 Cloud 转存，导出里的 URL 与 manifest
对不上，转换器会大面积 skip。所以这里只支持**本地文件**两种形态：

1. ``/data/local-files/?d=frames/xxx.jpg``（`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED`
   打开时的本地存储 URL）；
2. 绝对路径（直接导入绝对路径的任务 JSON）。

坐标换算（HANDOFF §2.6）：LS 里 keypoint 的 x/y 是**百分比**，转回像素必须乘
manifest 里的真实宽高，而不是当前图片的像素——两者在缩放过的预览里会不一样。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, quote, unquote, urlparse

from .catalog import Catalog
from .frames import ManifestEntry, load_manifest
from .schema import POINT, IGNORE, Click, write_clicks

#: LS 里 keypoint 的 x/y 是 0~100 的百分比
PERCENT = 100.0


def frame_name_from_url(url: str) -> str:
    """从 LS 的图片地址里取出帧文件名（兼容 local-files 与普通路径/URL）。"""
    if not url:
        return ""
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "d" in query:                       # /data/local-files/?d=frames%2Fxxx.jpg
        candidate = unquote(query["d"][0])
    else:
        candidate = unquote(parsed.path or url)
    return Path(candidate.replace("\\", "/")).name


def build_tasks(
    manifest: dict[str, ManifestEntry],
    *,
    frames_dir: Path,
    local_prefix: str = "frames",
    absolute: bool = False,
) -> list[dict[str, Any]]:
    """生成 Label Studio 导入用的任务列表。

    Args:
        manifest: 帧清单
        frames_dir: 帧实际所在目录
        local_prefix: ``DOCUMENT_ROOT`` 下到帧目录的相对路径（默认 ``frames``）
        absolute: True 时写绝对路径（适合不走 local storage 的小批量导入）
    """
    frames_dir = Path(frames_dir)
    tasks: list[dict[str, Any]] = []
    for frame in sorted(manifest):
        entry = manifest[frame]
        if absolute:
            image = str((frames_dir / frame).resolve())
        else:
            relative = f"{local_prefix.rstrip('/')}/{frame}"
            image = f"/data/local-files/?d={quote(relative)}"
        tasks.append({"data": {"image": image, "frame": frame, "video": entry.video}})
    return tasks


def _result_labels(value: dict[str, Any]) -> list[str]:
    """LS 的 KeyPointLabels 用 ``keypointlabels``；老版本/其它控件用 ``labels``。"""
    raw = value.get("keypointlabels") or value.get("labels") or []
    return [str(item) for item in raw]


def convert(
    export_path: Path,
    manifest_path: Path,
    out_dir: Path,
    *,
    catalog: Catalog,
    operator: str = "",
) -> dict[str, Any]:
    """LS 导出 JSON -> ``clicks/*.jsonl``。

    普通标签 -> ``type:"point"``；``IGNORE-occluded`` / ``IGNORE-blurry`` ->
    ``type:"ignore"`` 且带 reason。返回统计（转换数 / 跳过数 / 未知标签）。
    """
    manifest = load_manifest(manifest_path)
    if not manifest:
        raise RuntimeError(f"manifest 是空的: {manifest_path}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    export = json.loads(Path(export_path).read_text(encoding="utf-8"))
    if isinstance(export, dict):           # 有些导出是 {"tasks": [...]}
        export = export.get("tasks") or export.get("data") or []

    stats = {"clicks": 0, "skip_task": 0, "unknown_labels": {}, "frames": {}}
    unknown: dict[str, int] = stats["unknown_labels"]  # type: ignore[assignment]

    for task in export:
        data = task.get("data") or {}
        frame = str(data.get("frame") or "") or frame_name_from_url(str(data.get("image") or ""))
        entry = manifest.get(frame)
        if entry is None:
            stats["skip_task"] += 1
            continue
        width, height = entry.w, entry.h

        annotations = task.get("annotations") or []
        if not annotations and task.get("result"):
            annotations = [task]
        recorded: list[Click] = []
        for annotation in annotations:
            op = operator or str(annotation.get("completed_by") or annotation.get("id") or "")
            timestamp = str(annotation.get("created_at") or "") or None
            for result in annotation.get("result") or []:
                if result.get("type") not in ("keypointlabels", "keypoint"):
                    continue
                value = result.get("value") or {}
                labels = _result_labels(value)
                if not labels:
                    continue
                label = labels[0]
                if "x" not in value or "y" not in value:
                    continue
                x = float(value["x"]) / PERCENT * width
                y = float(value["y"]) / PERCENT * height
                if label.startswith("IGNORE-"):
                    click = Click(
                        frame=frame, x=x, y=y, cls=None, type=IGNORE,
                        reason=label.split("-", 1)[1], op=op,
                    )
                else:
                    click = Click(frame=frame, x=x, y=y, cls=label, type=POINT, op=op)
                if catalog.unknown_label(label):
                    unknown[label] = unknown.get(label, 0) + 1
                if timestamp:
                    click.ts = timestamp
                recorded.append(click.validate())

        if not recorded:
            stats["skip_task"] += 1
            continue
        write_clicks(out_dir / f"{Path(frame).stem}.jsonl", recorded, append=False)
        stats["clicks"] += len(recorded)
        stats["frames"][frame] = len(recorded)

    return stats


def missing_frames(clicks_dir: Path, manifest: dict[str, ManifestEntry]) -> list[str]:
    """clicks 里有、manifest 里没有的帧（排查导出/存储配置问题用）。"""
    names = {path.stem + ".jpg" for path in Path(clicks_dir).glob("*.jsonl")}
    return sorted(name for name in names if name not in manifest)
