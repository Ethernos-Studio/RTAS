# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""标签目录 / Label Studio 桥接 / 数据集构建。"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from eyesofpriestess.catalog import Catalog, load_catalog
from eyesofpriestess.dataset import BuildOptions, build
from eyesofpriestess.frames import ManifestEntry
from eyesofpriestess.ls import build_tasks, convert, frame_name_from_url
from eyesofpriestess.sam import Box
from eyesofpriestess.schema import Click, write_clicks


@pytest.fixture
def catalog() -> Catalog:
    return load_catalog(Path(__file__).resolve().parents[1] / "configs" / "enemies.json")


# -- catalog --------------------------------------------------------------
def test_catalog_maps_codes_to_yolo_classes(catalog):
    assert catalog.cls_for("B1") == "enemy"
    assert catalog.cls_for("ally") == "ally"
    assert catalog.cls_for("IGNORE-occluded") is None
    assert catalog.unknown_label("XYZ") is True
    assert catalog.unknown_label("B1") is False


def test_catalog_interface_contains_codes_and_hotkeys(catalog):
    xml = catalog.interface_xml()
    assert 'value="B1" hotkey="a"' in xml
    assert 'value="IGNORE-occluded"' in xml
    assert 'value="ally"' in xml
    assert xml.count("<Label") == len(catalog.labels)


def test_catalog_rejects_bad_class():
    catalog = Catalog({"classes": ["enemy"], "enemies": {"B1": {"cls": "not_a_class"}}})
    with pytest.raises(ValueError):
        catalog.cls_for("B1")


# -- LS 桥接 ---------------------------------------------------------------
def test_frame_name_from_local_files_url():
    url = "/data/local-files/?d=frames%2Fv000_1-7_f000123.jpg"
    assert frame_name_from_url(url) == "v000_1-7_f000123.jpg"
    assert frame_name_from_url("C:/data/frames/v000_1-7_f000123.jpg") == "v000_1-7_f000123.jpg"


def test_tasks_point_to_local_storage(tmp_path):
    manifest = {
        "v000_a_f000001.jpg": ManifestEntry("v000_a_f000001.jpg", "v000_a", 1, 0.03, 1264, 711)
    }
    tasks = build_tasks(manifest, frames_dir=tmp_path / "frames", local_prefix="frames")
    assert tasks[0]["data"]["image"] == "/data/local-files/?d=frames/v000_a_f000001.jpg"

    absolute = build_tasks(manifest, frames_dir=tmp_path / "frames", absolute=True)
    assert absolute[0]["data"]["image"].endswith("v000_a_f000001.jpg")
    assert str(tmp_path) in absolute[0]["data"]["image"]


def _ls_export(frame: str, points: list[tuple[float, float, str]]) -> list[dict]:
    results = [
        {
            "type": "keypointlabels",
            "value": {"x": x, "y": y, "width": 0.5, "keypointlabels": [label]},
        }
        for x, y, label in points
    ]
    return [{"data": {"image": f"/data/local-files/?d=frames/{frame}", "frame": frame},
             "annotations": [{"completed_by": 7, "created_at": "2026-11-13T10:00:00Z",
                              "result": results}]}]


def test_convert_percent_to_pixels(tmp_path, catalog):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"frame": "v000_a_f000001.jpg", "video": "v000_a", "frame_idx": 1,
                    "t": 0.03, "w": 1000, "h": 500}) + "\n",
        encoding="utf-8",
    )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(_ls_export("v000_a_f000001.jpg",
                              [(50.0, 50.0, "B1"), (20.0, 40.0, "IGNORE-blurry")]),
                   ensure_ascii=False),
        encoding="utf-8",
    )

    stats = convert(export, manifest, tmp_path / "clicks", catalog=catalog)

    assert stats["clicks"] == 2
    records = [
        json.loads(line)
        for line in (tmp_path / "clicks" / "v000_a_f000001.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    point, ignore = records
    assert (point["x"], point["y"]) == (500, 250)      # 百分比 -> 帧分辨率像素
    assert point["cls"] == "B1" and point["type"] == "point"
    assert ignore["type"] == "ignore" and ignore["reason"] == "blurry"
    assert ignore["x"] == 200 and ignore["y"] == 200
    assert point["op"] == "7"


def test_convert_skips_frames_outside_manifest(tmp_path, catalog):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"frame": "known.jpg", "video": "v000", "frame_idx": 1, "t": 0.0,
                    "w": 100, "h": 100}) + "\n",
        encoding="utf-8",
    )
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_ls_export("unknown.jpg", [(10.0, 10.0, "B1")])),
                      encoding="utf-8")

    stats = convert(export, manifest, tmp_path / "clicks", catalog=catalog)
    assert stats["clicks"] == 0 and stats["skip_task"] == 1


def test_convert_flags_unknown_labels(tmp_path, catalog):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"frame": "f.jpg", "video": "v", "frame_idx": 1, "t": 0.0,
                    "w": 100, "h": 100}) + "\n", encoding="utf-8")
    export = tmp_path / "export.json"
    export.write_text(json.dumps(_ls_export("f.jpg", [(10.0, 10.0, "ZZ9")])), encoding="utf-8")

    stats = convert(export, manifest, tmp_path / "clicks", catalog=catalog)
    assert stats["unknown_labels"] == {"ZZ9": 1}


# -- 数据集 ---------------------------------------------------------------
class FakeBoxer:
    """不跑 SAM 的假框器：点在哪就返回一个 12x10 的框。"""

    def __init__(self, size: tuple[int, int] = (12, 10), fail_every: int = 0):
        self.size = size
        self.fail_every = fail_every
        self.calls = 0

    def boxes(self, image, points, *, region=None, downscale=1.0):
        self.calls += 1
        out = []
        for index, (x, y) in enumerate(points):
            if self.fail_every and index % self.fail_every == 0:
                out.append(None)
                continue
            width, height = self.size
            out.append(Box(int(x), int(y), int(x) + width, int(y) + height,
                           area=width * height))
        return out


def _make_frames(root: Path, videos: dict[str, int], *, size=(320, 180)) -> dict[str, ManifestEntry]:
    root.mkdir(parents=True, exist_ok=True)
    entries: dict[str, ManifestEntry] = {}
    for video, count in videos.items():
        for index in range(1, count + 1):
            name = f"{video}_f{index:06d}.jpg"
            image = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
            cv2.circle(image, (20 + index * 3, 40), 6, (0, 180, 255), -1)
            cv2.imwrite(str(root / name), image)
            entries[name] = ManifestEntry(name, video, index, index / 30.0, size[0], size[1])
    (root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(entry.to_dict(), ensure_ascii=False) for entry in entries.values()) + "\n",
        encoding="utf-8",
    )
    return entries


def _write_clicks(clicks_dir: Path, entries: dict[str, ManifestEntry]) -> None:
    for index, frame in enumerate(entries):
        write_clicks(
            clicks_dir / f"{Path(frame).stem}.jsonl",
            [
                Click(frame=frame, x=40 + index, y=40, cls="B1"),
                Click(frame=frame, x=80, y=60, cls="ally"),
                Click(frame=frame, x=120, y=70, type="ignore", reason="occluded"),
            ],
            append=False,
        )


def test_build_creates_yolo_dataset(tmp_path, catalog):
    frames = tmp_path / "frames"
    entries = _make_frames(frames, {"v000_a": 6, "v001_b": 6})
    _write_clicks(tmp_path / "clicks", entries)

    report = build(
        clicks=tmp_path / "clicks", frames=frames, manifest_path=frames / "manifest.jsonl",
        out=tmp_path / "dataset", catalog=catalog, boxer=FakeBoxer(),
    )
    data = report.to_dict()

    dataset = tmp_path / "dataset"
    assert (dataset / "images/train").is_dir() and (dataset / "labels/train").is_dir()
    assert data["boxes"] == data["points"] == 24        # 12 帧 × 2 个正样本
    assert data["ignore"] == 12
    assert data["box_ratio"] == 1.0

    # 按视频切分：val 里的帧必须来自另一个视频
    assert len(data["val_videos"]) == 1
    val_video = data["val_videos"][0]
    val_images = list((dataset / "images/val").glob("*.jpg"))
    assert val_images
    assert all(path.name.startswith(val_video) for path in val_images)
    assert not any(path.name.startswith(val_video) for path in (dataset / "images/train").glob("*.jpg"))

    # ignore 不进 labels，但要有备查文件
    label_text = next((dataset / "labels/train").glob("*.txt")).read_text(encoding="utf-8")
    assert "99 " not in label_text
    ignore_text = next((dataset / "labels_ignore").glob("*.txt")).read_text(encoding="utf-8")
    assert ignore_text.startswith("99 ")

    yaml_text = (dataset / "data.yaml").read_text(encoding="utf-8")
    assert "enemy" in yaml_text and "ally" in yaml_text and "tile" in yaml_text
    assert "hp_bar" not in yaml_text      # 血条不进检测类别（交给 CV）

    # 身份图库素材按图鉴编号分目录
    assert (dataset / "crops" / "B1").is_dir()


def test_build_reports_missing_boxes(tmp_path, catalog):
    frames = tmp_path / "frames"
    entries = _make_frames(frames, {"v000_a": 3, "v001_b": 3})
    _write_clicks(tmp_path / "clicks", entries)

    report = build(
        clicks=tmp_path / "clicks", frames=frames, manifest_path=frames / "manifest.jsonl",
        out=tmp_path / "dataset", catalog=catalog, boxer=FakeBoxer(fail_every=2),
    )
    data = report.to_dict()

    assert data["missing"] > 0
    assert data["box_ratio"] < 0.8
    assert any("出框率" in warning for warning in data["warnings"])


def test_build_clears_stale_outputs(tmp_path, catalog):
    """重建时要把上一次的图/标签清掉，否则旧 split 的图会留在 val 里（过时 + 泄漏）。"""
    frames = tmp_path / "frames"
    entries = _make_frames(frames, {"v000_a": 3, "v001_b": 3})
    _write_clicks(tmp_path / "clicks", entries)
    out = tmp_path / "dataset"
    stale = out / "images" / "val" / "stale_old_frame.jpg"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old")

    build(clicks=tmp_path / "clicks", frames=frames,
          manifest_path=frames / "manifest.jsonl", out=out, catalog=catalog, boxer=FakeBoxer())

    assert not stale.exists()
    val_images = {path.name for path in (out / "images/val").glob("*.jpg")}
    assert all(name.startswith("v001_b") or name.startswith("v000_a") for name in val_images)
    assert val_images


def test_single_video_falls_back_to_time_split(tmp_path, catalog):
    frames = tmp_path / "frames"
    entries = _make_frames(frames, {"v000_only": 10})
    _write_clicks(tmp_path / "clicks", entries)

    report = build(
        clicks=tmp_path / "clicks", frames=frames, manifest_path=frames / "manifest.jsonl",
        out=tmp_path / "dataset", catalog=catalog, boxer=FakeBoxer(),
        options=BuildOptions(val_ratio=0.2),
    )
    data = report.to_dict()

    assert (tmp_path / "dataset/images/val").exists()
    assert any("只有 1 个视频" in warning for warning in data["warnings"])


# -- CLI ---------------------------------------------------------------
def test_rules_to_html_renders_headings_bullets_bold():
    """LS 的 Instructions 按 HTML 渲染，规矩要转成 HTML 才能正常显示。"""
    from eyesofpriestess.cli import rules_to_html

    html = rules_to_html(
        "# 标题\n\n一句话：**重点**\n\n## 小节\n- 第一条\n- 第二条\n"
    )

    assert "<h2>标题</h2>" in html
    assert "<h3>小节</h3>" in html
    assert "<b>重点</b>" in html
    assert "<ul>" in html and html.count("<li>") == 2
    assert "**" not in html and not html.lstrip().startswith("#")


def test_rules_to_html_escapes_markup():
    """正文里的尖括号要转义，别把规矩写成标签。"""
    from eyesofpriestess.cli import rules_to_html

    html = rules_to_html("- 目标 < 20px 且 a<b 时\n")
    assert "&lt;" in html and "<b>" not in html


def test_ls_serve_tells_you_to_install_when_missing(monkeypatch, capsys):
    from eyesofpriestess.cli import build_parser, cmd_ls_serve

    monkeypatch.setattr("shutil.which", lambda name: None)
    args = build_parser().parse_args(["ls-serve"])
    assert cmd_ls_serve(args) == 1
    assert "uv tool install label-studio" in capsys.readouterr().err


def test_dataset_yaml_uses_catalog_classes(tmp_path, catalog):
    """data.yaml 的 names 必须与 catalog.classes 顺序一致（YOLO 类别 id 不能错位）。"""
    import yaml

    frames = tmp_path / "frames"
    entries = _make_frames(frames, {"v000_a": 2, "v001_b": 2})
    _write_clicks(tmp_path / "clicks", entries)
    build(
        clicks=tmp_path / "clicks", frames=frames, manifest_path=frames / "manifest.jsonl",
        out=tmp_path / "dataset", catalog=catalog, boxer=FakeBoxer(),
    )

    config = yaml.safe_load((tmp_path / "dataset/data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {index: name for index, name in enumerate(catalog.classes)}
    assert config["path"] == str((tmp_path / "dataset").resolve())
