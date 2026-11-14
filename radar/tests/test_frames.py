# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""抽帧：ffmpeg 管道解码、去重、限流、manifest、续跑。

合成小视频的画面很"平"（一块底色 + 一个方块），边缘密度天然低于真实战场，
所以除了专门测平坦过滤的用例，其余统一 ``min_edges=0`` 关掉那一层。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from eyesofpriestess.frames import (
    extract,
    ffmpeg_exe,
    list_videos,
    load_manifest,
    video_id,
)


def make_video(path, *, frames: int = 90, size=(320, 180), moving: bool = True, fps: float = 30):
    """写一段合成视频：``moving`` 时每帧方块在动，否则全静止。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened(), "打不开 VideoWriter"
    width, height = size
    for index in range(frames):
        frame = make_frame(index, width, height, moving)
        writer.write(frame)
    writer.release()
    return path


def make_frame(index: int, width: int, height: int, moving: bool = True) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    x = (index * 7) % (width - 20) if moving else 10
    cv2.rectangle(frame, (x, 30), (x + 20, 60), (0, 200, 255), -1)
    return frame


def test_extract_moving_video(tmp_path):
    video = make_video(tmp_path / "fight.mp4")
    out = tmp_path / "frames"

    entries = extract(video, out, max_frames=50, sample_fps=5.0, min_gap=0.1, min_edges=0)

    assert entries, "应当抽到帧"
    assert (out / "manifest.jsonl").exists()
    manifest = load_manifest(out / "manifest.jsonl")
    assert len(manifest) == len(entries)
    for entry in manifest.values():
        assert (out / entry.frame).exists()
        assert entry.w == 320 and entry.h == 180
        assert entry.frame.startswith("v000_")


def test_static_video_is_deduped(tmp_path):
    video = make_video(tmp_path / "static.mp4", frames=120, moving=False)
    entries = extract(video, tmp_path / "frames", min_gap=0.05, diff_thresh=10.0, min_edges=0)
    assert len(entries) <= 2, f"静止画面不该抽这么多帧: {len(entries)}"


def test_rerun_does_not_duplicate_manifest(tmp_path):
    video = make_video(tmp_path / "fight.mp4")
    out = tmp_path / "frames"

    first = extract(video, out, max_frames=20, min_gap=0.1, min_edges=0)
    assert first

    second = extract(video, out, max_frames=20, min_gap=0.1, min_edges=0)
    assert second == [], "同样的命令重跑应当幂等"

    manifest = load_manifest(out / "manifest.jsonl")
    assert len(manifest) == len(first)


def test_bigger_max_frames_extends_extraction(tmp_path):
    """调大 --max-frames 才继续抽（续跑不重复、不丢帧）。"""
    video = make_video(tmp_path / "fight.mp4", frames=150)
    out = tmp_path / "frames"

    assert len(extract(video, out, max_frames=5, min_gap=0.05, min_edges=0)) == 5
    extra = extract(video, out, max_frames=10, min_gap=0.05, min_edges=0)
    assert len(extra) == 5

    manifest = load_manifest(out / "manifest.jsonl")
    assert len(manifest) == 10
    assert len({entry.frame for entry in manifest.values()}) == 10


def test_black_frames_are_skipped(tmp_path):
    """黑屏/加载画面不抽（标了也没用），但视频后面的正常帧要照抽。"""
    video = tmp_path / "mixed.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 90))
    for index in range(90):
        if index < 45:
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
        else:
            frame = np.full((90, 160, 3), 90, dtype=np.uint8)
            cv2.circle(frame, (20 + index, 40), 6, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    entries = extract(video, tmp_path / "frames", min_gap=0.05, min_brightness=5.0, min_edges=0)

    assert entries
    assert all(entry.frame_idx > 45 for entry in entries)


def test_flat_frames_are_skipped(tmp_path):
    """没有战场的画面（加载页/编队页）边缘密度低，不抽。"""
    video = tmp_path / "flat.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 180))
    for index in range(90):
        if index < 45:
            # 加载页：一整块渐变 + 一个 logo 方框，边缘很少
            frame = np.full((180, 320, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (120, 60), (200, 120), (60, 60, 60), -1)
        else:
            # 战场：满屏棋盘格线条，边缘密度高
            frame = np.full((180, 320, 3), 40, dtype=np.uint8)
            for x in range(0, 320, 10):
                cv2.line(frame, (x, 0), (x, 180), (200, 200, 200), 1)
            for y in range(0, 180, 10):
                cv2.line(frame, (0, y), (320, y), (200, 200, 200), 1)
            cv2.circle(frame, (30 + index, 90), 8, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()

    entries = extract(video, tmp_path / "frames", min_gap=0.05, min_brightness=0.0)

    assert entries
    assert all(entry.frame_idx > 45 for entry in entries)


def test_video_id_is_ascii_and_stable(tmp_path):
    assert video_id(tmp_path / "1-7实战.mp4", 3) == "v003_1-7"
    fallback = video_id(tmp_path / "实战.mp4", 0)
    assert fallback.startswith("v000_clip-")
    assert fallback.isascii()


def test_list_videos_filters_suffix(tmp_path):
    make_video(tmp_path / "a.mp4")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    assert [path.name for path in list_videos(tmp_path)] == ["a.mp4"]


def test_cv2_fallback_when_ffmpeg_missing(tmp_path, monkeypatch):
    """没有 ffmpeg 的机器照样能抽帧（走 cv2 逐帧）。"""
    video = make_video(tmp_path / "fight.mp4", frames=60)
    monkeypatch.setattr("eyesofpriestess.frames.ffmpeg_exe", lambda: None)

    entries = extract(video, tmp_path / "frames", max_frames=10, min_gap=0.1, min_edges=0)

    assert entries
    assert all((tmp_path / "frames" / entry.frame).exists() for entry in entries)


def test_ffmpeg_is_available_in_this_env():
    """imageio-ffmpeg 是依赖，正常环境里一定有 exe。"""
    assert ffmpeg_exe(), "找不到 ffmpeg，检查 imageio-ffmpeg 是否装好"
