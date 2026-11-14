"""视频 -> 可标注帧（原分辨率、去重、限流），并维护 ``manifest.jsonl``。

规则（HANDOFF §2.2 / §6）：

* 原分辨率输出 JPEG，**不做任何缩放**（小目标只有 40px，一缩就没了）；
* 感知哈希 + 帧差双判去重：画面几乎不变的连续帧只留第一张；
* 限流：相邻标注帧至少隔 ``min_gap`` 秒，防止特效密集段刷屏；
* manifest 只增不改，重跑时自动跳过已抽过的帧（续跑）。

解码走 **ffmpeg 管道**（``imageio-ffmpeg`` 自带静态 exe）：先用 ``fps`` 滤镜把候选帧
降到 ``sample_fps``，Python 只处理这 1/N 的帧——实测 1920x1080 的 100s 录像，
cv2 逐帧 43s，ffmpeg 3fps 管道 8.4s（5x）。没有 ffmpeg 时自动退回 cv2 逐帧。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

FRAME_SUFFIXES = (".mp4", ".mkv", ".flv", ".mov", ".avi", ".webm", ".ts")

#: 整帧平均亮度低于它 → 黑屏/加载画面，不值得标（这类帧大量存在的视频多半是录制时窗口被遮）
MIN_BRIGHTNESS = 5.0

#: 默认候选帧率：3fps 足够让去重挑出画面变化，又比逐帧快 5 倍
SAMPLE_FPS = 3.0
#: 相邻标注帧的最小间隔（秒）
MIN_GAP = 0.33
#: 边缘密度低于它 → 加载页/编队页/过场这类"没战场"的画面（实测战场 0.038~0.042，加载页 0.007）
MIN_EDGES = 0.015


@dataclass
class ManifestEntry:
    """manifest 一行：帧文件 + 来源 + 时间 + 分辨率（转换器要用 w/h 还原像素坐标）。"""

    frame: str
    video: str
    frame_idx: int
    t: float
    w: int
    h: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VideoStream:
    """ffmpeg 解码出来的候选帧流。"""

    fps: float          # 源视频帧率
    width: int
    height: int
    duration: float     # 秒（ffmpeg 不知道时是 0）
    frames: Iterator[tuple[float, np.ndarray]]   # (时间秒, BGR 帧)
    close: object       # 关闭底层生成器/子进程


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    """读 manifest；同帧重复行以最后一行为准。"""
    path = Path(path)
    entries: dict[str, ManifestEntry] = {}
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                entries[raw["frame"]] = ManifestEntry(
                    frame=raw["frame"],
                    video=raw["video"],
                    frame_idx=int(raw["frame_idx"]),
                    t=float(raw["t"]),
                    w=int(raw["w"]),
                    h=int(raw["h"]),
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return entries


def list_videos(src: Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    return sorted(
        path for path in src.rglob("*") if path.suffix.lower() in FRAME_SUFFIXES
    )


def video_id(path: Path, index: int) -> str:
    """ASCII 安全的视频 id：``v000_1-7`` / ``v001_clip`` / ``v002_ab12cd34``。

    Windows 上中文/空格路径在部分工具链会炸（HANDOFF §6.7），所以这里只留
    ``[A-Za-z0-9_-]``，全被滤掉时用文件名哈希兜底。
    """
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(path).stem).strip("-_")[:20]
    if not slug:
        digest = hashlib.md5(Path(path).stem.encode("utf-8")).hexdigest()[:8]
        slug = f"clip-{digest}"
    return f"v{index:03d}_{slug}"


def phash(frame: np.ndarray) -> np.ndarray:
    """64 位感知哈希（DCT 低频），对整体亮度/轻微压缩不敏感。"""
    gray = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8]
    reference = np.median(dct[1:, 1:])
    return (dct > reference).flatten()


def edge_ratio(frame: np.ndarray) -> float:
    """Canny 边缘像素占比：战场画面信息量大，加载页/编队页很"平"。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges)) / edges.size


def ffmpeg_exe() -> str | None:
    """找 ffmpeg：环境变量 ``RADAR_FFMPEG`` > imageio-ffmpeg 自带 > PATH。"""
    override = os.environ.get("RADAR_FFMPEG")
    if override and Path(override).is_file():
        return override
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # noqa: PLC0415 - 可选依赖

        return get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 没装就退回 cv2
        pass
    return shutil.which("ffmpeg")


def open_video(
    video: Path,
    *,
    sample_fps: float = SAMPLE_FPS,
    hwaccel: bool = False,
    start: float = 0.0,
) -> VideoStream:
    """用 ffmpeg 打开视频，按 ``sample_fps`` 吐候选帧；没有 ffmpeg 时逐帧兜底。"""
    exe = ffmpeg_exe()
    if exe:
        try:
            return _open_with_ffmpeg(exe, Path(video), sample_fps=sample_fps,
                                     hwaccel=hwaccel, start=start)
        except Exception as exc:  # noqa: BLE001 - 解码器不认这个文件就退回 cv2
            print(f"[warn] ffmpeg 打开失败，退回 cv2 逐帧: {exc}")
    return _open_with_cv2(Path(video), sample_fps=sample_fps, start=start)


def _open_with_ffmpeg(
    exe: str, video: Path, *, sample_fps: float, hwaccel: bool, start: float
) -> VideoStream:
    from imageio_ffmpeg import read_frames  # noqa: PLC0415

    input_params: list[str] = []
    if start > 0:
        input_params += ["-ss", f"{start:.3f}"]     # 先定位再解码，续跑很快
    if hwaccel:
        input_params += ["-hwaccel", "auto"]

    reader = read_frames(
        str(video),
        pix_fmt="bgr24",
        input_params=input_params,
        output_params=["-vf", f"fps={sample_fps}"],
    )
    meta = next(reader)
    width, height = meta["size"]
    frame_bytes = int(meta.get("bytes_per_frame") or width * height * 3)
    fps = float(meta.get("fps") or 0.0) or float(sample_fps)
    duration = float(meta.get("duration") or 0.0)
    output_fps = float(sample_fps)

    def generator() -> Iterator[tuple[float, np.ndarray]]:
        index = 0
        try:
            for chunk in reader:
                frame = np.frombuffer(chunk, np.uint8, count=frame_bytes)
                if frame.size != width * height * 3:
                    continue
                yield start + index / output_fps, frame.reshape(height, width, 3)
                index += 1
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

    frames = generator()
    return VideoStream(
        fps=fps, width=width, height=height, duration=duration,
        frames=frames, close=frames.close,
    )


def _open_with_cv2(video: Path, *, sample_fps: float, start: float) -> VideoStream:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    step = max(1, int(round(fps / max(sample_fps, 0.1))))

    def generator() -> Iterator[tuple[float, np.ndarray]]:
        index = int(round(start * fps))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                index += 1
                if (index - int(round(start * fps))) % step == 0:
                    yield index / fps, frame
        finally:
            cap.release()

    frames = generator()
    return VideoStream(fps=fps, width=width, height=height,
                       duration=(total / fps if fps else 0.0),
                       frames=frames, close=frames.close)


def _frame_name(vid: str, index: int) -> str:
    return f"{vid}_f{index:06d}.jpg"


def extract_video(
    video: Path,
    out_dir: Path,
    *,
    video_index: int = 0,
    max_frames: int = 400,
    diff_thresh: float = 10.0,
    hash_max: int = 6,
    sample_fps: float = SAMPLE_FPS,
    min_gap: float = MIN_GAP,
    jpg_quality: int = 95,
    min_brightness: float = MIN_BRIGHTNESS,
    min_edges: float = MIN_EDGES,
    hwaccel: bool = False,
    existing: dict[str, ManifestEntry] | None = None,
) -> list[ManifestEntry]:
    """抽一个视频的帧，返回**新增**的 manifest 条目。

    续跑语义：manifest 里已经有这个视频的 ``N`` 帧时，先 seek 到最后一帧之后，
    抽到 ``max_frames`` 为止（也就是 ``max_frames`` 是"这个视频总共要多少帧"）。
    因此同样的命令重跑是幂等的（已经抽满就跳过）；想多抽就调大 ``max_frames``。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = existing or {}
    vid = video_id(video, video_index)
    known = [entry for entry in existing.values() if entry.video == vid]
    if len(known) >= max_frames:
        print(f"[{vid}] -> 已抽满 {len(known)} 帧（max={max_frames}），跳过")
        return []
    resume_after = max((entry.frame_idx for entry in known), default=0)
    quota = max_frames - len(known)

    stream = open_video(
        video, sample_fps=sample_fps, hwaccel=hwaccel,
        start=(resume_after / stream_fps_guess(known, resume_after)) if resume_after else 0.0,
    )
    entries: list[ManifestEntry] = []
    prev_small: np.ndarray | None = None
    prev_hash: np.ndarray | None = None
    last_take = -10**9
    blank_skipped = 0
    flat_skipped = 0
    taken = 0

    try:
        for timestamp, frame in stream.frames:
            index = max(1, int(round(timestamp * stream.fps)))
            small = cv2.resize(frame, (160, 90))
            digest = phash(frame)
            if index <= resume_after:
                prev_small, prev_hash, last_take = small, digest, timestamp
                continue
            if float(frame.mean()) < min_brightness:
                blank_skipped += 1
                prev_small, prev_hash = small, digest
                continue
            if prev_small is not None:
                too_same = (
                    float(np.mean(cv2.absdiff(small, prev_small))) < diff_thresh
                    and int(np.count_nonzero(digest != prev_hash)) <= hash_max
                )
                if too_same or timestamp - last_take < min_gap:
                    prev_small, prev_hash = small, digest
                    continue

            # 去重和间隔都过了，才值得花 ~30ms 做 Canny：加载页/编队页没有战场，不写盘
            if min_edges > 0 and edge_ratio(frame) < min_edges:
                flat_skipped += 1
                prev_small, prev_hash = small, digest
                continue

            name = _frame_name(vid, index)
            prev_small, prev_hash = small, digest
            last_take = timestamp
            height, width = frame.shape[:2]
            entry = ManifestEntry(
                frame=name, video=vid, frame_idx=index,
                t=round(timestamp, 2), w=int(width), h=int(height),
            )
            path = out_dir / name
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]):
                raise RuntimeError(f"写帧失败: {path}")
            entries.append(entry)
            taken += 1
            if taken >= quota:
                break
    finally:
        close = stream.close
        if callable(close):
            close()

    suffix = ""
    if blank_skipped:
        suffix += f"，跳过黑屏 {blank_skipped} 帧"
    if flat_skipped:
        suffix += f"，跳过加载/编队页 {flat_skipped} 帧"
    print(
        f"[{vid}] -> 新增 {len(entries)} 帧（源 {stream.fps:.1f}fps，"
        f"候选 {sample_fps:g}fps{suffix}）"
    )
    return entries


def stream_fps_guess(known: list[ManifestEntry], resume_after: int) -> float:
    """续跑时用已知帧反推源帧率（manifest 里存了 t，比再问一次解码器便宜）。"""
    for entry in known:
        if entry.frame_idx == resume_after and entry.t > 0:
            return entry.frame_idx / entry.t
    return 30.0


def extract(
    src: Path,
    out_dir: Path,
    *,
    max_frames: int = 400,
    diff_thresh: float = 10.0,
    hash_max: int = 6,
    sample_fps: float = SAMPLE_FPS,
    min_gap: float = MIN_GAP,
    jpg_quality: int = 95,
    min_brightness: float = MIN_BRIGHTNESS,
    min_edges: float = MIN_EDGES,
    hwaccel: bool = False,
    jobs: int | None = None,
    append: bool = True,
) -> list[ManifestEntry]:
    """批量抽帧，manifest 追加写（``append=False`` 时重建）。

    多个视频时默认并行（``jobs=None`` = 最多 4 个线程）：每条线程各跑一个 ffmpeg
    子进程，解码与 JPEG 编码都在 C 里完成，几乎不抢 GIL。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    existing = {} if not append else load_manifest(manifest_path)
    known = set(existing)

    videos = list_videos(src)
    if jobs is None:
        jobs = min(4, len(videos)) or 1
    jobs = max(1, jobs)

    options = dict(
        diff_thresh=diff_thresh,
        hash_max=hash_max,
        sample_fps=sample_fps,
        min_gap=min_gap,
        jpg_quality=jpg_quality,
        min_brightness=min_brightness,
        min_edges=min_edges,
        hwaccel=hwaccel,
        existing=existing,
    )

    results: list[list[ManifestEntry]] = []
    if jobs == 1 or len(videos) <= 1:
        for index, video in enumerate(videos):
            results.append(extract_video(video, out_dir, video_index=index,
                                         max_frames=max_frames, **options))
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [
                pool.submit(extract_video, video, out_dir,
                            video_index=index, max_frames=max_frames, **options)
                for index, video in enumerate(videos)
            ]
            for future in futures:
                results.append(future.result())

    fresh: list[ManifestEntry] = []
    for entries in results:
        for entry in entries:
            if entry.frame in known:      # 保险：同名帧不重复记
                continue
            known.add(entry.frame)
            existing[entry.frame] = entry
            fresh.append(entry)

    fresh.sort(key=lambda entry: (entry.video, entry.frame_idx))
    mode = "a" if append else "w"
    with manifest_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for entry in fresh:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
    return fresh


def iter_frames(frames_dir: Path) -> Iterator[Path]:
    """按文件名顺序遍历帧。"""
    return iter(sorted(Path(frames_dir).glob("*.jpg")))
