# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of RTAS (Rhodeisland Tactical Autonomous System).
#
# RTAS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RTAS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with RTAS. If not, see <https://www.gnu.org/licenses/>.

"""Capture the Arknights window into ``__temp__`` for manual testing.

The tool finds a visible window whose title or process name contains
``明日方舟`` or ``arknights``, then captures its client area with Windows
Graphics Capture (WGC) and writes a PNG into the project's ``__temp__``
directory. WGC reads the window's own composed surface, so overlapping
windows do not appear in the result and the game does not need focus.

Usage::

    uv run rtas-capture --list
    uv run rtas-capture
    uv run rtas-capture --keyword MuMu --output-dir __temp__
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from PIL import Image

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class POINT(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    user32.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    if hasattr(user32, "SetProcessDpiAwarenessContext"):
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        user32.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
    user32.SetProcessDPIAware.argtypes = []
    user32.SetProcessDPIAware.restype = wintypes.BOOL

IS_WINDOWS = sys.platform == "win32"

DEFAULT_KEYWORDS: tuple[str, ...] = ("明日方舟", "arknights")
DEFAULT_PREFIX = "arknights"

#: 常见模拟器/游戏进程（窗口标题里未必带「明日方舟」）
GAME_PROCESS_HINTS: tuple[str, ...] = (
    "arknights",
    "hd-player",      # MuMu
    "mumuplayer",
    "mumunx",
    "nemu",
    "nox",
    "bluestacks",
    "ldplayer",
    "ldvbox",
    "dmmgameplayer",
)

#: 标题里可能出现 arknights（文件名）但肯定不是游戏的进程
NON_GAME_PROCESSES: tuple[str, ...] = (
    "code.exe",
    "code - insiders.exe",
    "explorer.exe",
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "notepad.exe",
    "notepad++.exe",
    "devenv.exe",
    "pycharm64.exe",
    "idea64.exe",
    "sublime_text.exe",
    "photoshop.exe",
    "acrobat.exe",
    "windowsterminal.exe",
)
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SW_SHOW = 5
SW_SHOWMINNOACTIVE = 7
SW_RESTORE = 9
WM_SYSCOMMAND = 0x0112
SC_RESTORE = 0xF120
PER_MONITOR_AWARE_V2 = -4

_DPI_READY = False


class CaptureError(RuntimeError):
    """Raised when a window cannot be captured."""


@dataclass(frozen=True)
class Region:
    """A rectangle in virtual-screen coordinates."""

    left: int
    top: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)



@dataclass(frozen=True)
class WindowInfo:
    """A top-level window discovered on the desktop."""

    hwnd: int
    title: str
    class_name: str
    process: str
    rect: Region
    client_rect: Region
    minimized: bool

    @property
    def display_title(self) -> str:
        return self.title or "(无标题)"


def _enable_dpi_awareness() -> None:
    """Make sure window rectangles use physical pixels."""
    global _DPI_READY
    if _DPI_READY or not IS_WINDOWS:
        return
    try:
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(PER_MONITOR_AWARE_V2))
        else:
            user32.SetProcessDPIAware()
    except OSError:
        pass
    _DPI_READY = True


def _window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def _class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def _process_name(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""

    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return ""
    try:
        buffer = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buffer))
        if kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(size)
        ):
            return Path(buffer.value).name
        return ""
    finally:
        kernel32.CloseHandle(handle)


def _window_rect(hwnd: int) -> Region:
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return Region(0, 0, 0, 0)
    return Region(
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top,
    )


def _client_rect(hwnd: int) -> Region:
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return Region(0, 0, 0, 0)

    origin = POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return Region(0, 0, 0, 0)

    return Region(
        origin.x,
        origin.y,
        max(0, rect.right - rect.left),
        max(0, rect.bottom - rect.top),
    )


def list_windows(
    *,
    include_hidden: bool = False,
    include_untitled: bool = False,
) -> list[WindowInfo]:
    """Return top-level windows sorted by usable client area (largest first)."""
    if not IS_WINDOWS:
        raise CaptureError("窗口截图仅支持 Windows。")

    _enable_dpi_awareness()
    windows: list[WindowInfo] = []

    @WNDENUMPROC
    def _callback(hwnd: int, _lparam: int) -> bool:
        if not include_hidden and not user32.IsWindowVisible(hwnd):
            return True

        title = _window_title(hwnd)
        if not title and not include_untitled:
            return True

        window_rect = _window_rect(hwnd)
        client_rect = _client_rect(hwnd)
        if window_rect.area <= 0:
            return True

        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                class_name=_class_name(hwnd),
                process=_process_name(hwnd),
                rect=window_rect,
                client_rect=client_rect,
                minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    if not user32.EnumWindows(_callback, 0):
        raise CaptureError("枚举窗口失败。")

    return sorted(windows, key=lambda item: (-item.client_rect.area, item.title.casefold()))


def _matches_keywords(title: str, process: str, keywords: Iterable[str]) -> bool:
    haystack = f"{title}\n{process}".casefold()
    return any(keyword.casefold() in haystack for keyword in keywords if keyword)


def find_windows(
    keywords: Sequence[str] | None = None,
    *,
    include_minimized: bool = True,
) -> list[WindowInfo]:
    """Find visible windows matching any keyword in their title or process name."""
    active_keywords = tuple(keywords or DEFAULT_KEYWORDS)
    matches = [
        window
        for window in list_windows()
        if _matches_keywords(window.title, window.process, active_keywords)
    ]
    if not include_minimized:
        matches = [window for window in matches if not window.minimized]
    return matches


def _game_score(window: WindowInfo) -> int:
    """给候选窗口打可信度分：进程像游戏 > 标题含「明日方舟」> 标题含 arknights。"""
    process = (window.process or "").casefold()
    title = window.title or ""
    score = 0
    if any(hint in process for hint in GAME_PROCESS_HINTS):
        score += 50
    if "明日方舟" in title:
        score += 50
    elif "arknights" in title.casefold():
        score += 20
    return score


def find_game_windows(keywords: Sequence[str] | None = None) -> list[WindowInfo]:
    """找「可截图的游戏窗口」：先按关键字找，再排掉编辑器/浏览器等。

    标题里带 ``arknights_20261113_201223.png`` 的 VS Code 窗口曾经被误认成游戏，
    所以这里既按进程名打分，也把明显不是游戏的进程剔除。返回值按可信度、
    再按客户区面积降序排列。
    """
    if keywords:
        candidates = find_windows(list(keywords))
    else:
        candidates = find_windows(list(DEFAULT_KEYWORDS) + list(GAME_PROCESS_HINTS))

    candidates = [
        window for window in candidates
        if (window.process or "").casefold() not in NON_GAME_PROCESSES
    ]
    return sorted(
        candidates,
        key=lambda window: (-_game_score(window), -window.client_rect.area),
    )


def find_window_by_hwnd(hwnd: int) -> WindowInfo | None:
    """Return a window by handle, including hidden and untitled windows."""
    for window in list_windows(include_hidden=True, include_untitled=True):
        if window.hwnd == hwnd:
            return window
    return None


def project_root() -> Path:
    """Best-effort project root, falling back to the current directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def default_temp_dir() -> Path:
    """Default screenshot destination: ``<project>/__temp__``."""
    return project_root() / "__temp__"


def _safe_prefix(prefix: str) -> str:
    """Keep generated filenames inside the requested output directory."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", prefix).strip(" .")
    return cleaned or DEFAULT_PREFIX


def unique_path(directory: Path, prefix: str = DEFAULT_PREFIX) -> Path:
    """Return a timestamped PNG path that does not exist yet."""
    directory = Path(directory)
    prefix = _safe_prefix(prefix)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = directory / f"{prefix}_{stamp}.png"
    index = 2
    while candidate.exists():
        candidate = directory / f"{prefix}_{stamp}_{index}.png"
        index += 1
    return candidate


def _activate_window(hwnd: int, delay: float) -> bool:
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    else:
        user32.ShowWindow(hwnd, SW_SHOW)

    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)

    if delay > 0:
        time.sleep(delay)

    return user32.GetForegroundWindow() == hwnd


def _wait_for_client_area(hwnd: int, timeout: float = 2.0) -> bool:
    """Wait for a restored window to expose a non-zero client area."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if not user32.IsIconic(hwnd) and _client_rect(hwnd).width > 0:
            return True
        time.sleep(0.05)
    return False


def capture_window(
    window: WindowInfo,
    *,
    activate: bool = False,
    delay: float = 0.3,
    client_area: bool = True,
) -> Image.Image:
    """Capture a window and return a Pillow image.

    Windows Graphics Capture reads the window's composed surface, so the
    window may be occluded or in the background. It can also capture a
    minimized window. Set ``activate`` to bring it to the foreground first.
    """
    if not IS_WINDOWS:
        raise CaptureError("窗口截图仅支持 Windows。")

    _enable_dpi_awareness()

    restore_minimize = False
    if activate:
        _activate_window(window.hwnd, delay)
    elif window.minimized:
        # A minimized DirectX window has no client surface for WGC to read.
        # Restore it for the capture, then put it back as it was. Some games
        # ignore ShowWindow and only respond to WM_SYSCOMMAND/SC_RESTORE.
        user32.SetForegroundWindow(window.hwnd)
        user32.ShowWindow(window.hwnd, SW_RESTORE)
        if not _wait_for_client_area(window.hwnd, timeout=0.8):
            user32.PostMessageW(window.hwnd, WM_SYSCOMMAND, SC_RESTORE, 0)
            _wait_for_client_area(window.hwnd, timeout=1.5)
        if _client_rect(window.hwnd).width <= 0:
            raise CaptureError(
                "窗口处于最小化且无法自动恢复，请先手动打开游戏窗口再截图。"
            )
        time.sleep(0.2)
        restore_minimize = True

    from PIL import Image

    from rtas.tools._wgc import WgcError, WindowCapture as WgcCapture

    try:
        with WgcCapture(
            window.title,
            window.class_name,
            client_area_only=client_area,
            capture_cursor=False,
        ) as capture:
            frame = capture.capture_bgra(timeout=1.2)
    except (OSError, WgcError) as exc:
        raise CaptureError(f"WGC 截图失败: {exc}") from exc
    finally:
        if restore_minimize:
            user32.ShowWindow(window.hwnd, SW_SHOWMINNOACTIVE)

    if frame is None:
        raise CaptureError("WGC 在超时时间内没有返回帧。")

    raw, width, height, row_pitch = frame
    image = Image.frombytes(
        "RGBA", (width, height), raw, "raw", "BGRA", row_pitch, 1
    )
    return image.convert("RGB")



def _is_mostly_black(image: Image.Image) -> bool:
    try:
        from PIL import ImageStat

        return ImageStat.Stat(image.convert("L")).mean[0] < 2.0
    except Exception:
        return False


def save_capture(
    image: Image.Image,
    output_dir: Path,
    *,
    prefix: str = DEFAULT_PREFIX,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = unique_path(output_dir, prefix=prefix)
    image.save(path, format="PNG")
    return path


def _configure_console() -> None:
    """Use UTF-8 for console output so window titles do not crash or mojibake."""
    if IS_WINDOWS:
        try:
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleCP(65001)
        except OSError:
            pass

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _display_text(text: str, limit: int = 80) -> str:
    """Strip invisible characters and collapse whitespace for console output."""
    for char in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        text = text.replace(char, "")
    text = " ".join(text.split())
    return text[:limit]


def _print_window_table(windows: Sequence[WindowInfo]) -> None:
    if not windows:
        print("未找到匹配的可见窗口。")
        return

    print(f"{'HWND':<12} {'客户端尺寸':<14} {'进程':<24} 标题")
    print("-" * 100)
    for window in windows:
        size = f"{window.client_rect.width}x{window.client_rect.height}"
        process = _display_text(window.process or "-", limit=24)
        print(
            f"0x{window.hwnd:08X}   {size:<14} {process:<24} "
            f"{_display_text(window.display_title)}"
        )


def run_capture_command(args: argparse.Namespace) -> int:
    """Execute the ``capture`` command from parsed arguments."""
    _configure_console()
    if not IS_WINDOWS:
        print("错误: 窗口截图仅支持 Windows。", file=sys.stderr)
        return 1

    keywords = tuple(args.keyword or DEFAULT_KEYWORDS)
    output_dir = Path(args.output_dir) if args.output_dir else default_temp_dir()

    if args.list_windows:
        windows = list_windows()
        if args.keyword:
            windows = [
                window
                for window in windows
                if _matches_keywords(window.title, window.process, keywords)
            ]
        _print_window_table(windows)
        return 0

    if args.hwnd:
        window = find_window_by_hwnd(args.hwnd)
        if window is None:
            print(f"错误: 未找到窗口句柄 0x{args.hwnd:08X}。", file=sys.stderr)
            return 1
        windows = [window]
    elif args.keyword:
        # 用户点名要哪个窗口，就按关键字顺序来，不再替他挑
        windows = find_windows(keywords)
    else:
        # 默认按「像不像游戏」排序：客户端启动器（Games.exe）标题也叫「明日方舟」，
        # 但真正的游戏窗口是 Arknights.exe，别把启动器当成游戏截下来。
        windows = find_game_windows()

    if not windows:
        print(
            "错误: 未找到《明日方舟》窗口。请先启动游戏，"
            "或用 --list 查看窗口标题后通过 --keyword 指定关键字。",
            file=sys.stderr,
        )
        return 1

    selected = windows if args.all else windows[:1]
    saved = 0
    for window in selected:
        try:
            image = capture_window(
                window,
                activate=args.activate,
                delay=args.delay,
                client_area=not args.full_window,
            )
            path = save_capture(image, output_dir, prefix=args.prefix)
        except Exception as exc:
            print(
                f"截图失败 [{_display_text(window.display_title)}]: {exc}",
                file=sys.stderr,
            )
            continue

        saved += 1
        if not args.quiet:
            print(
                f"已保存: {path}  "
                f"({image.width}x{image.height})  "
                f"[{_display_text(window.display_title)}]"
            )
        if _is_mostly_black(image):
            print(
                "警告: 截图接近全黑，可能窗口被遮挡或使用了独占全屏。",
                file=sys.stderr,
            )

    if saved == 0:
        return 1
    if not args.quiet:
        print(f"共保存 {saved} 张截图到: {output_dir}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register capture arguments on a parser (shared by CLI and entry point)."""
    parser.description = (
        "自动查找《明日方舟》窗口，用 Windows Graphics Capture 截取窗口本身"
        "（无视遮挡/后台），保存到项目 __temp__ 目录。"
    )
    parser.epilog = """
示例:
  rtas-capture                      # 后台/遮挡截图到 __temp__
  rtas-capture --list               # 列出当前所有可见窗口
  rtas-capture -k MuMu              # 按模拟器标题关键字查找
  rtas-capture --hwnd 0x00123456    # 按窗口句柄截图
  rtas-capture --all                # 截图所有匹配窗口
  rtas-capture --activate           # 先把窗口切到前台再截图
  rtas-capture -o D:\\shots          # 指定输出目录
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    parser.add_argument(
        "-k",
        "--keyword",
        action="append",
        metavar="TEXT",
        help="窗口标题或进程名关键字，可重复指定（默认: 明日方舟 / arknights）",
    )
    parser.add_argument(
        "--hwnd",
        type=lambda value: int(value, 0),
        help="按窗口句柄精确截图，支持 0x 前缀",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认: 项目根目录下的 __temp__）",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"文件名前缀（默认: {DEFAULT_PREFIX}）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="配合 --activate 使用的前台等待秒数（默认: 0.3）",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help="截图前把窗口切到前台（默认后台/遮挡截图，不打扰当前操作）",
    )
    parser.add_argument(
        "--full-window",
        action="store_true",
        help="包含标题栏和边框（默认只截客户区）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="保存所有匹配窗口，而不是只保存面积最大的一个",
    )
    parser.add_argument(
        "--list",
        dest="list_windows",
        action="store_true",
        help="列出可见窗口后退出",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="成功时不输出详细信息",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone entry point for ``rtas-capture``."""
    parser = argparse.ArgumentParser(prog="rtas-capture")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run_capture_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
