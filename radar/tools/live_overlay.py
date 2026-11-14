"""实时叠加层：抓游戏窗口 -> 推理 -> 透明置顶窗口把框画在游戏上面。
三层各跑各的线程，互不阻塞：
1. **抓屏线程**：Windows Graphics Capture（走 RTAS 里那个 wgc_python.dll，
   被遮挡/后台也能抓到），拿不到就退回 PrintWindow；
2. **推理线程**：YOLO（默认最新一版权重，imgsz 960）+ SimpleTracker + 身份匹配
   （只对新出现的 track 做 embedding，省时间）；
3. **UI 线程**：Tk 透明窗口（颜色键透明 + 点击穿透 + 置顶），把框和标签画出来。
用法（游戏开着、窗口未最小化）：
    uv run python tools/live_overlay.py                      # 叠加在游戏窗口上
    uv run python tools/live_overlay.py --window             # 独立窗口（调试用）
    uv run python tools/live_overlay.py --source video --video raw/1-8.mp4
    uv run python tools/live_overlay.py --record out/live.mp4
热键：Esc 退出，空格 暂停/继续，↑/↓ 调置信度。
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.wintypes as wintypes
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
DEFAULT_WGC = Path(r"E:\RTAS\src\rtas\tools\_vendor\wgc\wgc_python.dll")
GAME_TITLES = ("明日方舟", "arknights")
TRANSPARENT_KEY = "#ff00fe"          # Tk 颜色键：这个颜色会被抠成透明
# --------------------------------------------------------------------------- #
# 窗口发现与截图
# --------------------------------------------------------------------------- #
@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    client: tuple[int, int, int, int] = (0, 0, 0, 0)   # left, top, width, height
user32 = ctypes.windll.user32
def _client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    point = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    return point.x, point.y, rect.right - rect.left, rect.bottom - rect.top
def find_game_window(keyword: str | None = None) -> WindowInfo | None:
    """按标题关键字找可见的游戏窗口（默认匹配"明日方舟/arknights"）。"""
    keywords = (keyword,) if keyword else GAME_TITLES
    found: list[WindowInfo] = []
    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value
        if not any(key.lower() in title.lower() for key in keywords):
            return True
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        left, top, width, height = _client_rect(hwnd)
        if width < 320 or height < 240:
            return True
        found.append(WindowInfo(hwnd=hwnd, title=title, class_name=class_buffer.value,
                                client=(left, top, width, height)))
        return True
    user32.EnumWindows(callback, 0)
    if found:
        return found[0]

    # 兜底：按进程名找（游戏最小化后标题会读不到；RTAS 也会自己管理窗口）
    import subprocess

    try:
        output = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq Arknights.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        output = ""
    for line in output.splitlines():
        parts = [item.strip('"') for item in line.split('","')]
        if len(parts) < 2 or "Arknights" not in parts[0]:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        candidate = _main_window_of(pid)
        if candidate:
            title = candidate[0] or "Arknights"
            left, top, width, height = _client_rect(candidate[1])
            if width >= 320 and height >= 240:
                return WindowInfo(hwnd=candidate[1], title=title,
                                  class_name=candidate[2], client=(left, top, width, height))
    return None


def _main_window_of(pid: int) -> tuple[str, int, str] | None:
    """按进程 id 找它的主窗口（标题可能为空）。"""
    result: list[tuple[str, int, str]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value != pid or not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 256)
        result.append((buffer.value, hwnd, class_buffer.value))
        return True

    user32.EnumWindows(callback, 0)
    return result[0] if result else None
class WgcGrabber:
    """Windows Graphics Capture（复用 RTAS 的 wgc_python.dll）。"""
    def __init__(self, dll_path: Path, title: str, class_name: str) -> None:
        if not Path(dll_path).is_file():
            raise FileNotFoundError(f"找不到 WGC 组件: {dll_path}")
        self._lib = ctypes.CDLL(str(dll_path))
        self._lib.StartCapture.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.StartCapture.restype = ctypes.c_uint64
        self._lib.GetFrameMapped.argtypes = [
            ctypes.c_uint64, ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.GetFrameMapped.restype = ctypes.c_int
        self._lib.ReleaseMappedFrame.argtypes = [ctypes.c_uint64]
        self._lib.DestroyCapture.argtypes = [ctypes.c_uint64]
        self._handle = self._lib.StartCapture(title.encode("utf-8"), class_name.encode("utf-8"), 1)
        if not self._handle:
            raise RuntimeError("StartCapture 失败（窗口可能被保护或已最小化）")
    def grab(self) -> np.ndarray | None:
        data = ctypes.POINTER(ctypes.c_ubyte)()
        width, height, pitch = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        if not self._lib.GetFrameMapped(self._handle, ctypes.byref(data),
                                       ctypes.byref(width), ctypes.byref(height), ctypes.byref(pitch)):
            return None
        try:
            buffer = ctypes.string_at(ctypes.cast(data, ctypes.c_void_p).value,
                                      pitch.value * height.value)
            # row_pitch 是**字节/行**：缓冲是 height*pitch 个字节。
            # 原先按 (h, pitch, 4) reshape 又乘了 4，才会报 cannot reshape
            raw = np.frombuffer(buffer, dtype=np.uint8).reshape(height.value, pitch.value)
            frame = raw[:, : width.value * 4].reshape(height.value, width.value, 4)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        finally:
            self._lib.ReleaseMappedFrame(self._handle)
    def close(self) -> None:
        try:
            self._lib.DestroyCapture(self._handle)
        except Exception:
            pass
class PrintWindowGrabber:
    """兜底：PrintWindow(PW_RENDERFULLCONTENT)。"""
    def __init__(self, hwnd: int) -> None:
        self.hwnd = hwnd
    def grab(self) -> np.ndarray | None:
        left, top, width, height = _client_rect(self.hwnd)
        if width <= 0 or height <= 0:
            return None
        hdc = user32.GetWindowDC(self.hwnd)
        gdi32 = ctypes.windll.gdi32
        mem_dc = gdi32.CreateCompatibleDC(hdc)
        bitmap = gdi32.CreateCompatibleBitmap(hdc, width, height)
        gdi32.SelectObject(mem_dc, bitmap)
        try:
            if not user32.PrintWindow(self.hwnd, mem_dc, 2):
                return None
            header = np.zeros((height, width, 4), dtype=np.uint8)
            class BITMAPINFOHEADER(ctypes.Structure):
                _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                            ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                            ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                            ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                            ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                            ("biClrImportant", ctypes.c_uint32)]
            info = BITMAPINFOHEADER()
            info.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            info.biWidth, info.biHeight = width, -height
            info.biPlanes, info.biBitCount = 1, 32
            gdi32.GetDIBits(mem_dc, bitmap, 0, height, header.ctypes.data, ctypes.byref(info), 0)
            return cv2.cvtColor(header, cv2.COLOR_BGRA2BGR)
        finally:
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(self.hwnd, hdc)
    def close(self) -> None:
        pass
# --------------------------------------------------------------------------- #
# 推理管线
# --------------------------------------------------------------------------- #
@dataclass
class Unit:
    box: tuple[int, int, int, int]
    code: str = ""
    score: float = 0.0
    conf: float = 0.0
    track_id: int = -1
@dataclass
class PipelineConfig:
    weights: Path
    gallery: Path | None
    embedder: Path | None
    imgsz: int = 960
    conf: float = 0.3
    iou: float = 0.8
class Pipeline:
    """检测 + 跟踪 + 身份（身份只对新 track 做一次）。"""
    def __init__(self, config: PipelineConfig) -> None:
        from ultralytics import YOLO
        from eyesofpriestess.identity import Encoder, Gallery, crop_box
        from eyesofpriestess.tracking import SimpleTracker
        self.config = config
        self.model = YOLO(str(config.weights))
        self.tracker = SimpleTracker(iou_threshold=0.25)
        self.gallery = Gallery.load(config.gallery) if config.gallery else None
        self.encoder = None
        if self.gallery is not None:
            dim = int(self.gallery.prototypes.shape[1]) if self.gallery.prototypes.size else 512
            checkpoint = config.embedder if dim != 512 else None
            self.encoder = Encoder("resnet18", device="cuda", checkpoint=checkpoint)
        self.labels: dict[int, tuple[str, float]] = {}
        self.recent: list[tuple[list[float], tuple[str, float]]] = []   # 最近的身份结果，用于 track 更换时复用
        self.crop_box = crop_box
        self._fps = 0.0
        self._last = time.perf_counter()
    def step(self, frame: np.ndarray) -> list[Unit]:
        from eyesofpriestess.postprocess import MergeSuppressor
        result = self.model.predict(frame, imgsz=self.config.imgsz, conf=self.config.conf,
                                    iou=self.config.iou, verbose=False, device="0")[0]
        raw = []
        for box in result.boxes or []:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            raw.append([x1, y1, x2, y2])
        boxes = MergeSuppressor()(raw)
        units = self.tracker.update(boxes)
        def iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
            union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return 0.0 if union <= 0 else inter / union
        new_crops, new_ids = [], []
        for box, track_id in units:
            if track_id in self.labels or self.gallery is None:
                continue
            # tracker 换了 id 但位置几乎没变 -> 直接复用上一次的身份，别重复跑编码器
            reused = None
            for old_box, cached in self.recent:
                if iou(box, old_box) > 0.5:
                    reused = cached
                    break
            if reused is not None:
                self.labels[track_id] = reused
                continue
            new_crops.append(self.crop_box(frame, box, pad=0.0))
            new_ids.append(track_id)
        if new_crops:
            for track_id, match in zip(new_ids, self.gallery.identify(self.encoder.embed(new_crops))):
                self.labels[track_id] = (match.code, match.score)
        # 维护最近身份缓存（最多 40 条，避免无限增长）
        for box, track_id in units:
            if track_id in self.labels:
                self.recent.append((list(box), self.labels[track_id]))
        self.recent = self.recent[-40:]
        output: list[Unit] = []
        for box, track_id in units:
            code, score = self.labels.get(track_id, ("", 0.0))
            output.append(Unit(box=(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])),
                               code=code, score=score, track_id=track_id))
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 / max(1e-6, now - self._last) if self._last else 0.0
        self._last = now
        return output
    @property
    def fps(self) -> float:
        return self._fps
# --------------------------------------------------------------------------- #
# 叠加层 UI
# --------------------------------------------------------------------------- #
def make_click_through(window) -> None:
    GWL_EXSTYLE = -20
    WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = 0x80000, 0x20, 0x80
    hwnd = int(window.winfo_id())
    style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW)
def start_escape_watchdog(states: dict, root) -> None:
    """全局监听 Esc（GetAsyncKeyState 不需要窗口焦点）。

    叠加层是点击穿透的，Tk 永远拿不到焦点，所以不能用 Tk 的键盘绑定当急停。
    """
    import ctypes as _ctypes
    import threading as _threading
    import time as _time

    VK_ESCAPE = 0x1B
    user32 = _ctypes.windll.user32

    def loop() -> None:
        while states.get("running", False):
            if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                states["running"] = False
                try:
                    root.after(0, root.destroy)
                except Exception:
                    pass
                return
            _time.sleep(0.05)

    _threading.Thread(target=loop, daemon=True).start()


def run_layered_overlay(states: dict, source_name: str, duration: float = 0.0) -> None:
    """纯 Win32 分层窗口：逐像素透明、点击穿透、位置每帧跟随游戏客户区。"""
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from overlay_win32 import LayeredOverlay

    def load_font(size: int):
        for candidate in (r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\msyh.ttc",
                          r"C:\Windows\Fonts\simhei.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    font = load_font(15)
    hud_font = load_font(14)
    overlay_window = LayeredOverlay()
    hwnd = int(states.get("hwnd", 0))
    deadline = time.time() + duration if duration > 0 else None

    try:
        while states.get("running", True):
            if not overlay_window.pump():
                break
            if deadline is not None and time.time() > deadline:
                break

            # 只在游戏是前台窗口时显示：否则会把框画在别的程序上（很烦人）
            foreground = user32.GetForegroundWindow()
            if hwnd and foreground != hwnd:
                user32.ShowWindow(overlay_window.hwnd, 0)      # SW_HIDE
                time.sleep(0.1)
                continue
            user32.ShowWindow(overlay_window.hwnd, 4)          # SW_SHOWNOACTIVATE

            left, top, width, height = _client_rect(hwnd) if hwnd else (0, 0, 0, 0)
            if width <= 0 or height <= 0:          # 游戏最小化/关掉
                user32.ShowWindow(overlay_window.hwnd, 0)
                time.sleep(0.2)
                continue

            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            frame_w = states.get("frame_w") or width
            frame_h = states.get("frame_h") or height
            sx, sy = width / max(1, frame_w), height / max(1, frame_h)

            for unit in states.get("units", []):
                x, y, w, h = unit.box
                x, y, w, h = x * sx, y * sy, w * sx, h * sy
                label = f"#{unit.track_id} {unit.code}" if unit.code else f"#{unit.track_id}"
                if unit.score:
                    label += f"({unit.score:.2f})"
                draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 102, 235), width=2)
                text_y = max(0, y - 20)
                draw.text((x + 2, text_y), label, font=font, fill=(0, 255, 102, 255))
            draw.text((8, 6), f"{source_name}  {states.get('fps', 0):.1f} FPS  "
                              f"conf {states.get('conf', 0):.2f}",
                      font=hud_font, fill=(255, 255, 90, 230))
            if states.get("paused"):
                draw.text((width // 2 - 40, 10), "PAUSED", font=font,
                          fill=(255, 90, 90, 255))

            overlay_window.draw(np.asarray(canvas), left, top)
            time.sleep(0.033)
    finally:
        overlay_window.close()


def run_ui(states: dict, source_name: str, overlay: bool, duration: float = 0.0) -> None:
    """画框 + 跟随游戏窗口。

    关键点：
    * 叠加窗口的尺寸/位置**每帧都跟游戏客户区对齐**（RTAS 会移动窗口，启动时对一次没用）；
    * 检测框是在"抓到的画面"坐标系里，要按 client_w/frame_w 缩放后才是叠加层坐标；
    * 游戏最小化（客户区 0x0）时把叠加层藏起来，恢复后再显示。
    """
    import tkinter as tk
    root = tk.Tk()
    hwnd = int(states.get("hwnd", 0))
    if overlay:
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-transparentcolor", TRANSPARENT_KEY)
        root.geometry(f"{states['width']}x{states['height']}+{states['left']}+{states['top']}")
    else:
        root.title(f"EYESOFPRIESTESS · {source_name}")
        screen_w = root.winfo_screenwidth()
        debug_left = states['left'] + states['width'] + 24
        if debug_left + states['width'] > screen_w:
            debug_left = max(0, states['left'] - states['width'] - 24)
        root.geometry(f"{states['width']}x{states['height']}+{debug_left}+{states['top']}")
        root.configure(bg="#101010")
    canvas = tk.Canvas(root, width=states["width"], height=states["height"],
                       bg=TRANSPARENT_KEY if overlay else "#101010", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    root.update_idletasks()
    if overlay:
        make_click_through(root)

    def follow_window() -> tuple[int, int, int, int]:
        """把叠加层挪到游戏客户区；返回最新客户区。"""
        if not hwnd:
            return states["left"], states["top"], states["width"], states["height"]
        left, top, width, height = _client_rect(hwnd)
        if width <= 0 or height <= 0:
            return 0, 0, 0, 0
        if (left, top, width, height) != (states["left"], states["top"],
                                          states["width"], states["height"]):
            states.update(left=left, top=top, width=width, height=height)
            if overlay:
                root.geometry(f"{width}x{height}+{left}+{top}")
            canvas.config(width=width, height=height)
        return left, top, width, height

    def redraw() -> None:
        left, top, width, height = follow_window()
        if width <= 0 or height <= 0:          # 游戏最小化
            if overlay:
                root.withdraw()
            root.after(200, redraw)
            return
        if overlay and root.state() == "withdrawn":
            root.deiconify()
        canvas.delete("all")
        frame_w = states.get("frame_w") or width
        frame_h = states.get("frame_h") or height
        sx = width / max(1, frame_w)
        sy = height / max(1, frame_h)
        for unit in states.get("units", []):
            x, y, w, h = unit.box
            x, y, w, h = x * sx, y * sy, w * sx, h * sy
            label = f"#{unit.track_id} {unit.code}" if unit.code else f"#{unit.track_id}"
            if unit.score:
                label += f"({unit.score:.2f})"
            canvas.create_rectangle(x, y, x + w, y + h, outline="#00ff66", width=2)
            canvas.create_text(x, max(8, y - 8), text=label, anchor="sw",
                               fill="#00ff66", font=("Consolas", 13, "bold"))
        canvas.create_text(8, 8, text=f"{source_name}  {states.get('fps', 0):.1f} FPS  "
                                      f"conf {states.get('conf', 0):.2f}", anchor="nw",
                           fill="#ffff66", font=("Consolas", 12))
        if states.get("paused"):
            canvas.create_text(width // 2, 24, text="PAUSED", fill="#ff5555",
                               font=("Consolas", 16, "bold"))
        root.after(33, redraw)

    def on_key(event) -> None:
        if event.keysym == "Escape":
            states["running"] = False
            root.destroy()
        elif event.keysym == "space":
            states["paused"] = not states.get("paused", False)
        elif event.keysym in ("Up", "Down"):
            delta = 0.05 if event.keysym == "Up" else -0.05
            states["conf"] = min(0.9, max(0.05, states.get("conf", 0.3) + delta))

    if duration > 0:
        root.after(int(duration * 1000), root.destroy)
    root.bind("<Key>", on_key)
    start_escape_watchdog(states, root)
    redraw()
    root.mainloop()

# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="明日方舟实时叠加层（抓窗口 + 推理 + 透明置顶）")
    parser.add_argument("--source", choices=["window", "video"], default="window")
    parser.add_argument("--video", default="raw/1-8.mp4", help="--source video 时的视频")
    parser.add_argument("--window-title", help="窗口标题关键字（默认 明日方舟 / arknights）")
    parser.add_argument("--weights", help="YOLO 权重（默认最新训练产物）")
    parser.add_argument("--gallery", default="weights/gallery.npz")
    parser.add_argument("--embedder", default="weights/embedder.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--wgc", default=str(DEFAULT_WGC), help="wgc_python.dll 路径")
    parser.add_argument("--overlay", action="store_true",
                   help="透明叠加在游戏上（默认关闭：先跑 --window 调试，确认没问题再开）")
    parser.add_argument("--window", action="store_true", help="强制用普通窗口")
    parser.add_argument("--record", help="同时录制带框视频（mp4）")
    parser.add_argument("--duration", type=float, default=0.0, help="自动退出前跑多少秒（0=一直跑）")
    parser.add_argument("--target-fps", type=float, default=20.0, help="抓屏节流（太高会和推理抢 CPU）")
    parser.add_argument("--no-wgc", action="store_true", help="强制用 PrintWindow 抓屏")
    args = parser.parse_args()
    weights = Path(args.weights) if args.weights else _latest_weights()
    gallery = Path(args.gallery) if args.gallery and Path(args.gallery).exists() else None
    embedder = Path(args.embedder) if args.embedder and Path(args.embedder).exists() else None
    print("加载模型…")
    pipeline = Pipeline(PipelineConfig(weights=weights, gallery=gallery, embedder=embedder,
                                       imgsz=args.imgsz, conf=args.conf))
    # 预热：第一次推理要初始化 CUDA/模型（实测 7 秒），放在开窗之前，否则前几秒什么都看不到
    warmup = np.zeros((720, 1280, 3), dtype=np.uint8)
    pipeline.step(warmup)
    print("预热完成，开始实时推理")
    states: dict = {"running": True, "units": [], "fps": 0.0, "conf": args.conf,
                    "paused": False, "width": 1280, "height": 720, "left": 80, "top": 80}
    grabber = None
    camera = None
    source_name = args.source
    if args.source == "window":
        info = find_game_window(args.window_title)
        if info is None:
            print("找不到游戏窗口（标题含 明日方舟 / arknights）——先用 --source video 试", file=sys.stderr)
            return 2
        left, top, width, height = info.client
        print(f"窗口: {info.title}  客户区 {width}x{height} @ ({left},{top})")
        states.update(width=width, height=height, left=left, top=top, hwnd=info.hwnd)
        source_name = info.title
        grabber = None
        if not args.no_wgc:
            try:
                candidate = WgcGrabber(Path(args.wgc), info.title, info.class_name)
                # WGC 有时会"启动成功但永远不给帧"，先试抓 1.5 秒
                deadline = time.time() + 1.5
                got = None
                while time.time() < deadline and got is None:
                    got = candidate.grab()
                    if got is None:
                        time.sleep(0.05)
                if got is not None:
                    grabber = candidate
                    print(f"抓屏: Windows Graphics Capture（首帧 {got.shape[1]}x{got.shape[0]}）")
                else:
                    candidate.close()
                    print("WGC 1.5 秒没出帧，换 PrintWindow", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"WGC 不可用（{exc}），退回 PrintWindow", file=sys.stderr)
        if grabber is None:
            grabber = PrintWindowGrabber(info.hwnd)
            probe = grabber.grab()
            if probe is None or probe.size == 0:
                print("PrintWindow 也抓不到（游戏可能用了独占全屏/DX12）", file=sys.stderr)
                return 3
            print(f"抓屏: PrintWindow（首帧 {probe.shape[1]}x{probe.shape[0]}）")
    else:
        camera = cv2.VideoCapture(str(video))
        if not camera.isOpened():
            print(f"打不开视频 {video}", file=sys.stderr)
            return 2
        ok, frame = camera.read()
        if not ok:
            print("视频读不到帧", file=sys.stderr)
            return 2
        states.update(width=frame.shape[1], height=frame.shape[0], left=60, top=60)
        source_name = video.name
        camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frames: queue.Queue = queue.Queue(maxsize=2)
    writer = None
    # WGC 输出分辨率和窗口客户区可能不一致（实测 1265x711 vs 1024x576），
    # 所以 VideoWriter 等第一帧到了再按实际尺寸建（见 worker）
    target_interval = 1.0 / max(1.0, args.target_fps)
    def producer() -> None:
        while states["running"]:
            if states.get("paused"):
                time.sleep(0.05)
                continue
            frame = grabber.grab() if grabber else None
            if grabber is None and camera is not None:
                ok, frame = camera.read()
                if not ok:
                    camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                frames.put_nowait(frame)
            except queue.Full:
                pass
            time.sleep(max(0.0, target_interval - 0.002))
    def worker() -> None:
        nonlocal writer
        while states["running"]:
            try:
                frame = frames.get(timeout=0.2)
            except queue.Empty:
                continue
            pipeline.config.conf = states["conf"]
            start = time.perf_counter()
            units = pipeline.step(frame)
            states["units"] = units
            states["frames"] = states.get("frames", 0) + 1
            states["fps"] = 1.0 / max(1e-6, time.perf_counter() - start)
            states["frame_w"], states["frame_h"] = frame.shape[1], frame.shape[0]
            if args.record:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             30.0, (width, height))
                canvas = frame.copy()
                for unit in units:
                    x, y, w, h = unit.box
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 102), 2)
                    cv2.putText(canvas, f"#{unit.track_id} {unit.code}", (x, max(14, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 102), 2)
                writer.write(canvas)
    producer_thread = threading.Thread(target=producer, daemon=True)
    worker_thread = threading.Thread(target=worker, daemon=True)
    producer_thread.start()
    worker_thread.start()
    try:
        overlay_mode = bool(args.overlay) and not args.window
        if overlay_mode:
            run_layered_overlay(states, source_name, duration=args.duration)
        else:
            run_ui(states, source_name, overlay=False, duration=args.duration)
    finally:
        states["running"] = False
        worker_thread.join(timeout=3.0)
        producer_thread.join(timeout=2.0)
        if writer is not None:
            writer.release()
        if grabber is not None:
            grabber.close()
        if camera is not None:
            camera.release()
        print(f"处理 {states.get('frames', 0)} 帧，最后 {states.get('fps', 0):.1f} FPS")
        sys.stdout.flush()
        # Tk 在解释器退出阶段偶发 "Tcl_AsyncDelete: async handler deleted by the wrong thread"
        # （后台线程还活着时销毁 Tcl 解释器），这里直接结束进程，避免在收尾时炸
        import os
        os._exit(0)
    return 0
def _latest_weights() -> Path:
    candidates = sorted(
        (path for path in (ROOT / "runs" / "arknights").glob("*/weights/best.pt") if path.exists()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit("没有训练产物，先 `radar train`")
    return candidates[-1]
if __name__ == "__main__":
    raise SystemExit(main())
