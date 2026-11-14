"""Win32 分层窗口叠加层（逐像素 alpha，不用 Tk）。

为什么不用 Tk：Tk 的 `-transparentcolor` 在"无边框 + 手动改 exStyle"的组合下不可靠，
实测会整块渲染成黑色（把游戏/桌面全糊住）。改成标准的 `UpdateLayeredWindow` 路线：

* 窗口样式 `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW`，
  前两个保证"逐像素透明 + 点击穿透"；
* 每帧用 PIL 画一张 RGBA（默认全透明），再 `UpdateLayeredWindow(ULW_ALPHA)` 推给系统；
* 位置/尺寸**每帧跟随游戏客户区**，所以窗口被移动/缩放也能贴住；
* 自带消息泵（PeekMessage），Esc 由 `GetAsyncKeyState` 全局监听，点击穿透也能退出。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import time

import numpy as np

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
WS_POPUP = 0x80000000
ULW_ALPHA = 0x00000002
AC_SRC_ALPHA = 0x00000001
SW_SHOWNOACTIVATE = 4

VK_ESCAPE = 0x1B

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                             wintypes.WPARAM, wintypes.LPARAM)


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", ctypes.c_uint32), ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32), ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16), ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32), ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32), ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32)]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]


# ---- Win32 函数原型（不声明 argtypes 会 lparam 溢出 / GDI 调用失败） ----
LRESULT = ctypes.c_ssize_t

user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.DestroyWindow.argtypes = [wintypes.HWND]
user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, ctypes.c_void_p]
user32.LoadCursorW.restype = wintypes.HANDLE
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.PeekMessageW.restype = wintypes.BOOL
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.DispatchMessageW.restype = LRESULT
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT), ctypes.POINTER(wintypes.SIZE),
    wintypes.HDC, ctypes.POINTER(wintypes.POINT), wintypes.COLORREF,
    ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                            ctypes.c_void_p, ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT]
gdi32.SetDIBits.restype = ctypes.c_int
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


class LayeredOverlay:
    """一个逐像素透明的置顶窗口；每次 :meth:`draw` 同时更新内容和位置。"""

    def __init__(self, title: str = "EYESOFPRIESTESS overlay") -> None:
        self._wndproc = WNDPROC(self._proc)
        self._class_name = f"EOPOverlay{int(time.time() * 1000) % 100000}"
        hinstance = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.style = 0
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinstance
        wc.hCursor = user32.LoadCursorW(0, 32512)   # IDC_ARROW
        wc.lpszClassName = self._class_name
        if not user32.RegisterClassW(ctypes.byref(wc)):
            raise ctypes.WinError()
        self.hwnd = user32.CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            self._class_name, title, WS_POPUP, 0, 0, 100, 100, None, None, hinstance, None,
        )
        if not self.hwnd:
            raise ctypes.WinError()
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)

    # -- 窗口过程：不需要处理什么，默认即可 -------------------------------
    def _proc(self, hwnd, msg, wparam, lparam):  # noqa: ANN001
        if msg == 0x0010:      # WM_CLOSE
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def pump(self) -> bool:
        """处理消息泵；返回 False 表示该退出了（Esc 或窗口被关）。"""
        if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            return False
        msg = wintypes.MSG()
        handled = False
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):   # PM_REMOVE
            handled = True
            if msg.message == 0x0012:                                  # WM_QUIT
                return False
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        if not handled:
            time.sleep(0.005)          # 别空转抢 GIL（会让推理线程饿死）
        return True

    def draw(self, rgba: np.ndarray, left: int, top: int) -> None:
        """把一张 RGBA 图推到窗口（同时把窗口挪到 left/top）。"""
        height, width = rgba.shape[:2]
        bgra = np.dstack([rgba[:, :, 2], rgba[:, :, 1], rgba[:, :, 0], rgba[:, :, 3]])
        bgra = np.ascontiguousarray(bgra)

        screen_dc = user32.GetDC(0)
        mem_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        gdi32.SelectObject(mem_dc, bitmap)

        info = BITMAPINFOHEADER()
        info.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.biWidth, info.biHeight = width, -height
        info.biPlanes, info.biBitCount = 1, 32
        gdi32.SetDIBits(mem_dc, bitmap, 0, height, bgra.ctypes.data, ctypes.byref(info), 0)

        blend = BLENDFUNCTION(0, 0, 255, AC_SRC_ALPHA)
        size = wintypes.SIZE(width, height)
        src = wintypes.POINT(0, 0)
        dst = wintypes.POINT(left, top)
        user32.UpdateLayeredWindow(self.hwnd, screen_dc, ctypes.byref(dst), ctypes.byref(size),
                                   mem_dc, ctypes.byref(src), 0, ctypes.byref(blend), ULW_ALPHA)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)

    def close(self) -> None:
        try:
            user32.DestroyWindow(self.hwnd)
        except Exception:
            pass
