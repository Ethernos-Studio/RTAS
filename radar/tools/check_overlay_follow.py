"""叠加层跟随测试（用自己开的假窗口，游戏窗口拒绝被程序移动）。

流程：开一个 1280x720 的 Tk 窗口（标题 "EYESOFPRIESTESS TEST"）-> 叠加层贴上去 ->
移动假窗口 -> 截屏，看叠加层（黄色 HUD 文字）有没有跟过去。
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

DUMMY = '''
import tkinter as tk
from PIL import Image, ImageTk
root = tk.Tk()
root.title("EYESOFPRIESTESS TEST")
root.geometry("1280x720+40+40")
img = Image.open(r"E:\\RTAS\\radar\\frames\\v001_1-8_f000600.jpg").resize((1280, 720))
photo = ImageTk.PhotoImage(img)
tk.Label(root, image=photo).pack()
root.after(40000, root.destroy)
root.mainloop()
'''

user32 = ctypes.windll.user32


def client_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    point = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    return point.x, point.y, rect.right - rect.left, rect.bottom - rect.top


def find_by_title(title: str) -> int:
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if buffer.value == title:
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found[0] if found else 0


def main() -> int:
    dummy_path = ROOT / "out" / "_dummy_window2.py"
    dummy_path.write_text(DUMMY, encoding="utf-8")
    dummy = subprocess.Popen([sys.executable, str(dummy_path)], cwd=str(ROOT))
    time.sleep(2.5)
    hwnd = find_by_title("EYESOFPRIESTESS TEST")
    print("假窗口 hwnd:", hex(hwnd) if hwnd else None, client_rect(hwnd) if hwnd else None)
    if not hwnd:
        dummy.terminate()
        return 1

    overlay = subprocess.Popen(
        [sys.executable, "tools/live_overlay.py", "--source", "window",
         "--window-title", "EYESOFPRIESTESS TEST", "--overlay", "--imgsz", "960", "--duration", "16"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(5.0)                     # 预热 + 叠加层出现
    before = client_rect(hwnd)

    # 移动假窗口（自己进程创建的，可以程序移动）
    user32.MoveWindow(hwnd, before[0] + 160, before[1] + 110, before[2], before[3], True)
    time.sleep(2.0)
    after = client_rect(hwnd)
    print("移动前:", before, "-> 移动后:", after)

    from PIL import ImageGrab

    ImageGrab.grab().save(ROOT / "out" / "_follow_check.png")
    print("截屏已存 out/_follow_check.png")

    time.sleep(10.0)
    overlay.terminate()
    dummy.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
