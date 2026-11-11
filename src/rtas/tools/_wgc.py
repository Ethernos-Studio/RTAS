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

"""Minimal ctypes binding for the bundled Windows Graphics Capture DLL.

The DLL is ``wgc_python.dll`` from wgc-python 2.1.2 (MIT License). See
``_vendor/wgc/LICENSE`` and the repository's ``THIRD_PARTY_NOTICES.md``.
Only the raw BGRA frame API is used so the capture path does not require
NumPy or OpenCV.
"""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

DLL_PATH = Path(__file__).resolve().parent / "_vendor" / "wgc" / "wgc_python.dll"


class WgcError(RuntimeError):
    """Raised when the bundled WGC library cannot capture a window."""


_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not DLL_PATH.is_file():
        raise WgcError(f"WGC 组件缺失: {DLL_PATH}")

    library = ctypes.CDLL(str(DLL_PATH))

    library.StartCapture.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    library.StartCapture.restype = ctypes.c_uint64
    library.DestroyCapture.argtypes = [ctypes.c_uint64]
    library.DestroyCapture.restype = None
    library.GetFrameMapped.argtypes = [
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.GetFrameMapped.restype = ctypes.c_int
    library.ReleaseMappedFrame.argtypes = [ctypes.c_uint64]
    library.ReleaseMappedFrame.restype = None
    library.PauseCapture.argtypes = [ctypes.c_uint64]
    library.PauseCapture.restype = None
    library.ResumeCapture.argtypes = [ctypes.c_uint64]
    library.ResumeCapture.restype = None
    library.GetLastErrorMsg.argtypes = []
    library.GetLastErrorMsg.restype = ctypes.c_char_p

    if hasattr(library, "SetCursorCaptureEnabled"):
        library.SetCursorCaptureEnabled.argtypes = [ctypes.c_uint64, ctypes.c_int]
        library.SetCursorCaptureEnabled.restype = None

    _lib = library
    return library


def last_error() -> str:
    """Return the DLL's last error message, if any."""
    try:
        message = _load().GetLastErrorMsg()
    except (OSError, AttributeError):
        return ""
    return message.decode("utf-8", errors="replace") if message else ""


class WindowCapture:
    """Capture one window through Windows Graphics Capture.

    The window may be occluded or in the background; the captured pixels come
    from DWM's composed window surface, not from the screen.
    """

    def __init__(
        self,
        title: str,
        class_name: str,
        *,
        client_area_only: bool = True,
        capture_cursor: bool = False,
    ) -> None:
        self._lib = _load()
        self._handle = self._lib.StartCapture(
            title.encode("utf-8"),
            class_name.encode("utf-8"),
            int(client_area_only),
        )
        if not self._handle:
            raise WgcError(last_error() or "StartCapture 失败")

        if not capture_cursor and hasattr(self._lib, "SetCursorCaptureEnabled"):
            self._lib.SetCursorCaptureEnabled(self._handle, 0)

    def __enter__(self) -> "WindowCapture":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _frame(self) -> tuple[int, int, int, int] | None:
        data = ctypes.POINTER(ctypes.c_ubyte)()
        width = ctypes.c_int()
        height = ctypes.c_int()
        row_pitch = ctypes.c_int()
        if not self._lib.GetFrameMapped(
            self._handle,
            ctypes.byref(data),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(row_pitch),
        ):
            return None
        return (
            ctypes.cast(data, ctypes.c_void_p).value or 0,
            width.value,
            height.value,
            row_pitch.value,
        )

    def capture_bgra(self, timeout: float = 1.0) -> tuple[bytes, int, int, int] | None:
        """Return ``(bgra_bytes, width, height, row_pitch)`` or ``None``."""
        self._lib.ResumeCapture(self._handle)
        deadline = time.perf_counter() + max(0.0, timeout)
        frame = None
        while time.perf_counter() < deadline:
            frame = self._frame()
            if frame is not None:
                break
            time.sleep(0.002)

        if frame is None:
            self._lib.PauseCapture(self._handle)
            return None

        pointer, width, height, row_pitch = frame
        try:
            raw = ctypes.string_at(pointer, row_pitch * height)
        finally:
            self._lib.ReleaseMappedFrame(self._handle)
            self._lib.PauseCapture(self._handle)
        return raw, width, height, row_pitch

    def close(self) -> None:
        handle = getattr(self, "_handle", 0)
        if handle:
            self._lib.DestroyCapture(handle)
            self._handle = 0
