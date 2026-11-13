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

# @author error-0x12
# @brief Tesseract 后端

"""Tesseract 后端。

优先使用 ``pytesseract``（若已安装），否则直接调用 ``tesseract`` 可执行文件
输出 TSV 结果。两条路径都会把单词合并成行，并给出整行的边界框。

启用方式：安装 Tesseract-OCR（含 ``chi_sim`` 语言包），然后把
``tesseract.exe`` 加入 PATH，或设置 ``RTAS_TESSERACT_CMD``。
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .base import BackendInfo, OCRBackend, OCRText, clamp_confidence
from .deps import spec_exists

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "ch_sim": "chi_sim",
    "ch_tra": "chi_tra",
    "en": "eng",
    "ja": "jpn",
    "ko": "kor",
    "ru": "rus",
    "fr": "fra",
    "de": "deu",
}

_COMMON_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def find_tesseract() -> str | None:
    """定位 tesseract 可执行文件。"""
    explicit = os.environ.get("RTAS_TESSERACT_CMD")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _COMMON_PATHS:
        if Path(candidate).is_file():
            return candidate
    return None


class TesseractBackend(OCRBackend):
    name = "tesseract"
    display_name = "Tesseract"
    kind = "binary"
    supports_boxes = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cmd = find_tesseract()
        self.psm = str(self.options.get("psm", 6))

    @classmethod
    def probe(cls) -> BackendInfo:
        cmd = find_tesseract()
        if cmd:
            return BackendInfo(cls.name, cls.display_name, True, cls.kind, cmd)
        return BackendInfo(
            cls.name,
            cls.display_name,
            False,
            cls.kind,
            "未找到 tesseract.exe（安装后加入 PATH 或设置 RTAS_TESSERACT_CMD）",
        )

    def _tesseract_lang(self) -> str:
        langs = [_LANG_MAP.get(lang, lang) for lang in self.languages]
        return "+".join(dict.fromkeys(langs)) or "eng"

    def initialize(self) -> None:
        if not self.cmd:
            raise RuntimeError(
                "Tesseract 未安装或不在 PATH 中，请设置 RTAS_TESSERACT_CMD 指向 tesseract.exe"
            )
        self._initialized = True

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        self.initialize()
        if spec_exists("pytesseract"):
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = self.cmd
            data = pytesseract.image_to_data(
                image,
                lang=self._tesseract_lang(),
                config=f"--psm {self.psm}",
                output_type=pytesseract.Output.DICT,
            )
            return self._parse_tsv_dict(data)
        return self._recognize_cli(image)

    # -- pytesseract -----------------------------------------------------
    @staticmethod
    def _parse_tsv_dict(data: dict[str, list[Any]]) -> list[OCRText]:
        grouped: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(data.get("text", [])):
            if not str(word).strip():
                continue
            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            grouped.setdefault(key, []).append(i)

        results: list[OCRText] = []
        for indices in grouped.values():
            words = [str(data["text"][i]).strip() for i in indices]
            confs = [clamp_confidence(data["conf"][i]) for i in indices]
            left = min(int(data["left"][i]) for i in indices)
            top = min(int(data["top"][i]) for i in indices)
            right = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
            bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
            results.append(
                OCRText(
                    "".join(words) if _is_cjk(words) else " ".join(words),
                    sum(confs) / len(confs) if confs else 0.0,
                    [[left, top], [right, top], [right, bottom], [left, bottom]],
                )
            )
        return results

    # -- CLI -------------------------------------------------------------
    def _recognize_cli(self, image: np.ndarray) -> list[OCRText]:
        assert self.cmd is not None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            if not cv2.imwrite(tmp_path, image):
                raise RuntimeError("临时图像写入失败")
            creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
            proc = subprocess.run(
                [
                    self.cmd,
                    tmp_path,
                    "stdout",
                    "-l",
                    self._tesseract_lang(),
                    "--psm",
                    self.psm,
                    "tsv",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
                timeout=float(self.options.get("timeout", 60.0)),
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Tesseract 执行失败: {proc.stderr.strip()[:200]}")
            return self._parse_tsv_text(proc.stdout)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _parse_tsv_text(tsv: str) -> list[OCRText]:
        rows = [line.split("\t") for line in tsv.splitlines() if line.strip()]
        if not rows:
            return []
        header = rows[0]
        try:
            idx = {name: header.index(name) for name in
                   ("block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text")}
        except ValueError:
            return []

        grouped: dict[tuple[int, int, int], list[list[str]]] = {}
        for row in rows[1:]:
            if len(row) < len(header):
                continue
            if not row[idx["text"]].strip():
                continue
            key = (int(row[idx["block_num"]]), int(row[idx["par_num"]]), int(row[idx["line_num"]]))
            grouped.setdefault(key, []).append(row)

        results: list[OCRText] = []
        for group in grouped.values():
            words = [row[idx["text"]].strip() for row in group]
            confs = [clamp_confidence(row[idx["conf"]]) for row in group]
            left = min(int(row[idx["left"]]) for row in group)
            top = min(int(row[idx["top"]]) for row in group)
            right = max(int(row[idx["left"]]) + int(row[idx["width"]]) for row in group)
            bottom = max(int(row[idx["top"]]) + int(row[idx["height"]]) for row in group)
            results.append(
                OCRText(
                    "".join(words) if _is_cjk(words) else " ".join(words),
                    sum(confs) / len(confs) if confs else 0.0,
                    [[left, top], [right, top], [right, bottom], [left, bottom]],
                )
            )
        return results


def _is_cjk(words: list[str]) -> bool:
    """中文等无空格语言直接拼接，避免插入多余空格。"""
    for word in words:
        for ch in word:
            if "\u4e00" <= ch <= "\u9fff":
                return True
    return False
