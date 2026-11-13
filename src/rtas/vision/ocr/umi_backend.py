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
# @brief Umi-OCR 后端（HTTP）

"""Umi-OCR 后端。

Umi-OCR 是 Windows 上的本地 OCR 工具，内置 PaddleOCR/RapidOCR，
开启「HTTP 服务」后监听 ``http://127.0.0.1:1224``。RTAS 通过它的
``/api/ocr`` 接口识别，不需要在本项目里安装 Paddle 运行库。

启用方式：打开 Umi-OCR → 设置 → 全局设置 → HTTP 服务 → 开启。
可用环境变量 ``RTAS_UMI_URL`` 或 ``--ocr-url`` 指定地址。
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

import numpy as np

from .base import (
    BackendInfo,
    OCRBackend,
    OCRText,
    clamp_confidence,
    encode_png_base64,
    normalize_bbox,
)

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:1224"


class UmiOCRBackend(OCRBackend):
    name = "umi"
    display_name = "Umi-OCR"
    kind = "http"
    supports_boxes = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.url = (
            self.options.get("url")
            or os.environ.get("RTAS_UMI_URL")
            or DEFAULT_URL
        ).rstrip("/")
        self.timeout = float(self.options.get("timeout", 30.0))

    @classmethod
    def probe(cls) -> BackendInfo:
        url = os.environ.get("RTAS_UMI_URL", DEFAULT_URL)
        parsed = urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 1224
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return BackendInfo(cls.name, cls.display_name, True, cls.kind)
        except OSError:
            return BackendInfo(
                cls.name,
                cls.display_name,
                False,
                cls.kind,
                f"未连接到 {url}（请先启动 Umi-OCR 并开启 HTTP 服务）",
            )

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        import requests

        payload = {
            "base64": encode_png_base64(image),
            "options": {"data.format": "dict"},
        }
        resp = requests.post(f"{self.url}/api/ocr", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, dict) and body.get("code") not in (None, 100):
            raise RuntimeError(f"Umi-OCR 返回错误: {body.get('code')} {body.get('data')}")
        return self._parse(body)

    @staticmethod
    def _parse(body: Any) -> list[OCRText]:
        data = body.get("data") if isinstance(body, dict) else body
        results: list[OCRText] = []

        def add(text: Any, score: Any = 1.0, box: Any = None) -> None:
            text = str(text or "").strip()
            if text:
                results.append(OCRText(text, clamp_confidence(score), normalize_bbox(box)))

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    add(
                        item.get("text"),
                        item.get("score", item.get("confidence", 1.0)),
                        item.get("box") or item.get("bbox"),
                    )
                elif isinstance(item, (list, tuple)) and item:
                    add(item[0])
        elif isinstance(data, dict):
            # data.format=dict 的另一种形态：{"text": [...], "score": [...]}
            texts = data.get("text") or []
            scores = data.get("score") or []
            boxes = data.get("box") or []
            for i, text in enumerate(texts):
                add(
                    text,
                    scores[i] if i < len(scores) else 1.0,
                    boxes[i] if i < len(boxes) else None,
                )
        elif isinstance(data, str):
            for line in data.splitlines():
                add(line)
        return results
