# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Ethernos Studio
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCR 后端适配层测试（不加载任何模型）。"""

from __future__ import annotations

import numpy as np
import pytest

from rtas.vision.ocr import (
    BACKENDS,
    BackendInfo,
    OCRBackend,
    OCRText,
    engine_names,
    list_backends,
)
from rtas.vision.ocr.base import clamp_confidence, full_image_bbox, normalize_bbox
from rtas.vision.ocr.paddleocr_backend import PaddleOCRBackend
from rtas.vision.ocr.tesseract_backend import TesseractBackend
from rtas.vision.ocr.umi_backend import UmiOCRBackend


class DummyBackend(OCRBackend):
    name = "dummy"
    display_name = "Dummy"

    @classmethod
    def probe(cls) -> BackendInfo:
        return BackendInfo(cls.name, cls.display_name, True)

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        return [
            OCRText("剩余可放置角色", 0.91, [[0, 0], [10, 0], [10, 5], [0, 5]]),
            OCRText("5", 0.75, [[20, 0], [30, 0], [30, 5], [20, 5]]),
        ]


def test_readtext_detail_shapes():
    backend = DummyBackend()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    assert backend.readtext(image, detail=0) == ["剩余可放置角色", "5"]

    detailed = backend.readtext(image, detail=1)
    assert detailed[0][1] == "剩余可放置角色"
    assert detailed[0][2] == pytest.approx(0.91)
    assert detailed[0][0][0] == [0, 0]


def test_ocr_text_center_and_bbox_helpers():
    text = OCRText("x", 0.5, [[0, 0], [10, 0], [10, 20], [0, 20]])
    assert text.center == (5, 10)
    assert OCRText("x", 0.5).center is None

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert full_image_bbox(image) == [[0, 0], [199, 0], [199, 99], [0, 99]]

    assert normalize_bbox([0, 1, 2, 3]) == [[0, 1], [2, 1], [2, 3], [0, 3]]
    assert normalize_bbox(np.array([[5, 6], [7, 8]])) == [[5, 6], [7, 8]]
    assert normalize_bbox(None) is None


def test_clamp_confidence():
    assert clamp_confidence(0.5) == 0.5
    assert clamp_confidence(95) == pytest.approx(0.95)  # Tesseract 0~100
    assert clamp_confidence(-1) == 0.0
    assert clamp_confidence("bad") == 0.0


def test_paddle_parse_v2():
    raw = [
        [
            [[[0, 0], [10, 0], [10, 5], [0, 5]], ("剩余可放置角色", 0.93)],
            [[[0, 8], [10, 8], [10, 12], [0, 12]], ("5", 0.88)],
        ]
    ]
    results = PaddleOCRBackend._parse(raw)
    assert [r.text for r in results] == ["剩余可放置角色", "5"]
    assert results[0].bbox == [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert results[0].confidence == pytest.approx(0.93)


def test_paddle_parse_v3():
    raw = [
        {
            "rec_texts": ["源石交易所", "组合包"],
            "rec_scores": [0.97, 0.95],
            "rec_polys": [
                [[0, 0], [10, 0], [10, 5], [0, 5]],
                [[0, 8], [10, 8], [10, 12], [0, 12]],
            ],
        }
    ]
    results = PaddleOCRBackend._parse(raw)
    assert [r.text for r in results] == ["源石交易所", "组合包"]
    assert results[1].confidence == pytest.approx(0.95)


def test_tesseract_tsv_parse():
    ts = "\t"
    header = ts.join([
        "level", "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text",
    ])
    rows = [
        ts.join(["5", "1", "1", "1", "1", "1", "100", "50", "40", "20", "91.5", "剩余"]),
        ts.join(["5", "1", "1", "1", "1", "2", "140", "50", "60", "20", "80.0", "可放置"]),
        ts.join(["5", "1", "1", "1", "2", "1", "100", "80", "10", "20", "70.0", "5"]),
    ]
    results = TesseractBackend._parse_tsv_text("\n".join([header, *rows]))
    assert len(results) == 2
    assert results[0].text == "剩余可放置"
    assert results[0].bbox == [[100, 50], [200, 50], [200, 70], [100, 70]]
    assert results[0].confidence == pytest.approx(0.8575, abs=1e-4)
    assert results[1].text == "5"


def test_tesseract_tsv_skips_empty_text():
    header = "level\tblock_num\tpar_num\tline_num\tleft\ttop\twidth\theight\tconf\ttext"
    row = "5\t1\t1\t1\t0\t0\t10\t10\t90\t"
    assert TesseractBackend._parse_tsv_text(header + "\n" + row) == []


def test_umi_parse_variants():
    dict_list = {
        "code": 100,
        "data": [
            {"text": "终端", "score": 0.99, "box": [[1, 2], [3, 2], [3, 4], [1, 4]]},
            {"text": "", "score": 0.5},
        ],
    }
    results = UmiOCRBackend._parse(dict_list)
    assert len(results) == 1
    assert results[0].text == "终端"
    assert results[0].bbox == [[1, 2], [3, 2], [3, 4], [1, 4]]

    string_data = {"code": 100, "data": "首页\n干员"}
    assert [r.text for r in UmiOCRBackend._parse(string_data)] == ["首页", "干员"]


def test_registry_and_engine_names():
    names = engine_names()
    assert names[0] == "auto"
    for expected in ("easyocr", "paddleocr", "umi", "tesseract"):
        assert expected in names
        assert expected in BACKENDS

    infos = list_backends()
    assert {info.name for info in infos} == set(BACKENDS)


def test_unknown_engine_rejected():
    from rtas.vision.ocr import create_backend

    with pytest.raises(ValueError):
        create_backend("nope")
