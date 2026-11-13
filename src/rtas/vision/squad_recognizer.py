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

"""
编队识别模块

从编队编辑/编队选择截图中识别 6x2 共 12 张干员卡片：

- 卡片网格：按图像比例定位（实测 1264x711 -> 6 列 x 2 行），任意 16:9 分辨率自适应；
- 名称：底部名称条放大 6 倍、多套预处理各跑一次 OCR，保留多个候选，
  交给 ``rtas.data.operator_matcher`` 用干员库做模糊校正；
- 等级：等级圆圈内数字，放大 + 二值化后用纯数字白名单识别；
- 精英化：模板匹配（模板缺失时按 E0 处理，不阻塞其它字段）。

Author: RTAS Vision Team
Version: 2.0.0
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from PIL import Image, ImageDraw, ImageFont

# torch / easyocr 延迟加载：这里只探测是否安装，真正导入推迟到使用时
from .ocr_runtime import (
    TORCH_AVAILABLE,
    create_reader,
    cuda_available,
)

logger = logging.getLogger(__name__)

#: 精英化图标模板缓存（解包的 40x30 / 40x38 渲染图，只保留实体像素掩码）
_ELITE_TEMPLATE_CACHE: Dict[int, np.ndarray] = {}


class EliteLevel(Enum):
    """精英化等级"""
    E0 = 0
    E1 = 1
    E2 = 2


@dataclass
class OperatorCard:
    """干员卡片信息"""
    # 位置信息
    position: Tuple[int, int, int, int]  # x, y, w, h
    grid_position: Tuple[int, int]  # row, col

    # 识别结果
    elite_level: EliteLevel = EliteLevel.E0
    level: int = 1
    name: str = ""
    name_confidence: float = 0.0

    # 名称候选（按置信度降序），供数据库模糊校正使用
    name_candidates: List[Tuple[str, float]] = field(default_factory=list)

    # 原始数据
    raw_elite_icon: Optional[np.ndarray] = None
    raw_level_image: Optional[np.ndarray] = None
    raw_name_image: Optional[np.ndarray] = None

    # 数据库查询结果
    operator_info: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        elite_str = f"E{self.elite_level.value}" if self.elite_level.value > 0 else "E0"
        return f"{self.name} [{elite_str} Lv.{self.level}]"


@dataclass
class SquadConfig:
    """编队识别配置。

    网格与卡片内区域都用**比例**表示，因此 720p / 1080p / 4K 截图通用。
    默认值来自 1264x711 实测（Photoshop 量取）：
    第一张卡 (152,106)-(265,354)，列间距 140，行间距 254。
    """

    # -- 卡片网格（相对整图宽高）--------------------------------------
    grid_origin_ratio: Tuple[float, float] = (152 / 1264, 106 / 711)
    grid_card_ratio: Tuple[float, float] = (113 / 1264, 248 / 711)
    grid_spacing_ratio: Tuple[float, float] = (140 / 1264, 254 / 711)
    cards_per_row: int = 6
    max_rows: int = 2

    # -- 卡片内区域（相对卡片宽高）------------------------------------
    # 名称条：卡片底部约 14%（实测 y 330~354）
    name_region_ratio: Tuple[float, float, float, float] = (0.0, 0.86, 1.0, 0.14)
    # 等级圆圈：卡片左侧 40%，y 290~330
    level_region_ratio: Tuple[float, float, float, float] = (0.0, 0.735, 0.40, 0.17)
    # 精英化徽章：卡片左侧中下部（实测 (152,245)~(200,290)，含 E2 第二层刀的下沿）
    elite_region_ratio: Tuple[float, float, float, float] = (0.0, 0.5524, 0.4425, 0.238)

    # -- 精英化徽章 ----------------------------------------------------
    # 解包出来的标准图标（带 alpha，40x30 / 40x38）：E0 没有徽章，
    # E1 一层交叉双刀，E2 是**同一份刀再叠一份**（大小相同、错位重叠）。
    elite_templates: Dict[int, Path] = field(default_factory=lambda: {
        1: Path("templates/icons/elitism/level_1_render.png"),
        2: Path("templates/icons/elitism/level_2_render.png"),
    })

    # -- OCR -----------------------------------------------------------
    ocr_languages: List[str] = field(default_factory=lambda: ['ch_sim', 'en'])
    ocr_gpu: Union[bool, str] = 'auto'
    name_confidence_threshold: float = 0.15  # 低于此值仍保留候选，交给数据库校正

    @classmethod
    def preset_squad_selection(cls) -> "SquadConfig":
        """编队选择界面（代理作战）：与编辑界面同一套 6x2 网格。"""
        return cls()

    @classmethod
    def preset_squad_edit(cls) -> "SquadConfig":
        """编队编辑界面：6 列 x 2 行。"""
        return cls()


class SquadRecognizer:
    """编队识别器：从整张截图中识别 12 张卡片。"""

    NAME_UPSCALE = 5      # 名称条放大倍数
    LEVEL_UPSCALE = 5     # 等级区域放大倍数

    def __init__(self, config: Optional[SquadConfig] = None):
        self.config = config or SquadConfig()
        self._ocr_reader = None
        self._initialized = False

    # -- 初始化 -----------------------------------------------------------
    def _check_gpu_available(self) -> bool:
        if not TORCH_AVAILABLE:
            return False
        try:
            return cuda_available()
        except Exception:
            return False

    def initialize(self) -> bool:
        try:
            logger.info("初始化编队识别器...")
            use_gpu = False
            if self.config.ocr_gpu == 'auto':
                use_gpu = self._check_gpu_available()
                logger.info("自动检测到GPU，启用GPU加速" if use_gpu else "未检测到GPU，使用CPU")
            elif self.config.ocr_gpu is True:
                use_gpu = self._check_gpu_available()
                if not use_gpu:
                    logger.warning("强制使用GPU但GPU不可用，将使用CPU")

            try:
                self._ocr_reader = create_reader(self.config.ocr_languages, gpu=use_gpu)
                logger.info(f"EasyOCR初始化完成 (GPU={use_gpu})")
            except ImportError:
                logger.error("EasyOCR未安装，请执行: uv sync")
                return False

            self._initialized = True
            logger.info("编队识别器初始化完成")
            return True
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False

    # -- 网格 -------------------------------------------------------------
    def card_rects(self, image: np.ndarray) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
        """按比例算出 6x2 卡片的像素矩形。"""
        height, width = image.shape[:2]
        origin_x = self.config.grid_origin_ratio[0] * width
        origin_y = self.config.grid_origin_ratio[1] * height
        card_w = self.config.grid_card_ratio[0] * width
        card_h = self.config.grid_card_ratio[1] * height
        step_x = self.config.grid_spacing_ratio[0] * width
        step_y = self.config.grid_spacing_ratio[1] * height

        rects: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
        for row in range(self.config.max_rows):
            for col in range(self.config.cards_per_row):
                x = int(round(origin_x + col * step_x))
                y = int(round(origin_y + row * step_y))
                w, h = int(round(card_w)), int(round(card_h))
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    logger.debug(f"位置({row},{col})超出图像范围")
                    continue
                rects.append((row, col, (x, y, w, h)))
        return rects

    def _crop_ratio(self, card_image: np.ndarray,
                    ratio: Tuple[float, float, float, float]) -> np.ndarray:
        h, w = card_image.shape[:2]
        x0 = int(round(ratio[0] * w))
        y0 = int(round(ratio[1] * h))
        x1 = int(round((ratio[0] + ratio[2]) * w))
        y1 = int(round((ratio[1] + ratio[3]) * h))
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))
        return card_image[y0:y1, x0:x1]

    # -- 主流程 -----------------------------------------------------------
    def recognize_squad(self, image: np.ndarray) -> List[OperatorCard]:
        """识别整张截图里的编队卡片。"""
        if not self._initialized:
            raise RuntimeError("识别器未初始化")

        height, width = image.shape[:2]
        logger.info(f"图像尺寸: {width}x{height}，按 6x2 网格定位卡片")

        # 1) 定位所有有效卡片（空位跳过）
        entries: List[Tuple[Tuple[int, int], Tuple[int, int, int, int], np.ndarray]] = []
        for row, col, rect in self.card_rects(image):
            x, y, w, h = rect
            card_image = image[y:y + h, x:x + w]
            if not self._is_valid_card(card_image):
                logger.debug(f"位置({row},{col})是空位")
                continue
            entries.append(((row, col), rect, card_image))

        if not entries:
            logger.warning("没有找到任何卡片")
            return []

        card_images = [item[2] for item in entries]

        # 2) 名称/等级整行批量 OCR：12 张拼成一张图，一次调用识别完
        name_candidates = self._batch_names(card_images)
        levels = self._batch_levels(card_images)

        # 3) 组装结果
        cards: List[OperatorCard] = []
        for index, (grid, rect, card_image) in enumerate(entries):
            card = OperatorCard(position=rect, grid_position=grid)
            card.raw_elite_icon = self._crop_ratio(card_image, self.config.elite_region_ratio)
            card.raw_level_image = self._crop_ratio(card_image, self.config.level_region_ratio)
            card.raw_name_image = self._crop_ratio(card_image, self.config.name_region_ratio)
            card.name_candidates = name_candidates[index]
            if card.name_candidates:
                card.name, card.name_confidence = card.name_candidates[0]
            card.level = levels[index]
            card.elite_level = self._recognize_elite_level(card_image)
            if not card.name and not card.level:
                logger.debug(f"位置{grid}既无名称也无等级，判定为助战位/按钮，跳过")
                continue
            if any(keyword in card.name for keyword in ("助战", "开始行动", "代理")):
                logger.debug(f"位置{grid}是「{card.name}」，不是干员卡片，跳过")
                continue
            cards.append(card)
            suffix = (f" (候选: {', '.join(n for n, _ in card.name_candidates[:3])})"
                      if len(card.name_candidates) > 1 else "")
            logger.info(f"识别到干员: {card}{suffix}")

        logger.info(f"共识别到 {len(cards)} 个干员")
        return cards

    #: 空位判定：等级圆圈或名称条的“发白像素”占比，低于阈值就当作空位/按钮
    LEVEL_BRIGHT_MIN = 0.008
    NAME_BRIGHT_MIN = 0.10

    @staticmethod
    def _bright_ratio(value: np.ndarray, saturation: np.ndarray,
                      ratio: Tuple[float, float, float, float]) -> float:
        """区域里「又亮又不花」的像素占比（白字/白圈占比）。"""
        height, width = value.shape[:2]
        x0, y0, w, h = ratio
        patch_v = value[int(y0 * height):int((y0 + h) * height),
                        int(x0 * width):int((x0 + w) * width)]
        patch_s = saturation[int(y0 * height):int((y0 + h) * height),
                             int(x0 * width):int((x0 + w) * width)]
        if patch_v.size == 0:
            return 0.0
        return float(((patch_v > 190) & (patch_s < 70)).mean())

    def _is_valid_card(self, card_image: np.ndarray) -> bool:
        """空位判定：空位有斜纹和「+」，光靠边缘密度挡不住，还要看等级圈/名称条。"""
        gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY) \
            if card_image.ndim == 3 else card_image
        std = float(np.std(gray))
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)
        if std <= 12 or edge_ratio <= 0.02:
            return False

        if card_image.ndim == 2:
            return True
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        value, saturation = hsv[:, :, 2], hsv[:, :, 1]
        # 真卡必有等级圆圈；名称条发白也可以兜底（等级数字被挡时）
        level = self._bright_ratio(value, saturation, self.config.level_region_ratio)
        name = self._bright_ratio(value, saturation, self.config.name_region_ratio)
        return level >= self.LEVEL_BRIGHT_MIN or name >= self.NAME_BRIGHT_MIN

    # -- 批量 OCR ---------------------------------------------------------
    def _ocr_montage(self, montage: np.ndarray,
                     allowlist: Optional[str] = None) -> List[Any]:
        """对整条拼接图做一次 OCR；失败时返回空列表。"""
        if self._ocr_reader is None:
            return []
        try:
            if allowlist:
                return self._ocr_reader.readtext(
                    montage, detail=1, paragraph=False, allowlist=allowlist)
            return self._ocr_reader.readtext(montage, detail=1, paragraph=False)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"批量 OCR 失败: {exc}")
            return []

    #: 每次拼多少张卡：太高会被 EasyOCR 的 canvas_size 缩放，反而更慢更糊
    MONTAGE_CHUNK = 6

    def _montage(self, strips: List[np.ndarray], upscale: int,
                 interpolation: int, gap: int = 14) -> Tuple[np.ndarray, int]:
        """把多张等宽小图纵向堆叠成一张（放大后），并返回单条步长。"""
        scaled = [cv2.resize(strip, None, fx=upscale, fy=upscale,
                             interpolation=interpolation) for strip in strips]
        height = scaled[0].shape[0]
        width = scaled[0].shape[1]
        stride = height + gap
        canvas = np.zeros((stride * len(scaled), width), dtype=np.uint8)
        for index, strip in enumerate(scaled):
            top = index * stride
            canvas[top:top + height, :] = strip[:height, :width]
        return canvas, stride

    @staticmethod
    def _group_boxes(detections: List[Any], stride: int,
                     count: int) -> List[List[Tuple[str, float]]]:
        """按纵向位置把 OCR 结果分回各自的卡片。"""
        groups: List[List[Tuple[str, float]]] = [[] for _ in range(count)]
        for detection in detections:
            try:
                box, text, confidence = detection[0], detection[1], detection[2]
            except (IndexError, TypeError):
                continue
            ys = [point[1] for point in box]
            center = (min(ys) + max(ys)) / 2.0
            index = int(center // stride)
            if 0 <= index < count:
                groups[index].append((str(text), float(confidence)))
        return groups

    def _batch_field(
        self,
        strips: List[np.ndarray],
        upscale: int,
        interpolation: int,
        variants: Tuple[str, ...],
        collect: Any,
        allowlist: Optional[str] = None,
    ) -> None:
        """把 strips 分块拼接后 OCR，按块回调 collect(index, [(text, conf), ...])。"""
        total = len(strips)
        for start in range(0, total, self.MONTAGE_CHUNK):
            chunk = strips[start:start + self.MONTAGE_CHUNK]
            for kind in variants:
                prepared = [self._prep(strip, kind) for strip in chunk]
                montage, stride = self._montage(prepared, upscale, interpolation)
                detections = self._ocr_montage(montage, allowlist=allowlist)
                groups = self._group_boxes(detections, stride, len(chunk))
                for offset, items in enumerate(groups):
                    collect(start + offset, items)

    def _prep(self, strip: np.ndarray, kind: str, clip: float = 3.0,
              tile: int = 8) -> np.ndarray:
        """按 kind 生成一种预处理图（灰度）。"""
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
        if kind == "gray":
            return gray
        if kind == "clahe":
            return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
        if kind == "bright":
            _, binary = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
            return cv2.bitwise_not(binary)
        # otsu
        enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(binary)

    # -- 名称 -------------------------------------------------------------
    #: 名称条取两段：下面一段保证字形完整，上面一段容忍纵向偏移
    NAME_BANDS: Tuple[Tuple[float, float], ...] = ((0.78, 1.0),)
    NAME_VARIANTS = ("gray", "bright", "clahe")
    NOISE_NAME = re.compile(r'^[A-Z0-9]{1,5}$')  # ORIG / CRC / T2 这类图标文字

    def _batch_names(self, cards: List[np.ndarray]) -> List[List[Tuple[str, float]]]:
        count = len(cards)
        pool: List[Dict[str, float]] = [dict() for _ in range(count)]

        def collect(index: int, items: List[Tuple[str, float]]) -> None:
            for text, confidence in items:
                cleaned = self._clean_name(text)
                if self._is_noise_name(cleaned):
                    continue
                if confidence > pool[index].get(cleaned, 0.0):
                    pool[index][cleaned] = confidence

        for band_start, band_end in self.NAME_BANDS:
            strips = [
                self._crop_ratio(card, (0.0, band_start, 1.0, band_end - band_start))
                for card in cards
            ]
            self._batch_field(strips, self.NAME_UPSCALE, cv2.INTER_LANCZOS4,
                              self.NAME_VARIANTS, collect)

        return [sorted(item.items(), key=lambda kv: kv[1], reverse=True)[:5]
                for item in pool]

    def _clean_name(self, name: str) -> str:
        """去掉 OCR 常见的干扰字符，保留中英文数字与间隔号。"""
        name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9·]', '', name)
        return name.strip('·')

    def _is_noise_name(self, text: str) -> bool:
        """过滤图标文字/纯数字之类的噪声候选。"""
        if not text or len(text) > 12:
            return True
        if text.isdigit():
            return True
        return bool(self.NOISE_NAME.match(text))

    # -- 等级 -------------------------------------------------------------
    LEVEL_VARIANTS = ("gray", "bright")

    def _batch_levels(self, cards: List[np.ndarray]) -> List[int]:
        count = len(cards)
        best_levels = [0] * count
        best_confidence = [0.0] * count

        def collect(index: int, items: List[Tuple[str, float]]) -> None:
            for text, confidence in items:
                numbers = re.findall(r'\d+', text)
                if not numbers:
                    continue
                level = int(numbers[0])
                if 1 <= level <= 90 and confidence > best_confidence[index]:
                    best_levels[index] = level
                    best_confidence[index] = confidence

        strips = [self._crop_ratio(card, self.config.level_region_ratio) for card in cards]
        self._batch_field(strips, self.LEVEL_UPSCALE, cv2.INTER_CUBIC,
                          self.LEVEL_VARIANTS, collect, allowlist="0123456789")
        return best_levels

    # -- 精英化 -----------------------------------------------------------
    #: 徽章搜索区域（相对卡片：左、上、宽、高）。实测徽章 x≈0.09~0.12、
    #: y≈0.59~0.65、宽≈0.195 卡宽，这里留余量给不同分辨率与界面缩放。
    ELITE_BAND = (0.02, 0.54, 0.45, 0.24)
    #: 模板宽 40px 对应 113px 宽卡片时的缩放（实测 0.73），按卡宽等比换算
    ELITE_SCALE_BASE = 0.73
    ELITE_SCALE_SPAN = 0.08
    ELITE_MIN_WHITENESS = 0.25   # 徽章区域白色占比下限：低于此值说明压根没徽章
    ELITE_MIN_DARK = 0.35        # 暗像素占比下限：整片白立绘会被挡掉
    ELITE_E2_RATIO = 0.35        # 第二层白度 / 第一层白度，超过即 E2

    def elite_templates(self) -> Dict[int, np.ndarray]:
        """加载解包的精英化图标，取实体像素的二值掩码（进程内缓存）。"""
        if _ELITE_TEMPLATE_CACHE:
            return _ELITE_TEMPLATE_CACHE
        for level, relative in self.config.elite_templates.items():
            path = Path(relative)
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
            if image is None or image.ndim != 3 or image.shape[2] != 4:
                logger.warning(f"精英化模板不可用: {path}")
                continue
            _ELITE_TEMPLATE_CACHE[level] = (image[:, :, 3] > 110).astype(np.float32)
        return _ELITE_TEMPLATE_CACHE

    @staticmethod
    def _whiteness(card_image: np.ndarray) -> np.ndarray:
        """转成 0~1 的「白度图」：徽章是发白的刀，用白度比灰度更好对齐。"""
        hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32)
        return (np.clip((value - 170.0) / 80.0, 0, 1) * (saturation < 80)).astype(np.float32)

    def _locate_badge(self, roi: np.ndarray, templates: Dict[int, np.ndarray],
                      card_width: int) -> Optional[Tuple[float, Tuple[int, int]]]:
        """在徽章带里定位图标：用两层的并集做模板，避免偏向 E1 或 E2。"""
        first, second = templates[1], templates[2]
        height = max(first.shape[0], second.shape[0])
        outline = np.zeros((height, first.shape[1]), np.float32)
        outline[:first.shape[0]] = first
        outline = np.maximum(outline, second)

        base = self.ELITE_SCALE_BASE * card_width / 113.0
        unit = card_width / 113.0 * 0.01
        steps = int(round(self.ELITE_SCALE_SPAN / 0.01))
        best: Tuple[float, float, Tuple[int, int]] = (-2.0, base, (0, 0))
        for index in range(-steps, steps + 1):
            scale = base + index * unit
            width = int(round(first.shape[1] * scale))
            tall = int(round(height * scale))
            if width < 8 or tall < 8 or width >= roi.shape[1] or tall >= roi.shape[0]:
                continue
            template = cv2.resize(outline, (width, tall), interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            if score > best[0]:
                best = (float(score), scale, location)
        if best[0] < -1:
            return None
        return best[1], best[2]

    def _recognize_elite_level(self, card_image: np.ndarray) -> EliteLevel:
        """按解包的精英化图标判断 E0 / E1 / E2。

        图标是「交叉双刀」：E0 没有；E1 只有一份；E2 是**同一份再叠一份**
        （两块大小完全一样、上下错位重叠），所以放大才能分辨，缩到 20px 会并成一团。
        判定分三步：
        1. 用两层的并集模板在徽章带里定位（同时定出缩放）；
        2. 在 E1 的高度范围内量化白度与暗像素占比 —— 判断「这里到底有没有徽章」；
        3. 看 E1 之下那一段（只有 E2 才有的第二层），白度显著更高就是 E2。
        """
        if card_image.size == 0:
            return EliteLevel.E0
        templates = self.elite_templates()
        if 1 not in templates or 2 not in templates:
            return EliteLevel.E0

        card_height, card_width = card_image.shape[:2]
        whiteness = self._whiteness(card_image)
        left = int(round(card_width * self.ELITE_BAND[0]))
        top = int(round(card_height * self.ELITE_BAND[1]))
        right = min(card_width, left + int(round(card_width * self.ELITE_BAND[2])))
        bottom = min(card_height, top + int(round(card_height * self.ELITE_BAND[3])))
        roi = whiteness[top:bottom, left:right]
        if roi.size == 0:
            return EliteLevel.E0

        located = self._locate_badge(roi, templates, card_width)
        if located is None:
            return EliteLevel.E0
        scale, (offset_x, offset_y) = located

        width = int(round(templates[1].shape[1] * scale))
        single = int(round(templates[1].shape[0] * scale))
        double = int(round(templates[2].shape[0] * scale))
        px, py = left + offset_x, top + offset_y
        upper = whiteness[py:py + single, px:px + width]
        below = whiteness[py + single:py + double, px:px + width]
        if upper.size == 0 or below.size == 0:
            return EliteLevel.E0

        brightness = float(upper.mean())
        dark = float((upper < 0.15).mean())
        if brightness < self.ELITE_MIN_WHITENESS or dark < self.ELITE_MIN_DARK:
            return EliteLevel.E0  # 没有徽章，或是一整片白立绘而非交叉双刀
        if float(below.mean()) / max(brightness, 1e-3) >= self.ELITE_E2_RATIO:
            return EliteLevel.E2
        return EliteLevel.E1

    # -- 可视化（保留原实现，改为用卡片的实际框）---------------------------
    def visualize_result(
        self,
        image: np.ndarray,
        cards: List[OperatorCard],
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        vis_image = image.copy()
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(vis_image_rgb)
        draw = ImageDraw.Draw(pil_image)

        font = None
        for font_path in ("C:/Windows/Fonts/simhei.ttf",
                          "C:/Windows/Fonts/simsun.ttc",
                          "C:/Windows/Fonts/msyh.ttc"):
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except Exception:  # noqa: BLE001
                continue
        if font is None:
            font = ImageFont.load_default()

        for card in cards:
            x, y, w, h = card.position
            color = (0, 255, 0) if card.name_confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)

            elite_str = f"E{card.elite_level.value}"
            lines = [f"{card.name}", f"[{elite_str} Lv.{card.level}]"]
            if card.operator_info:
                stars = card.operator_info.get('rarity', 0)
                lines.append('★' * stars if stars else '')
                lines.append(f"{card.operator_info.get('profession', '')} | "
                             f"{card.operator_info.get('position', '')}")
                lines.append(f"费用: {card.operator_info.get('cost', 0)}")

            line_height, padding = 22, 5
            max_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
            total_height = len(lines) * line_height + padding * 2
            draw.rectangle([(x, y - total_height), (x + max_width, y)],
                           fill=(color[2], color[1], color[0]))
            for i, line in enumerate(lines):
                draw.text((x, y - total_height + padding + i * line_height),
                          line, font=font, fill=(255, 255, 255))

        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"可视化结果已保存: {output_path}")
        return vis_image

    def shutdown(self) -> None:
        self._ocr_reader = None
        self._initialized = False
        logger.info("编队识别器已关闭")
