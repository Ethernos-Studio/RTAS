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
文字定位模块

提供图像中文字的检测和定位功能，支持VLM/LLM操作图像时获取精确像素位置

Author: RTAS Vision Team
Version: 1.0.0
"""

import cv2
import numpy as np
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from difflib import SequenceMatcher

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    """文字区域"""
    text: str                           # 识别的文字
    confidence: float                   # 置信度
    bbox: List[Tuple[int, int]]        # 边界框坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    center: Tuple[int, int]            # 中心点坐标
    area: int                          # 区域面积

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'area': self.area,
            'x': self.center[0],
            'y': self.center[1],
        }


@dataclass
class TextMatchResult:
    """文字匹配结果"""
    query: str                          # 查询文字
    matched_text: str                   # 匹配到的文字
    similarity: float                   # 相似度
    region: Optional[TextRegion]        # 文字区域
    match_type: str                     # 匹配类型: exact, partial, fuzzy

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'query': self.query,
            'matched_text': self.matched_text,
            'similarity': self.similarity,
            'match_type': self.match_type,
            'position': self.region.to_dict() if self.region else None,
            'x': self.region.center[0] if self.region else None,
            'y': self.region.center[1] if self.region else None,
        }


class TextLocator:
    """
    文字定位器

    检测图像中的文字并返回精确像素位置，供VLM/LLM使用
    """

    # 默认OCR配置
    DEFAULT_OCR_LANGUAGES = ['ch_sim', 'en']
    DEFAULT_CONFIDENCE_THRESHOLD = 0.3

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        confidence_threshold: float = 0.3,
        use_gpu: bool = False
    ):
        """
        初始化文字定位器

        Args:
            languages: OCR语言列表，默认 ['ch_sim', 'en']
            confidence_threshold: 置信度阈值
            use_gpu: 是否使用GPU
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR未安装，请运行: uv sync")

        self.languages = languages or self.DEFAULT_OCR_LANGUAGES
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu

        self._ocr_reader = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        初始化OCR引擎

        Returns:
            是否初始化成功
        """
        try:
            logger.info(f"初始化OCR引擎，语言: {self.languages}")
            self._ocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False
            )
            self._initialized = True
            logger.info("OCR引擎初始化完成")
            return True

        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            return False

    def detect_text(
        self,
        image: Union[np.ndarray, Path, str],
        detail: int = 1
    ) -> List[TextRegion]:
        """
        检测图像中的所有文字

        Args:
            image: 图像（numpy数组或路径）
            detail: 详细级别，0=简单，1=详细

        Returns:
            文字区域列表
        """
        if not self._initialized:
            if not self.initialize():
                return []

        try:
            # 加载图像
            if isinstance(image, (str, Path)):
                img = cv2.imread(str(image))
                if img is None:
                    logger.error(f"无法加载图像: {image}")
                    return []
            else:
                img = image.copy()

            # 运行OCR
            results = self._ocr_reader.readtext(img, detail=detail)

            # 解析结果
            regions = []
            for result in results:
                if detail == 0:
                    # 简单模式: [text, confidence]
                    text, confidence = result
                    bbox = []
                    center = (0, 0)
                    area = 0
                else:
                    # 详细模式: [bbox, text, confidence]
                    bbox, text, confidence = result
                    # 计算中心点
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
                    # 计算面积（使用多边形面积公式）
                    area = self._calculate_polygon_area(bbox)

                # 过滤低置信度
                if confidence >= self.confidence_threshold:
                    region = TextRegion(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox if isinstance(bbox, list) else [],
                        center=center,
                        area=area
                    )
                    regions.append(region)

            # 按面积排序（大的在前，通常是重要文字）
            regions.sort(key=lambda r: r.area, reverse=True)

            logger.debug(f"检测到 {len(regions)} 个文字区域")
            return regions

        except Exception as e:
            logger.error(f"文字检测失败: {e}")
            return []

    def locate_text(
        self,
        image: Union[np.ndarray, Path, str],
        query: str,
        match_mode: str = 'fuzzy',
        similarity_threshold: float = 0.6
    ) -> Optional[TextMatchResult]:
        """
        定位指定文字在图像中的位置

        Args:
            image: 图像
            query: 要查找的文字
            match_mode: 匹配模式: exact(精确), partial(部分), fuzzy(模糊)
            similarity_threshold: 相似度阈值

        Returns:
            匹配结果，未找到返回None
        """
        regions = self.detect_text(image)
        if not regions:
            return None

        best_match = None
        best_similarity = 0.0

        for region in regions:
            text = region.text

            if match_mode == 'exact':
                # 精确匹配
                if query == text:
                    return TextMatchResult(
                        query=query,
                        matched_text=text,
                        similarity=1.0,
                        region=region,
                        match_type='exact'
                    )

            elif match_mode == 'partial':
                # 部分匹配
                if query in text or text in query:
                    similarity = len(query) / max(len(query), len(text))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = TextMatchResult(
                            query=query,
                            matched_text=text,
                            similarity=similarity,
                            region=region,
                            match_type='partial'
                        )

            else:  # fuzzy
                # 模糊匹配
                similarity = self._calculate_similarity(query, text)
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_match = TextMatchResult(
                        query=query,
                        matched_text=text,
                        similarity=similarity,
                        region=region,
                        match_type='fuzzy'
                    )

        return best_match

    def locate_multiple(
        self,
        image: Union[np.ndarray, Path, str],
        queries: List[str],
        match_mode: str = 'fuzzy',
        similarity_threshold: float = 0.6
    ) -> Dict[str, Optional[TextMatchResult]]:
        """
        批量定位多个文字

        Args:
            image: 图像
            queries: 要查找的文字列表
            match_mode: 匹配模式
            similarity_threshold: 相似度阈值

        Returns:
            查询文字到匹配结果的映射
        """
        results = {}
        for query in queries:
            result = self.locate_text(image, query, match_mode, similarity_threshold)
            results[query] = result
        return results

    def get_all_text_positions(
        self,
        image: Union[np.ndarray, Path, str]
    ) -> List[Dict[str, Any]]:
        """
        获取图像中所有文字的位置信息

        Args:
            image: 图像

        Returns:
            文字位置列表，每个元素包含:
            - text: 文字内容
            - x, y: 中心点坐标
            - confidence: 置信度
            - bbox: 边界框
        """
        regions = self.detect_text(image)
        return [region.to_dict() for region in regions]

    def visualize_text_locations(
        self,
        image: Union[np.ndarray, Path, str],
        output_path: Optional[Path] = None,
        highlight_queries: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        可视化文字位置

        Args:
            image: 图像
            output_path: 输出路径（可选）
            highlight_queries: 要高亮显示的文字列表（可选）

        Returns:
            可视化图像
        """
        # 加载图像
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                return np.array([])
        else:
            img = image.copy()

        # 检测文字
        regions = self.detect_text(img)

        # 高亮匹配的文字
        highlight_set = set(highlight_queries or [])

        for region in regions:
            bbox = region.bbox
            is_highlight = any(q in region.text or region.text in q for q in highlight_set)

            # 绘制边界框
            color = (0, 255, 0) if is_highlight else (0, 165, 255)
            thickness = 3 if is_highlight else 2

            if len(bbox) >= 4:
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, color, thickness)

            # 绘制中心点
            cv2.circle(img, region.center, 5, (0, 0, 255), -1)

            # 绘制文字标签
            label = f"{region.text} ({region.confidence:.2f})"
            cv2.putText(img, label, (region.center[0] - 50, region.center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存
        if output_path:
            cv2.imwrite(str(output_path), img)

        return img

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            相似度 (0-1)
        """
        # 标准化字符串
        s1 = self._normalize_text(s1)
        s2 = self._normalize_text(s2)

        if not s1 or not s2:
            return 0.0

        # 使用SequenceMatcher计算相似度
        return SequenceMatcher(None, s1, s2).ratio()

    def _normalize_text(self, text: str) -> str:
        """
        标准化文本

        Args:
            text: 原始文本

        Returns:
            标准化后的文本
        """
        # 移除空格和特殊字符
        text = re.sub(r'\s+', '', text)
        # 转小写
        text = text.lower()
        return text.strip()

    def _calculate_polygon_area(self, bbox: List[Tuple[int, int]]) -> int:
        """
        计算多边形面积

        Args:
            bbox: 边界框坐标

        Returns:
            面积
        """
        if len(bbox) < 3:
            return 0

        # 使用鞋带公式
        area = 0
        n = len(bbox)
        for i in range(n):
            j = (i + 1) % n
            area += bbox[i][0] * bbox[j][1]
            area -= bbox[j][0] * bbox[i][1]

        return abs(area) // 2

    def shutdown(self):
        """关闭定位器"""
        self._ocr_reader = None
        self._initialized = False
        logger.info("文字定位器已关闭")


# 便捷函数
def locate_text_in_image(
    image_path: Union[str, Path],
    query: str,
    return_center: bool = True
) -> Optional[Union[Tuple[int, int], Dict[str, Any]]]:
    """
    便捷函数：定位图像中文字的位置

    Args:
        image_path: 图像路径
        query: 要查找的文字
        return_center: 是否只返回中心点坐标

    Returns:
        如果return_center为True，返回 (x, y) 坐标；
        否则返回完整的匹配结果字典
    """
    locator = TextLocator()
    if not locator.initialize():
        return None

    try:
        result = locator.locate_text(image_path, query)
        if result and result.region:
            if return_center:
                return result.region.center
            else:
                return result.to_dict()
        return None
    finally:
        locator.shutdown()


def get_all_text_in_image(
    image_path: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    便捷函数：获取图像中所有文字的位置

    Args:
        image_path: 图像路径

    Returns:
        文字位置列表
    """
    locator = TextLocator()
    if not locator.initialize():
        return []

    try:
        return locator.get_all_text_positions(image_path)
    finally:
        locator.shutdown()
