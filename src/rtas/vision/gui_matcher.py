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
GUI界面匹配模块 - 基于OpenCV模板匹配和OCR文字识别

本模块提供两种界面元素定位方式：
1. 模板匹配：适用于固定样式的按钮和图标
2. OCR文字识别：适用于主界面等有主题变化但文字固定的场景

Author: RTAS Vision Team
Version: 1.0.0
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
import time
import threading

# 导入EasyOCR
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

class MatchMethod(Enum):
    """匹配方法枚举"""
    TEMPLATE = auto()      # 模板匹配
    OCR = auto()           # OCR文字识别
    HYBRID = auto()        # 混合模式


@dataclass
class MatchResult:
    """
    匹配结果数据类

    Attributes:
        name: 元素名称
        position: 匹配位置 (x, y, width, height)
        confidence: 置信度 [0.0, 1.0]
        method: 使用的匹配方法
        matched_text: 识别的文字（OCR模式下）
        template_path: 模板路径（模板匹配模式下）
        center: 中心点坐标 (x, y)
    """
    name: str
    position: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    method: MatchMethod
    matched_text: Optional[str] = None
    template_path: Optional[str] = None

    @property
    def center(self) -> Tuple[int, int]:
        """计算中心点坐标"""
        x, y, w, h = self.position
        return (x + w // 2, y + h // 2)

    @property
    def top_left(self) -> Tuple[int, int]:
        """左上角坐标"""
        return (self.position[0], self.position[1])

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """右下角坐标"""
        x, y, w, h = self.position
        return (x + w, y + h)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'position': self.position,
            'confidence': self.confidence,
            'method': self.method.name,
            'matched_text': self.matched_text,
            'template_path': self.template_path,
            'center': self.center
        }


@dataclass
class TemplateConfig:
    """模板配置"""
    name: str
    path: Path
    threshold: float = 0.8
    multi_scale: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    scale_steps: int = 20


@dataclass
class OCRConfig:
    """OCR配置"""
    target_texts: List[str]                    # 目标文字列表
    region: Optional[Tuple[float, float, float, float]] = None  # 搜索区域 (相对坐标)
    confidence_threshold: float = 0.6
    languages: List[str] = field(default_factory=lambda: ['ch_sim', 'en'])


@dataclass
class GUIMatcherConfig:
    """GUI匹配器配置"""
    templates_dir: Path = Path("templates/btn")
    use_gpu: bool = True
    ocr_timeout: float = 5.0
    default_template_threshold: float = 0.8
    default_ocr_confidence: float = 0.6
    enable_cache: bool = True
    cache_size: int = 100


# =============================================================================
# 模板匹配器
# =============================================================================

class TemplateMatcher:
    """
    模板匹配器

    基于OpenCV的模板匹配算法，支持多尺度匹配以适应不同分辨率。

    算法说明：
    - 使用归一化相关系数 (NCC) 进行匹配
    - 支持多尺度搜索以适应不同分辨率
    - 使用非极大值抑制 (NMS) 去除重叠匹配
    """

    def __init__(self, config: GUIMatcherConfig):
        self.config = config
        self._template_cache: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

    def load_template(self, template_path: Path) -> Optional[np.ndarray]:
        """
        加载模板图像

        Args:
            template_path: 模板文件路径

        Returns:
            模板图像，加载失败返回None
        """
        path_str = str(template_path)

        # 检查缓存
        if self.config.enable_cache and path_str in self._template_cache:
            return self._template_cache[path_str]

        # 加载图像
        template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        if template is None:
            logger.error(f"无法加载模板: {template_path}")
            return None

        # 存入缓存
        if self.config.enable_cache:
            with self._lock:
                if len(self._template_cache) >= self.config.cache_size:
                    # LRU: 移除最早的项
                    oldest = next(iter(self._template_cache))
                    del self._template_cache[oldest]
                self._template_cache[path_str] = template

        return template

    def match(
        self,
        image: np.ndarray,
        template: np.ndarray,
        threshold: Optional[float] = None,
        multi_scale: bool = True
    ) -> Optional[MatchResult]:
        """
        执行模板匹配

        Args:
            image: 搜索图像
            template: 模板图像
            threshold: 匹配阈值，默认使用配置值
            multi_scale: 是否使用多尺度匹配

        Returns:
            最佳匹配结果，未找到返回None
        """
        if threshold is None:
            threshold = self.config.default_template_threshold

        if multi_scale:
            return self._match_multi_scale(image, template, threshold)
        else:
            return self._match_single_scale(image, template, threshold)

    def _match_single_scale(
        self,
        image: np.ndarray,
        template: np.ndarray,
        threshold: float
    ) -> Optional[MatchResult]:
        """单尺度匹配"""
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 模板不能比图像大
        if tmpl_gray.shape[0] > img_gray.shape[0] or tmpl_gray.shape[1] > img_gray.shape[1]:
            return None

        # 执行匹配
        result = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = tmpl_gray.shape[:2]
            return MatchResult(
                name="",
                position=(max_loc[0], max_loc[1], w, h),
                confidence=float(max_val),
                method=MatchMethod.TEMPLATE
            )

        return None

    def _match_multi_scale(
        self,
        image: np.ndarray,
        template: np.ndarray,
        threshold: float
    ) -> Optional[MatchResult]:
        """
        多尺度匹配

        在不同缩放比例下搜索模板，适应不同分辨率。
        """
        best_match = None
        best_confidence = 0.0

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmpl_h, tmpl_w = template.shape[:2]

        # 生成尺度序列
        scales = np.linspace(
            self.config.default_template_threshold,
            1.2,
            20
        )

        for scale in scales:
            # 缩放模板
            new_w = int(tmpl_w * scale)
            new_h = int(tmpl_h * scale)

            # 跳过无效尺寸
            if new_w < 10 or new_h < 10:
                continue
            if new_w > img_gray.shape[1] or new_h > img_gray.shape[0]:
                continue

            # 缩放模板
            resized = cv2.resize(template, (new_w, new_h))
            resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # 执行匹配
            result = cv2.matchTemplate(img_gray, resized_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_confidence:
                best_confidence = max_val
                if max_val >= threshold:
                    best_match = MatchResult(
                        name="",
                        position=(max_loc[0], max_loc[1], new_w, new_h),
                        confidence=float(max_val),
                        method=MatchMethod.TEMPLATE
                    )

        return best_match

    def match_all(
        self,
        image: np.ndarray,
        template: np.ndarray,
        threshold: Optional[float] = None,
        max_results: int = 10
    ) -> List[MatchResult]:
        """
        查找所有匹配位置

        Args:
            image: 搜索图像
            template: 模板图像
            threshold: 匹配阈值
            max_results: 最大结果数

        Returns:
            匹配结果列表
        """
        if threshold is None:
            threshold = self.config.default_template_threshold

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 模板不能比图像大
        if tmpl_gray.shape[0] > img_gray.shape[0] or tmpl_gray.shape[1] > img_gray.shape[1]:
            return []

        # 执行匹配
        result = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)

        # 获取所有超过阈值的点
        locations = np.where(result >= threshold)
        scores = result[locations]

        # 按分数排序
        indices = np.argsort(scores)[::-1][:max_results]

        matches = []
        h, w = tmpl_gray.shape[:2]

        for idx in indices:
            y, x = locations[0][idx], locations[1][idx]
            matches.append(MatchResult(
                name="",
                position=(int(x), int(y), w, h),
                confidence=float(scores[idx]),
                method=MatchMethod.TEMPLATE
            ))

        # 应用NMS去除重叠
        matches = self._apply_nms(matches, threshold=0.5)

        return matches

    def _apply_nms(
        self,
        matches: List[MatchResult],
        threshold: float = 0.5
    ) -> List[MatchResult]:
        """
        应用非极大值抑制

        Args:
            matches: 匹配结果列表
            threshold: IoU阈值

        Returns:
            过滤后的结果
        """
        if not matches:
            return []

        # 按置信度排序
        matches = sorted(matches, key=lambda x: x.confidence, reverse=True)

        keep = []
        while matches:
            current = matches.pop(0)
            keep.append(current)

            # 移除与当前框重叠过多的框
            matches = [
                m for m in matches
                if self._calculate_iou(current.position, m.position) < threshold
            ]

        return keep

    @staticmethod
    def _calculate_iou(
        box1: Tuple[int, int, int, int],
        box2: Tuple[int, int, int, int]
    ) -> float:
        """计算IoU（交并比）"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # 计算交集
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        intersection = (xi2 - xi1) * (yi2 - yi1)

        # 计算并集
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


# =============================================================================
# OCR文字识别器
# =============================================================================

class OCRMatcher:
    """
    OCR文字识别器

    基于EasyOCR的文字识别，用于定位界面上的文字元素。
    特别适用于主界面等有主题变化但文字固定的场景。

    特性：
    - 支持多语言文字识别
    - 支持指定搜索区域提高速度
    - 支持模糊匹配
    """

    def __init__(self, config: GUIMatcherConfig):
        self.config = config
        self._reader: Optional[easyocr.Reader] = None
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> bool:
        """初始化OCR引擎"""
        if self._initialized:
            return True

        if not EASYOCR_AVAILABLE:
            logger.error("EasyOCR未安装")
            return False

        try:
            import torch
            gpu_available = torch.cuda.is_available() and self.config.use_gpu

            logger.info(f"初始化OCR (GPU={gpu_available})...")
            self._reader = easyocr.Reader(
                ['ch_sim', 'en'],
                gpu=gpu_available,
                verbose=False
            )
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"OCR初始化失败: {e}")
            return False

    def shutdown(self):
        """关闭OCR引擎"""
        self._reader = None
        self._initialized = False

    def match(
        self,
        image: np.ndarray,
        target_texts: List[str],
        region: Optional[Tuple[float, float, float, float]] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[MatchResult]:
        """
        通过OCR识别文字位置

        Args:
            image: 搜索图像
            target_texts: 目标文字列表
            region: 搜索区域（相对坐标 x, y, w, h）
            confidence_threshold: 置信度阈值

        Returns:
            匹配结果列表
        """
        if not self._initialized or self._reader is None:
            raise RuntimeError("OCR引擎未初始化")

        if confidence_threshold is None:
            confidence_threshold = self.config.default_ocr_confidence

        # 提取搜索区域
        if region:
            h, w = image.shape[:2]
            x = int(region[0] * w)
            y = int(region[1] * h)
            rw = int(region[2] * w)
            rh = int(region[3] * h)
            search_image = image[y:y+rh, x:x+rw]
            offset_x, offset_y = x, y
        else:
            search_image = image
            offset_x, offset_y = 0, 0

        # 执行OCR
        with self._lock:
            results = self._reader.readtext(search_image, detail=1)

        matches = []

        for bbox, text, conf in results:
            if conf < confidence_threshold:
                continue

            # 检查是否匹配目标文字
            for target in target_texts:
                similarity = self._calculate_text_similarity(text, target)
                if similarity >= 0.7:  # 文字相似度阈值
                    # 计算边界框
                    points = np.array(bbox, dtype=np.int32)
                    x_min, y_min = points.min(axis=0)
                    x_max, y_max = points.max(axis=0)

                    position = (
                        int(x_min + offset_x),
                        int(y_min + offset_y),
                        int(x_max - x_min),
                        int(y_max - y_min)
                    )

                    matches.append(MatchResult(
                        name=target,
                        position=position,
                        confidence=float(conf) * similarity,
                        method=MatchMethod.OCR,
                        matched_text=text
                    ))

        return matches

    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """
        计算文字相似度

        使用简单的包含关系和编辑距离结合的方式。
        """
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        # 精确匹配
        if t1 == t2:
            return 1.0

        # 包含匹配
        if t1 in t2 or t2 in t1:
            return 0.9

        # 计算编辑距离
        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


# =============================================================================
# 主GUI匹配器
# =============================================================================

class GUIMatcher:
    """
    GUI界面匹配器（主类）

    整合模板匹配和OCR文字识别，提供统一的界面元素定位接口。

    使用示例：
        >>> matcher = GUIMatcher()
        >>> matcher.initialize()
        >>>
        >>> # 模板匹配
        >>> result = matcher.match_template(image, "home_btn.png")
        >>>
        >>> # OCR文字识别（主界面按钮）
        >>> results = matcher.match_text(image, ["编队", "干员", "任务"])
        >>>
        >>> # 点击位置
        >>> if result:
        ...     click(result.center)
    """

    # 主界面按钮文字定义（基于截图分析）
    MAIN_MENU_TEXTS = {
        'squad': ['编队'],
        'operator': ['干员', '角色管理'],
        'recruit': ['招募', '公开招募'],
        'mission': ['任务'],
        'base': ['基建'],
        'store': ['采购中心'],
        'friend': ['好友'],
        'archive': ['档案'],
        'warehouse': ['仓库'],
    }

    def __init__(self, config: Optional[GUIMatcherConfig] = None):
        self.config = config or GUIMatcherConfig()
        self.template_matcher = TemplateMatcher(self.config)
        self.ocr_matcher = OCRMatcher(self.config)
        self._initialized = False

    def initialize(self) -> bool:
        """初始化匹配器"""
        if self._initialized:
            return True

        # 初始化OCR
        if not self.ocr_matcher.initialize():
            logger.warning("OCR初始化失败，将仅使用模板匹配")

        self._initialized = True
        return True

    def shutdown(self):
        """关闭匹配器"""
        self.ocr_matcher.shutdown()
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False

    def match_template(
        self,
        image: np.ndarray,
        template_name: str,
        threshold: Optional[float] = None
    ) -> Optional[MatchResult]:
        """
        模板匹配

        Args:
            image: 搜索图像
            template_name: 模板文件名
            threshold: 匹配阈值

        Returns:
            匹配结果
        """
        template_path = self.config.templates_dir / template_name
        template = self.template_matcher.load_template(template_path)

        if template is None:
            return None

        result = self.template_matcher.match(image, template, threshold)
        if result:
            result.name = template_name
            result.template_path = str(template_path)

        return result

    def match_templates(
        self,
        image: np.ndarray,
        template_names: List[str],
        threshold: Optional[float] = None
    ) -> Dict[str, Optional[MatchResult]]:
        """
        批量模板匹配

        Args:
            image: 搜索图像
            template_names: 模板文件名列表
            threshold: 匹配阈值

        Returns:
            模板名到结果的映射
        """
        results = {}
        for name in template_names:
            results[name] = self.match_template(image, name, threshold)
        return results

    def match_text(
        self,
        image: np.ndarray,
        texts: Union[str, List[str]],
        region: Optional[Tuple[float, float, float, float]] = None
    ) -> List[MatchResult]:
        """
        OCR文字匹配

        Args:
            image: 搜索图像
            texts: 目标文字或文字列表
            region: 搜索区域（相对坐标）

        Returns:
            匹配结果列表
        """
        if isinstance(texts, str):
            texts = [texts]

        return self.ocr_matcher.match(image, texts, region)

    def find_main_menu_buttons(
        self,
        image: np.ndarray,
        buttons: Optional[List[str]] = None
    ) -> Dict[str, Optional[MatchResult]]:
        """
        查找主界面按钮

        使用OCR识别主界面上的按钮文字，适用于不同主题。

        Args:
            image: 游戏截图
            buttons: 要查找的按钮列表，默认查找所有

        Returns:
            按钮名到结果的映射
        """
        if buttons is None:
            buttons = list(self.MAIN_MENU_TEXTS.keys())

        # 收集所有目标文字
        all_texts = []
        text_to_button = {}
        for btn in buttons:
            if btn in self.MAIN_MENU_TEXTS:
                for text in self.MAIN_MENU_TEXTS[btn]:
                    all_texts.append(text)
                    text_to_button[text] = btn

        # 执行OCR匹配
        matches = self.match_text(image, all_texts)

        # 整理结果
        results = {btn: None for btn in buttons}
        for match in matches:
            if match.matched_text in text_to_button:
                btn_name = text_to_button[match.matched_text]
                # 保留置信度最高的结果
                if results[btn_name] is None or match.confidence > results[btn_name].confidence:
                    results[btn_name] = match

        return results

    def visualize_matches(
        self,
        image: np.ndarray,
        matches: List[MatchResult],
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        可视化匹配结果

        Args:
            image: 原始图像
            matches: 匹配结果列表
            save_path: 保存路径

        Returns:
            可视化后的图像
        """
        vis = image.copy()

        colors = {
            MatchMethod.TEMPLATE: (0, 255, 0),  # 绿色
            MatchMethod.OCR: (255, 0, 0),       # 蓝色
            MatchMethod.HYBRID: (0, 255, 255),  # 黄色
        }

        for i, match in enumerate(matches):
            if match is None:
                continue

            x, y, w, h = match.position
            color = colors.get(match.method, (128, 128, 128))

            # 绘制矩形
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # 绘制文字标签
            label = f"{match.name}: {match.confidence:.2f}"
            if match.matched_text:
                label += f" ({match.matched_text})"

            # 文字背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis, (x, y - text_h - 5), (x + text_w, y), color, -1)
            cv2.putText(vis, label, (x, y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 绘制中心点
            cx, cy = match.center
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)

        if save_path:
            cv2.imwrite(str(save_path), vis)

        return vis


# =============================================================================
# 便捷函数
# =============================================================================

def find_template(
    image: np.ndarray,
    template_path: Union[str, Path],
    threshold: float = 0.8
) -> Optional[MatchResult]:
    """
    便捷函数：查找单个模板

    Args:
        image: 搜索图像
        template_path: 模板路径
        threshold: 匹配阈值

    Returns:
        匹配结果
    """
    config = GUIMatcherConfig()
    matcher = TemplateMatcher(config)

    template = matcher.load_template(Path(template_path))
    if template is None:
        return None

    return matcher.match(image, template, threshold)


def find_main_buttons(image: np.ndarray) -> Dict[str, Optional[MatchResult]]:
    """
    便捷函数：查找主界面按钮

    Args:
        image: 游戏截图

    Returns:
        按钮位置字典
    """
    with GUIMatcher() as matcher:
        return matcher.find_main_menu_buttons(image)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("GUI Matcher Module")
    print("=" * 50)

    # 测试配置
    config = GUIMatcherConfig()
    print(f"模板目录: {config.templates_dir}")
    print(f"GPU加速: {config.use_gpu}")

    # 测试模板匹配器
    print("\n模板匹配器测试:")
    template_matcher = TemplateMatcher(config)

    # 创建测试图像
    test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    test_template = np.ones((50, 100, 3), dtype=np.uint8) * 255

    # 放置模板到图像
    test_image[100:150, 200:300] = test_template

    result = template_matcher.match(test_image, test_template, threshold=0.9)
    if result:
        print(f"  匹配成功: 位置={result.position}, 置信度={result.confidence:.3f}")
    else:
        print("  未找到匹配")

    # 测试OCR匹配器
    print("\nOCR匹配器测试:")
    if EASYOCR_AVAILABLE:
        ocr_matcher = OCRMatcher(config)
        if ocr_matcher.initialize():
            print("  OCR初始化成功")
            ocr_matcher.shutdown()
        else:
            print("  OCR初始化失败")
    else:
        print("  EasyOCR未安装")

    print("\n模块加载完成!")
