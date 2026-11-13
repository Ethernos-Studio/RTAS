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

从编队截图中识别干员卡片信息：
- 精英化等级（通过图标匹配）
- 干员等级（数字识别）
- 干员名称（OCR文字识别）

Author: RTAS Vision Team
Version: 1.0.0
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

from . import ImagePreprocessor

# torch / easyocr 延迟加载：这里只探测是否安装，真正导入推迟到使用时
from .ocr_runtime import (
    TORCH_AVAILABLE,
    create_reader,
    cuda_available,
)

logger = logging.getLogger(__name__)


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
    """编队识别配置"""
    # 卡片布局配置 (基于1920x1080分辨率)
    first_card_x: int = 130
    first_card_y: int = 100
    card_width: int = 200
    card_height: int = 460  # 增加高度以包含完整名称区域
    card_spacing_x: int = 240
    card_spacing_y: int = 440
    cards_per_row: int = 4
    max_rows: int = 2

    # 参考分辨率
    reference_width: int = 1920
    reference_height: int = 1080

    # 精英化图标配置 (基于实际图片坐标 120,315 ~ 195,370)
    # 相对于卡片位置: (3, 253, 82, 61) @ 1920x1080
    elite_icon_size: Tuple[int, int] = (82, 61)  # 宽, 高
    elite_icon_position: Tuple[int, int] = (3, 253)  # 相对于卡片左上角的偏移

    # 等级数字配置 (135,410~210,480 相对于原始图像 -> 相对于卡片: 5,310,75,70)
    # 往下移动10像素: y=310+10=320
    level_region: Tuple[int, int, int, int] = (5, 320, 75, 70)  # x, y, w, h 相对于卡片

    # 名称区域配置 (从等级区域底部到卡片底部)
    # 等级底部 = 320 + 70 = 390, 卡片高度 = 420, 所以名称区域: y=390
    # 高度扩展30像素: h=40+30=70
    name_region: Tuple[int, int, int, int] = (0, 390, 200, 70)  # x, y, w, h 相对于卡片

    # 精英化图标模板路径 (使用从实际截图提取的模板)
    elite_icon_templates: Dict[int, Path] = field(default_factory=lambda: {
        1: Path("templates/icons/elitism/level_1_extracted.png"),
        2: Path("templates/icons/elitism/level_2_proxy.png"),  # 使用从代理界面提取的模板
    })

    # OCR配置
    ocr_languages: List[str] = field(default_factory=lambda: ['ch_sim', 'en'])
    ocr_gpu: Union[bool, str] = 'auto'  # True=强制使用GPU, False=强制使用CPU, 'auto'=自动检测

    # 匹配阈值
    elite_match_threshold: float = 0.5  # 降低阈值以适应更多情况
    name_confidence_threshold: float = 0.3  # 降低名称识别阈值

    @classmethod
    def preset_squad_selection(cls) -> "SquadConfig":
        """
        预设配置：编队选择界面（代理作战）
        适用于底部有"本次行动配置不可更改"的界面
        使用6列布局（与编队编辑界面相同）
        """
        # 使用与edit布局相同的配置
        return cls.preset_squad_edit()

    @classmethod
    def preset_squad_edit(cls) -> "SquadConfig":
        """
        预设配置：编队编辑界面
        适用于底部显示编队名称的界面
        """
        config = cls()
        # 卡片布局配置
        config.first_card_x = 160
        config.first_card_y = 120
        config.card_width = 190
        config.card_height = 410
        config.card_spacing_x = 233
        config.card_spacing_y = 435
        config.cards_per_row = 6  # 新布局每行6个卡片
        config.max_rows = 2

        # 精英化图标配置 (160, 350~ 245, 420 相对于原始图像)
        # 相对于卡片: (0, 230, 85, 70)
        config.elite_icon_position = (0, 230)
        config.elite_icon_size = (85, 70)

        # 等级数字配置 (160, 420 ~ 245 490 相对于原始图像)
        # 相对于卡片: (0, 300, 85, 70)
        config.level_region = (0, 300, 85, 70)

        # 名称区域配置 (160, 490 ~ 350, 530 相对于原始图像)
        # 相对于卡片: (0, 370, 190, 40)
        config.name_region = (0, 370, 190, 40)

        return config


class SquadRecognizer:
    """
    编队识别器

    从编队截图中识别所有干员卡片信息
    """

    def __init__(self, config: Optional[SquadConfig] = None):
        """
        初始化编队识别器

        Args:
            config: 识别配置，使用默认配置如果为None
        """
        self.config = config or SquadConfig()
        self._elite_templates: Dict[int, np.ndarray] = {}
        self._ocr_reader = None
        self._initialized = False

    def _check_gpu_available(self) -> bool:
        """
        检测GPU是否可用

        Returns:
            GPU是否可用
        """
        if not TORCH_AVAILABLE:
            return False
        try:
            return cuda_available()
        except Exception:
            return False

    def initialize(self) -> bool:
        """
        初始化识别器

        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化编队识别器...")

            # 加载精英化图标模板
            self._load_elite_templates()

            # 确定是否使用GPU
            use_gpu = False
            if self.config.ocr_gpu == 'auto':
                use_gpu = self._check_gpu_available()
                if use_gpu:
                    logger.info("自动检测到GPU，启用GPU加速")
                else:
                    logger.info("未检测到GPU，使用CPU")
            elif self.config.ocr_gpu is True:
                use_gpu = self._check_gpu_available()
                if not use_gpu:
                    logger.warning("强制使用GPU但GPU不可用，将使用CPU")
            else:
                use_gpu = False
                logger.info("使用CPU模式")

            # 初始化OCR（此处才会真正导入 easyocr）
            try:
                self._ocr_reader = create_reader(
                    self.config.ocr_languages,
                    gpu=use_gpu
                )
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

    def _load_elite_templates(self) -> None:
        """加载精英化图标模板"""
        for level, template_path in self.config.elite_icon_templates.items():
            if template_path.exists():
                template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    # 调整模板大小
                    template = cv2.resize(template, self.config.elite_icon_size)
                    self._elite_templates[level] = template
                    logger.debug(f"加载精英化{level}图标模板: {template_path}")
                else:
                    logger.warning(f"无法加载模板: {template_path}")
            else:
                logger.warning(f"模板文件不存在: {template_path}")

    def _get_scaled_config(self, image: np.ndarray) -> Dict[str, float]:
        """
        根据图像尺寸获取缩放后的配置

        Args:
            image: 输入图像

        Returns:
            缩放后的配置参数
        """
        h, w = image.shape[:2]
        scale_x = w / self.config.reference_width
        scale_y = h / self.config.reference_height

        return {
            'first_card_x': int(self.config.first_card_x * scale_x),
            'first_card_y': int(self.config.first_card_y * scale_y),
            'card_width': int(self.config.card_width * scale_x),
            'card_height': int(self.config.card_height * scale_y),
            'card_spacing_x': int(self.config.card_spacing_x * scale_x),
            'card_spacing_y': int(self.config.card_spacing_y * scale_y),
            'scale_x': scale_x,
            'scale_y': scale_y,
        }

    def recognize_squad(self, image: np.ndarray) -> List[OperatorCard]:
        """
        识别编队中的所有干员卡片

        Args:
            image: 编队截图

        Returns:
            干员卡片列表
        """
        if not self._initialized:
            raise RuntimeError("识别器未初始化")

        # 获取缩放后的配置
        scaled = self._get_scaled_config(image)
        logger.info(f"图像尺寸: {image.shape[1]}x{image.shape[0]}, 缩放比例: {scaled['scale_x']:.3f}x{scaled['scale_y']:.3f}")

        cards = []

        # 遍历所有可能的卡片位置
        for row in range(self.config.max_rows):
            for col in range(self.config.cards_per_row):
                # 计算卡片位置（使用缩放后的配置）
                card_x = scaled['first_card_x'] + col * scaled['card_spacing_x']
                card_y = scaled['first_card_y'] + row * scaled['card_spacing_y']
                card_w = scaled['card_width']
                card_h = scaled['card_height']

                # 检查是否在图像范围内
                h, w = image.shape[:2]
                if card_x + card_w > w or card_y + card_h > h:
                    logger.debug(f"位置({row},{col})超出图像范围")
                    continue

                # 提取卡片区域
                card_image = image[card_y:card_y+card_h, card_x:card_x+card_w]

                # 检查是否是有效的干员卡片（通过检测是否有内容）
                if not self._is_valid_card(card_image):
                    logger.debug(f"位置({row},{col})不是有效卡片")
                    continue

                # 识别卡片
                card = self._recognize_card(
                    card_image,
                    (card_x, card_y, card_w, card_h),
                    (row, col),
                    scaled
                )

                if card:
                    cards.append(card)
                    logger.info(f"识别到干员: {card}")

        logger.info(f"共识别到 {len(cards)} 个干员")
        return cards

    def _is_valid_card(self, card_image: np.ndarray) -> bool:
        """
        检查是否是有效的干员卡片

        通过检测卡片区域是否有干员头像（上半部分有内容）

        Args:
            card_image: 卡片图像

        Returns:
            是否是有效卡片
        """
        # 转换为灰度图
        if len(card_image.shape) == 3:
            gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = card_image

        h, w = gray.shape

        # 检查卡片上半部分（头像区域）是否有足够的变化
        # 空位卡片上半部分通常是纯色或渐变
        upper_region = gray[:int(h*0.6), :]

        # 计算上半部分的标准差
        std = np.std(upper_region)

        # 如果标准差太小，说明是空位
        # 同时检查是否有明显的边缘（干员头像有明确轮廓）
        edges = cv2.Canny(upper_region, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        return std > 20 and edge_ratio > 0.05

    def _recognize_card(
        self,
        card_image: np.ndarray,
        position: Tuple[int, int, int, int],
        grid_position: Tuple[int, int],
        scaled_config: Dict[str, float]
    ) -> Optional[OperatorCard]:
        """
        识别单个卡片

        Args:
            card_image: 卡片图像
            position: 卡片在原始图像中的位置 (x, y, w, h)
            grid_position: 卡片在网格中的位置 (row, col)
            scaled_config: 缩放后的配置

        Returns:
            识别到的干员卡片或None
        """
        card = OperatorCard(
            position=position,
            grid_position=grid_position
        )

        # 1. 识别精英化等级
        card.elite_level = self._recognize_elite_level(card_image, scaled_config)

        # 2. 识别等级
        card.level = self._recognize_level(card_image, scaled_config)

        # 3. 识别名称
        name_result = self._recognize_name(card_image, scaled_config)
        if name_result:
            card.name, card.name_confidence = name_result
        else:
            # 如果名称识别失败，可能不是有效卡片
            return None

        # 保存原始图像（用于调试）
        card.raw_elite_icon = self._extract_elite_region(card_image, scaled_config)
        card.raw_level_image = self._extract_level_region(card_image, scaled_config)
        card.raw_name_image = self._extract_name_region(card_image, scaled_config)

        return card

    def _extract_elite_region(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> np.ndarray:
        """提取精英化图标区域"""
        scale_x = scaled_config['scale_x']
        scale_y = scaled_config['scale_y']
        x = int(self.config.elite_icon_position[0] * scale_x)
        y = int(self.config.elite_icon_position[1] * scale_y)
        w = int(self.config.elite_icon_size[0] * scale_x)
        h = int(self.config.elite_icon_size[1] * scale_y)
        h_img, w_img = card_image.shape[:2]
        x = min(x, w_img - 1)
        y = min(y, h_img - 1)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return card_image[y:y+h, x:x+w]

    def _extract_level_region(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> np.ndarray:
        """提取等级数字区域"""
        scale_x = scaled_config['scale_x']
        scale_y = scaled_config['scale_y']
        x, y, w, h = self.config.level_region
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        h_img, w_img = card_image.shape[:2]
        x = min(x, w_img - 1)
        y = min(y, h_img - 1)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return card_image[y:y+h, x:x+w]

    def _extract_name_region(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> np.ndarray:
        """提取名称区域"""
        scale_x = scaled_config['scale_x']
        scale_y = scaled_config['scale_y']
        x, y, w, h = self.config.name_region
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
        h_img, w_img = card_image.shape[:2]
        x = min(x, w_img - 1)
        y = min(y, h_img - 1)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        return card_image[y:y+h, x:x+w]

    def _recognize_elite_level(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> EliteLevel:
        """
        识别精英化等级

        通过模板匹配精英化图标

        Args:
            card_image: 卡片图像
            scaled_config: 缩放配置

        Returns:
            精英化等级
        """
        if not self._elite_templates:
            return EliteLevel.E0

        # 提取精英化图标区域
        elite_region = self._extract_elite_region(card_image, scaled_config)

        # 转换为灰度图
        if len(elite_region.shape) == 3:
            elite_gray = cv2.cvtColor(elite_region, cv2.COLOR_BGR2GRAY)
        else:
            elite_gray = elite_region

        # 首先检查区域是否有明显的精英化图标特征
        # E0区域边缘比例低，E1/E2有明显的图标特征（边缘比例高）
        std = np.std(elite_gray)
        edge_ratio = self._calculate_edge_ratio(elite_gray)

        # 如果边缘比例太低，说明没有精英化图标，是E0
        # E1/E2的边缘比例通常在0.15以上，E0在0.08以下
        if edge_ratio < 0.08:
            logger.debug(f"精英化区域边缘比例太低，判断为E0 (edge={edge_ratio:.3f})")
            return EliteLevel.E0

        best_match_level = 0
        best_match_score = 0.0

        # 获取实际区域大小
        actual_h, actual_w = elite_gray.shape[:2]

        # 对每个模板进行匹配
        for level, template in self._elite_templates.items():
            # 调整模板大小以匹配实际区域
            template_resized = cv2.resize(template, (actual_w, actual_h))

            # 模板匹配
            result = cv2.matchTemplate(elite_gray, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            logger.debug(f"精英化{level}匹配分数: {max_val:.3f}")

            if max_val > best_match_score:
                best_match_score = max_val
                best_match_level = level

        # 检查是否超过阈值
        # 提高阈值到0.7以减少误判
        high_confidence_threshold = 0.7
        if best_match_score >= high_confidence_threshold:
            logger.debug(f"识别到精英化等级: E{best_match_level} (分数: {best_match_score:.3f})")
            return EliteLevel(best_match_level)
        elif best_match_score >= self.config.elite_match_threshold:
            # 中等置信度，需要额外验证
            # 检查E1和E2的分数差距
            # 如果两个分数接近，可能是误判
            logger.debug(f"中等置信度匹配: E{best_match_level} (分数: {best_match_score:.3f})")
            return EliteLevel(best_match_level)
        else:
            logger.debug(f"未识别到精英化图标 (最佳分数: {best_match_score:.3f})")
            return EliteLevel.E0

    def _calculate_edge_ratio(self, gray_image: np.ndarray) -> float:
        """计算图像边缘比例"""
        edges = cv2.Canny(gray_image, 50, 150)
        return np.count_nonzero(edges) / edges.size

    def _recognize_level(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> int:
        """
        识别干员等级

        使用OCR识别等级数字

        Args:
            card_image: 卡片图像
            scaled_config: 缩放配置

        Returns:
            等级数字
        """
        # 提取等级区域
        level_region = self._extract_level_region(card_image, scaled_config)

        # 多种预处理方法
        preprocessed_images = []

        # 方法1: 原始图像
        preprocessed_images.append(level_region)

        # 方法2: 灰度+二值化
        gray = cv2.cvtColor(level_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        preprocessed_images.append(binary_bgr)

        # 方法3: 放大2x
        enlarged = cv2.resize(level_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        preprocessed_images.append(enlarged)

        # 方法4: 放大+二值化
        gray_large = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        _, binary_large = cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_large_bgr = cv2.cvtColor(binary_large, cv2.COLOR_GRAY2BGR)
        preprocessed_images.append(binary_large_bgr)

        # OCR识别 - 尝试所有预处理方法
        best_level = 1
        best_confidence = 0.0

        try:
            for img in preprocessed_images:
                results = self._ocr_reader.readtext(img)

                for result in results:
                    text = result[1]
                    confidence = result[2]

                    # 提取数字
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        level = int(numbers[0])
                        if 1 <= level <= 90:  # 合理等级范围
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_level = level
                                logger.debug(f"识别到等级: {level} (置信度: {confidence:.3f})")

        except Exception as e:
            logger.warning(f"等级识别失败: {e}")

        return best_level

    def _recognize_name(self, card_image: np.ndarray, scaled_config: Dict[str, float]) -> Optional[Tuple[str, float]]:
        """
        识别干员名称

        使用OCR识别名称文字

        Args:
            card_image: 卡片图像
            scaled_config: 缩放配置

        Returns:
            (名称, 置信度) 或 None
        """
        # 提取名称区域
        name_region = self._extract_name_region(card_image, scaled_config)

        # 多种预处理方法，选择最佳结果
        best_name = None
        best_confidence = 0.0

        # 方法1: CLAHE增强
        lab = cv2.cvtColor(name_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # 方法2: 灰度+二值化
        gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 方法3: 反转+二值化（白底黑字）
        inverted = cv2.bitwise_not(gray)
        _, binary_inv = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 方法4: 高斯模糊后增强
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, binary_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 尝试所有方法
        for idx, processed in enumerate([enhanced, binary, binary_inv, binary_sharp]):
            try:
                results = self._ocr_reader.readtext(processed)
                for result in results:
                    text = result[1]
                    confidence = result[2]
                    text = self._clean_name(text)
                    if text and confidence > best_confidence:
                        best_confidence = confidence
                        best_name = text
            except Exception:
                continue

        if best_name and best_confidence >= self.config.name_confidence_threshold:
            logger.debug(f"识别到名称: '{best_name}' (置信度: {best_confidence:.3f})")
            return best_name, best_confidence

        return None

    def _clean_name(self, name: str) -> str:
        """
        清理干员名称

        移除不必要的字符

        Args:
            name: 原始名称

        Returns:
            清理后的名称
        """
        # 移除常见干扰字符
        name = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', name)
        return name.strip()

    def visualize_result(
        self,
        image: np.ndarray,
        cards: List[OperatorCard],
        output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        可视化识别结果

        Args:
            image: 原始图像
            cards: 识别到的干员卡片列表
            output_path: 输出路径（可选）

        Returns:
            可视化图像
        """
        vis_image = image.copy()

        # 转换为PIL图像以支持中文
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(vis_image_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 尝试加载中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()

        for card in cards:
            x, y, w, h = card.position

            # 绘制卡片边框 (使用OpenCV)
            color = (0, 255, 0) if card.name_confidence > 0.8 else (0, 165, 255)
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)

            # 构建显示信息
            elite_str = f"E{card.elite_level.value}"
            lines = [f"{card.name}", f"[{elite_str} Lv.{card.level}]"]

            # 如果有干员信息，添加星级、职业、费用
            if card.operator_info:
                stars = card.operator_info.get('rarity', 0)
                profession = card.operator_info.get('profession', '')
                cost = card.operator_info.get('cost', 0)
                position = card.operator_info.get('position', '')

                # 星级显示为★符号
                star_str = '★' * stars if stars > 0 else ''
                lines.append(f"{star_str}")
                lines.append(f"{profession} | {position}")
                lines.append(f"费用: {cost}")

            # 计算文字区域尺寸
            line_height = 22
            padding = 5
            max_width = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                max_width = max(max_width, text_width)

            total_height = len(lines) * line_height + padding * 2

            # 绘制文字背景
            draw.rectangle(
                [(x, y - total_height), (x + max_width, y)],
                fill=(color[2], color[1], color[0])  # RGB格式
            )

            # 绘制多行文字
            for i, line in enumerate(lines):
                draw.text(
                    (x, y - total_height + padding + i * line_height),
                    line,
                    font=font,
                    fill=(255, 255, 255)
                )

        # 转换回OpenCV格式
        vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 保存
        if output_path:
            cv2.imwrite(str(output_path), vis_image)
            logger.info(f"可视化结果已保存: {output_path}")

        return vis_image

    def shutdown(self) -> None:
        """关闭识别器"""
        self._ocr_reader = None
        self._elite_templates.clear()
        self._initialized = False
        logger.info("编队识别器已关闭")
