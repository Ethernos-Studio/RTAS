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
增强版GUI界面匹配模块

改进功能：
1. 图像预处理提高OCR识别率（针对PRTS立体效果）
2. 修复可视化中文显示问题
3. 支持终端识别
4. 支持活动识别
5. 支持理智值识别

Author: RTAS Vision Team
Version: 2.0.0
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import time
import threading
import re

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

class UIElementType(Enum):
    """UI元素类型"""
    BUTTON = auto()
    TEXT = auto()
    ICON = auto()
    TERMINAL = auto()
    ACTIVITY = auto()
    RESOURCE = auto()


@dataclass
class UIElement:
    """UI元素"""
    name: str
    element_type: UIElementType
    position: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    text: Optional[str] = None
    value: Optional[Any] = None  # 用于存储数值（如理智值）

    @property
    def center(self) -> Tuple[int, int]:
        """中心点坐标"""
        x, y, w, h = self.position
        return (x + w // 2, y + h // 2)

    @property
    def top_left(self) -> Tuple[int, int]:
        """左上角"""
        return (self.position[0], self.position[1])

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """右下角"""
        x, y, w, h = self.position
        return (x + w, y + h)


@dataclass
class SanityInfo:
    """理智信息"""
    current: int
    max: int
    position: Optional[Tuple[int, int]] = None

    @property
    def percentage(self) -> float:
        """理智百分比"""
        return self.current / self.max * 100 if self.max > 0 else 0

    def __str__(self) -> str:
        return f"{self.current}/{self.max} ({self.percentage:.1f}%)"


@dataclass
class TerminalInfo:
    """终端信息"""
    name: str
    position: Tuple[int, int]
    is_active: bool = True


@dataclass
class ActivityInfo:
    """活动信息"""
    name: str
    position: Tuple[int, int]
    is_new: bool = False


@dataclass
class ResourceInfo:
    """资源信息（龙门币、合成玉、源石）"""
    name: str
    amount: int
    position: Optional[Tuple[int, int]] = None

    def __str__(self) -> str:
        return f"{self.name}: {self.amount:,}"


@dataclass
class MainMenuResult:
    """主界面识别结果"""
    buttons: Dict[str, Optional[UIElement]] = field(default_factory=dict)
    terminal: Optional[TerminalInfo] = None
    activities: List[ActivityInfo] = field(default_factory=list)
    sanity: Optional[SanityInfo] = None
    resources: Dict[str, ResourceInfo] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'buttons': {
                name: {
                    'found': elem is not None,
                    'position': elem.center if elem else None,
                    'confidence': elem.confidence if elem else 0
                }
                for name, elem in self.buttons.items()
            },
            'terminal': {
                'name': self.terminal.name,
                'position': self.terminal.position
            } if self.terminal else None,
            'activities': [
                {'name': act.name, 'position': act.position}
                for act in self.activities
            ],
            'sanity': {
                'current': self.sanity.current,
                'max': self.sanity.max,
                'percentage': self.sanity.percentage
            } if self.sanity else None,
            'resources': {
                name: {
                    'amount': res.amount,
                    'position': res.position
                }
                for name, res in self.resources.items()
            }
        }


# =============================================================================
# 图像预处理
# =============================================================================

class ImagePreprocessor:
    """图像预处理器 - 针对PRTS立体效果优化"""

    @staticmethod
    def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        增强图像以提高OCR识别率

        针对明日方舟PRTS立体效果的处理：
        1. 降噪
        2. 对比度增强
        3. 边缘增强
        4. 二值化
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. 降噪
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 2. CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 3. 锐化
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        # 4. 自适应二值化
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        # 转回BGR格式
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        return result

    @staticmethod
    def extract_text_region(
        image: np.ndarray,
        region: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """提取指定区域的图像"""
        h, w = image.shape[:2]
        x = int(region[0] * w)
        y = int(region[1] * h)
        rw = int(region[2] * w)
        rh = int(region[3] * h)
        return image[y:y+rh, x:x+rw]


# =============================================================================
# 增强版OCR匹配器
# =============================================================================

class EnhancedOCRMatcher:
    """增强版OCR匹配器"""

    # 主界面按钮定义（基于实际测试优化的区域，避免重叠）
    # 区域格式: (x, y, w, h) 相对于图像的归一化坐标
    MAIN_MENU_DEFINITIONS = {
        'squad': {
            'texts': ['编队', '编', '队'],
            'region': (1000/1920, 430/1080, 340/1920, 140/1080),  # x=1000, y=430, w=340, h=140
            'color_hint': 'white_on_dark'
        },
        'operator': {
            'texts': ['干员', '角色管理', '干'],
            'region': (0.65, 0.40, 0.20, 0.20),
            'color_hint': 'white_on_dark'
        },
        'recruit': {
            'texts': ['招募', '公开招募'],
            'region': (1400/1920, 720/1080, 230/1920, 130/1080),  # x=1400, y=720, w=230, h=130
            'color_hint': 'white_on_blue'
        },
        'headhunt': {
            'texts': ['干员寻访'],
            'region': (1630/1920, 720/1080, 270/1920, 130/1080),  # x=1630, y=720, w=270, h=130
            'color_hint': 'white_on_blue'
        },
        'mission': {
            'texts': ['任务', '务'],
            'region': (0.50, 0.75, 0.20, 0.20),
            'color_hint': 'white_on_dark'
        },
        'base': {
            'texts': ['基建', '基'],
            'region': (0.65, 0.80, 0.20, 0.15),
            'color_hint': 'white_on_dark'
        },
        'store': {
            'texts': ['采购中心', '采购'],
            'region': (0.55, 0.62, 0.20, 0.15),
            'color_hint': 'white_on_blue'
        },
        'friend': {
            'texts': ['好友', '友', '好'],
            'region': (0.22, 0.72, 0.15, 0.15),
            'color_hint': 'white_on_dark'
        },
        'archive': {
            'texts': ['档案', '档'],
            'region': (0.20, 0.75, 0.20, 0.20),
            'color_hint': 'white_on_dark'
        },
        'warehouse': {
            'texts': ['仓库', '仓', '3天'],
            'region': (0.78, 0.82, 0.18, 0.12),
            'color_hint': 'white_on_dark'
        },
    }

    # 终端区域定义（用户提供的精确坐标: 1340,160 ~ 1580,300）
    TERMINAL_REGION = (1340/1920, 160/1080, 240/1920, 140/1080)  # x=1340, y=160, w=240, h=140

    # 活动区域定义（终端右侧）
    ACTIVITY_REGION = (0.78, 0.12, 0.20, 0.20)

    # 理智区域定义（用户提供的精确坐标）
    # 当前理智显示区域: 1080,200 ~ 1340,340 (显示大数字，如"168")
    # 理智/上限显示区域: 1080,340 ~ 1340,410 (显示"理智/168")
    SANITY_CURRENT_DISPLAY_REGION = (1080/1920, 200/1080, 260/1920, 140/1080)  # x=1080, y=200, w=260, h=140
    SANITY_MAX_DISPLAY_REGION = (1080/1920, 340/1080, 260/1920, 70/1080)       # x=1080, y=340, w=260, h=70

    # 资源区域定义（用户提供的精确坐标）
    # 龙门币: 1080,90 ~ 1250,160
    LMD_REGION = (1080/1920, 90/1080, 170/1920, 70/1080)      # x=1080, y=90, w=170, h=70
    # 合成玉: 1340,60 ~ 1510,120
    ORIGINITE_REGION = (1340/1920, 60/1080, 170/1920, 60/1080)  # x=1340, y=60, w=170, h=60
    # 源石: 1630,40 ~ 1780,100
    ORUNDUM_REGION = (1630/1920, 40/1080, 150/1920, 60/1080)   # x=1630, y=40, w=150, h=60

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._reader: Optional[easyocr.Reader] = None
        self._lock = threading.RLock()
        self._initialized = False
        self._preprocessor = ImagePreprocessor()

    def initialize(self) -> bool:
        """初始化OCR引擎"""
        if self._initialized:
            return True

        if not EASYOCR_AVAILABLE:
            logger.error("EasyOCR未安装")
            return False

        try:
            import torch
            gpu_available = torch.cuda.is_available() and self.use_gpu

            logger.info(f"初始化增强版OCR (GPU={gpu_available})...")
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

    def recognize_all_text(
        self,
        image: np.ndarray,
        use_preprocessing: bool = True
    ) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        识别图像中的所有文字

        Returns:
            [(text, (x, y, w, h), confidence), ...]
        """
        if not self._initialized or self._reader is None:
            raise RuntimeError("OCR引擎未初始化")

        # 预处理
        if use_preprocessing:
            processed = self._preprocessor.enhance_for_ocr(image)
        else:
            processed = image

        # 执行OCR
        with self._lock:
            results = self._reader.readtext(processed, detail=1)

        # 整理结果
        recognized = []
        for bbox, text, conf in results:
            points = np.array(bbox, dtype=np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            position = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
            recognized.append((text, position, float(conf)))

        return recognized

    def find_main_menu_buttons(
        self,
        image: np.ndarray
    ) -> Dict[str, Optional[UIElement]]:
        """
        查找主界面所有按钮

        使用区域搜索 + 图像预处理提高识别率
        """
        results = {}
        h, w = image.shape[:2]

        for button_id, definition in self.MAIN_MENU_DEFINITIONS.items():
            # 提取搜索区域
            region = definition['region']
            x = int(region[0] * w)
            y = int(region[1] * h)
            rw = int(region[2] * w)
            rh = int(region[3] * h)

            # 确保区域在图像范围内
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = min(rw, w - x)
            rh = min(rh, h - y)

            if rw <= 0 or rh <= 0:
                results[button_id] = None
                continue

            region_image = image[y:y+rh, x:x+rw]

            # 尝试原始图像和预处理图像
            best_match = None
            best_confidence = 0.0

            for use_preprocessing in [False, True]:
                if use_preprocessing:
                    processed = self._preprocessor.enhance_for_ocr(region_image)
                else:
                    processed = region_image

                # 识别文字
                with self._lock:
                    ocr_results = self._reader.readtext(processed, detail=1)

                # 查找匹配
                for bbox, text, conf in ocr_results:
                    if conf < 0.4:  # 降低阈值以提高识别率
                        continue

                    # 检查是否匹配目标文字
                    for target_text in definition['texts']:
                        similarity = self._calculate_text_similarity(text, target_text)
                        combined_confidence = conf * similarity

                        if similarity >= 0.5 and combined_confidence > best_confidence:
                            # 计算绝对位置
                            points = np.array(bbox, dtype=np.int32)
                            x_min, y_min = points.min(axis=0)
                            x_max, y_max = points.max(axis=0)

                            abs_x = x + int(x_min)
                            abs_y = y + int(y_min)
                            abs_w = int(x_max - x_min)
                            abs_h = int(y_max - y_min)

                            best_match = UIElement(
                                name=button_id,
                                element_type=UIElementType.BUTTON,
                                position=(abs_x, abs_y, abs_w, abs_h),
                                confidence=combined_confidence,
                                text=text
                            )
                            best_confidence = combined_confidence

            results[button_id] = best_match

        return results

    def recognize_terminal(self, image: np.ndarray) -> Optional[TerminalInfo]:
        """识别终端

        明日方舟主界面右上角的"终端"大文字
        """
        h, w = image.shape[:2]
        region = self.TERMINAL_REGION

        x = int(region[0] * w)
        y = int(region[1] * h)
        rw = int(region[2] * w)
        rh = int(region[3] * h)

        region_image = image[y:y+rh, x:x+rw]

        # 尝试原始图像和预处理图像
        best_match = None
        best_confidence = 0.0

        for use_preprocessing in [False, True]:
            if use_preprocessing:
                processed = self._preprocessor.enhance_for_ocr(region_image)
            else:
                processed = region_image

            with self._lock:
                results = self._reader.readtext(processed, detail=1)

            # 优先查找包含"终端"的文字
            for bbox, text, conf in results:
                # 清理OCR结果中的噪声字符
                cleaned_text = text.strip().replace(']', '').replace('[', '')

                if '终端' in cleaned_text:
                    points = np.array(bbox, dtype=np.int32)
                    center_x = x + int(np.mean([p[0] for p in points]))
                    center_y = y + int(np.mean([p[1] for p in points]))

                    if conf > best_confidence:
                        best_match = TerminalInfo(
                            name='终端',
                            position=(center_x, center_y)
                        )
                        best_confidence = conf

        return best_match

    def recognize_activities(self, image: np.ndarray) -> List[ActivityInfo]:
        """识别活动"""
        h, w = image.shape[:2]
        region = self.ACTIVITY_REGION

        x = int(region[0] * w)
        y = int(region[1] * h)
        rw = int(region[2] * w)
        rh = int(region[3] * h)

        region_image = image[y:y+rh, x:x+rw]
        processed = self._preprocessor.enhance_for_ocr(region_image)

        with self._lock:
            results = self._reader.readtext(processed, detail=1)

        activities = []
        for bbox, text, conf in results:
            if conf > 0.6 and len(text) > 1:
                points = np.array(bbox, dtype=np.int32)
                center_x = x + int(np.mean([p[0] for p in points]))
                center_y = y + int(np.mean([p[1] for p in points]))
                activities.append(ActivityInfo(
                    name=text,
                    position=(center_x, center_y)
                ))

        return activities

    def recognize_sanity(
        self,
        image: np.ndarray,
        save_debug_images: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Optional[SanityInfo]:
        """识别理智值

        识别主界面显示的"理智/上限"格式，如"135/168"
        使用用户提供的精确坐标区域

        Args:
            image: 输入图像
            save_debug_images: 是否保存调试图像
            debug_dir: 调试图像保存目录

        Returns:
            SanityInfo对象或None
        """
        h, w = image.shape[:2]

        # 分别识别两个区域
        # 1. 当前理智显示区域 (大数字)
        # 2. 理智/上限显示区域

        current_sanity = None
        max_sanity = None

        # ===== 区域1: 当前理智显示区域 =====
        region1 = self.SANITY_CURRENT_DISPLAY_REGION
        x1 = int(region1[0] * w)
        y1 = int(region1[1] * h)
        rw1 = int(region1[2] * w)
        rh1 = int(region1[3] * h)

        region1_image = image[y1:y1+rh1, x1:x1+rw1]

        if save_debug_images and debug_dir:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / 'sanity_current_display.png'), region1_image)
            print(f"[调试] 当前理智区域已保存: x={x1}, y={y1}, w={rw1}, h={rh1}")

        # 识别当前理智 (大数字)
        for use_preprocessing in [False, True]:
            if use_preprocessing:
                processed = self._preprocessor.enhance_for_ocr(region1_image)
            else:
                processed = region1_image

            with self._lock:
                results = self._reader.readtext(processed, detail=1)

            if save_debug_images and debug_dir:
                print(f"[调试] 当前理智区域OCR结果: {len(results)}")
                for bbox, text, conf in results:
                    print(f"[调试]   - '{text}' (置信度: {conf:.3f})")

            for bbox, text, conf in results:
                # 查找纯数字 (当前理智值)
                match = re.search(r'(\d{1,3})', text.replace(' ', ''))
                if match:
                    val = int(match.group(1))
                    if 0 <= val <= 180:  # 理智值范围
                        current_sanity = val
                        if save_debug_images and debug_dir:
                            print(f"[调试] 当前理智识别成功: {current_sanity}")
                        break
            if current_sanity is not None:
                break

        # ===== 区域2: 理智/上限显示区域 =====
        region2 = self.SANITY_MAX_DISPLAY_REGION
        x2 = int(region2[0] * w)
        y2 = int(region2[1] * h)
        rw2 = int(region2[2] * w)
        rh2 = int(region2[3] * h)

        region2_image = image[y2:y2+rh2, x2:x2+rw2]

        if save_debug_images and debug_dir:
            cv2.imwrite(str(debug_dir / 'sanity_max_display.png'), region2_image)
            print(f"[调试] 理智上限区域已保存: x={x2}, y={y2}, w={rw2}, h={rh2}")

        best_confidence = 0.0
        best_position = None

        # 识别理智上限
        for use_preprocessing in [False, True]:
            if use_preprocessing:
                processed = self._preprocessor.enhance_for_ocr(region2_image)
                if save_debug_images and debug_dir:
                    cv2.imwrite(str(debug_dir / 'sanity_max_processed.png'), processed)
            else:
                processed = region2_image

            with self._lock:
                results = self._reader.readtext(processed, detail=1)

            if save_debug_images and debug_dir:
                print(f"[调试] 理智上限区域OCR结果: {len(results)}")
                for bbox, text, conf in results:
                    print(f"[调试]   - '{text}' (置信度: {conf:.3f})")

            for bbox, text, conf in results:
                cleaned_text = text.replace(' ', '').replace('O', '0').replace('o', '0')

                if save_debug_images and debug_dir:
                    print(f"[调试] 处理文字: '{cleaned_text}'")

                # 策略1: 匹配 "理智/上限" 格式
                match = re.search(r'(?:理智)?(\d{1,3})[/\\]?(\d{3})', cleaned_text)
                if match:
                    # 如果当前理智还没识别到，使用这里的值
                    if current_sanity is None:
                        current_sanity = int(match.group(1)) if match.group(1) else 0
                    max_val = int(match.group(2))

                    if save_debug_images and debug_dir:
                        print(f"[调试] 匹配成功: 当前={current_sanity}, 上限={max_val}")

                    if 127 <= max_val <= 180:
                        max_sanity = max_val
                        points = np.array(bbox, dtype=np.int32)
                        center_x = x2 + int(np.mean([p[0] for p in points]))
                        center_y = y2 + int(np.mean([p[1] for p in points]))
                        best_position = (center_x, center_y)
                        best_confidence = conf
                        break

                # 策略2: 处理 "/" 被识别成 "7" 的情况
                # 如 "理智7168" -> 取最后3位
                match2 = re.search(r'(?:理智)?(\d{1,3})?(\d{3})', cleaned_text)
                if match2 and match2.group(2):
                    max_val = int(match2.group(2))
                    if save_debug_images and debug_dir:
                        print(f"[调试] 策略2匹配: 上限={max_val}")

                    if 127 <= max_val <= 180:
                        max_sanity = max_val
                        # 如果当前理智还没识别到，尝试从前面获取
                        if current_sanity is None and match2.group(1):
                            current_sanity = int(match2.group(1))
                        points = np.array(bbox, dtype=np.int32)
                        center_x = x2 + int(np.mean([p[0] for p in points]))
                        center_y = y2 + int(np.mean([p[1] for p in points]))
                        best_position = (center_x, center_y)
                        best_confidence = conf
                        break

            if max_sanity is not None:
                break

        # 如果识别到了上限，但没有识别到当前理智，使用上限作为默认值
        if max_sanity is not None and current_sanity is None:
            current_sanity = max_sanity

        if max_sanity is not None:
            return SanityInfo(
                current=current_sanity or max_sanity,
                max=max_sanity,
                position=best_position or (x2 + rw2//2, y2 + rh2//2)
            )

        return None

    def recognize_resources(
        self,
        image: np.ndarray,
        save_debug_images: bool = False,
        debug_dir: Optional[Path] = None
    ) -> Dict[str, ResourceInfo]:
        """识别资源数量（龙门币、合成玉、源石）"""
        resources = {}
        h, w = image.shape[:2]

        resource_configs = [
            ('lmd', '龙门币', self.LMD_REGION),
            ('originite', '合成玉', self.ORIGINITE_REGION),
            ('orundum', '源石', self.ORUNDUM_REGION),
        ]

        for res_id, res_name, region in resource_configs:
            x = int(region[0] * w)
            y = int(region[1] * h)
            rw = int(region[2] * w)
            rh = int(region[3] * h)

            region_image = image[y:y+rh, x:x+rw]

            best_amount = None
            best_confidence = 0.0
            best_text = ""

            for use_preprocessing in [False, True]:
                if use_preprocessing:
                    processed = self._preprocessor.enhance_for_ocr(region_image)
                else:
                    processed = region_image

                if save_debug_images and debug_dir and use_preprocessing:
                    debug_path = debug_dir / f"resource_{res_id}_processed.png"
                    cv2.imwrite(str(debug_path), processed)
                    logger.debug(f"[调试] {res_name}预处理图像已保存: {debug_path}")

                with self._lock:
                    results = self._reader.readtext(processed, detail=1)

                logger.debug(f"[调试] {res_name}区域OCR结果 ({'预处理' if use_preprocessing else '原始'}):")
                for bbox, text, conf in results:
                    cleaned_text = text.replace(' ', '').replace('O', '0').replace('o', '0')
                    match = re.search(r'(\d{2,7}(?:,\d{3})*|\d{2,7})', cleaned_text)
                    if match:
                        try:
                            amount = int(match.group(1).replace(',', ''))
                            logger.debug(f"  - '{text}' -> 数字: {amount}, 置信度: {conf:.3f}")
                            if amount >= 0 and conf > best_confidence:
                                best_amount = amount
                                best_confidence = conf
                                best_text = text
                        except ValueError:
                            continue

            if best_amount is not None:
                logger.debug(f"[调试] {res_name}识别成功: {best_amount} (原始: '{best_text}', 置信度: {best_confidence:.3f})")
                resources[res_id] = ResourceInfo(
                    name=res_name,
                    amount=best_amount,
                    position=(x + rw//2, y + rh//2)
                )
            else:
                logger.debug(f"[调试] {res_name}识别失败")

        return resources

    @staticmethod
    def _calculate_text_similarity(text1: str, text2: str) -> float:
        """计算文字相似度"""
        t1 = text1.strip().lower()
        t2 = text2.strip().lower()

        if t1 == t2:
            return 1.0

        if t1 in t2 or t2 in t1:
            return 0.9

        from difflib import SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()


# =============================================================================
# 可视化工具
# =============================================================================

class Visualizer:
    """可视化工具 - 支持中文显示"""

    def __init__(self):
        self._font = None
        self._font_size = 20
        self._init_font()

    def _init_font(self):
        """初始化字体"""
        # 尝试加载中文字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]

        for font_path in font_paths:
            if Path(font_path).exists():
                self._font_path = font_path
                return

        self._font_path = None

    def draw_elements(
        self,
        image: np.ndarray,
        elements: List[UIElement],
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        绘制UI元素

        支持中文显示
        """
        vis = image.copy()

        # 颜色映射
        colors = {
            UIElementType.BUTTON: (0, 255, 0),    # 绿色
            UIElementType.TEXT: (255, 0, 0),      # 蓝色
            UIElementType.ICON: (0, 255, 255),    # 黄色
            UIElementType.TERMINAL: (255, 0, 255), # 紫色
            UIElementType.ACTIVITY: (0, 165, 255), # 橙色
            UIElementType.RESOURCE: (255, 255, 0), # 青色
        }

        for elem in elements:
            if elem is None:
                continue

            x, y, w, h = elem.position
            color = colors.get(elem.element_type, (128, 128, 128))

            # 绘制矩形
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

            # 绘制中心点
            cx, cy = elem.center
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)

            # 准备标签文字
            label = f"{elem.name}"
            if elem.text:
                label += f": {elem.text}"
            if elem.value is not None:
                label += f" = {elem.value}"
            label += f" ({elem.confidence:.2f})"

            # 使用PIL绘制中文
            vis = self._draw_chinese_text(vis, label, (x, y - 25), color)

        if save_path:
            cv2.imwrite(str(save_path), vis)

        return vis

    def _draw_chinese_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int]
    ) -> np.ndarray:
        """使用PIL绘制中文文字"""
        try:
            from PIL import Image, ImageDraw, ImageFont

            # 转换颜色 (BGR -> RGB)
            rgb_color = (color[2], color[1], color[0])

            # OpenCV转PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)

            # 加载字体
            if self._font_path:
                font = ImageFont.truetype(self._font_path, self._font_size)
            else:
                font = ImageFont.load_default()

            # 绘制文字背景
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x, y = position

            # 确保不超出边界
            y = max(text_h + 5, y)

            draw.rectangle([x, y - text_h - 5, x + text_w, y], fill=(0, 0, 0))

            # 绘制文字
            draw.text((x, y - text_h - 5), text, font=font, fill=rgb_color)

            # PIL转回OpenCV
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        except ImportError:
            # 如果没有PIL，使用OpenCV绘制（不支持中文）
            x, y = position
            cv2.putText(image, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            return image

    def draw_main_menu_result(
        self,
        image: np.ndarray,
        result: MainMenuResult,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """绘制主界面识别结果"""
        vis = image.copy()

        elements = []

        # 添加按钮
        for name, elem in result.buttons.items():
            if elem:
                elements.append(elem)

        # 添加终端信息
        if result.terminal:
            elem = UIElement(
                name="终端",
                element_type=UIElementType.TERMINAL,
                position=(result.terminal.position[0] - 50, result.terminal.position[1] - 20, 100, 40),
                confidence=1.0,
                text=result.terminal.name
            )
            elements.append(elem)

        # 添加活动
        for act in result.activities:
            elem = UIElement(
                name="活动",
                element_type=UIElementType.ACTIVITY,
                position=(act.position[0] - 40, act.position[1] - 15, 80, 30),
                confidence=1.0,
                text=act.name
            )
            elements.append(elem)

        # 添加理智
        if result.sanity:
            elem = UIElement(
                name="理智",
                element_type=UIElementType.RESOURCE,
                position=(result.sanity.position[0] - 40, result.sanity.position[1] - 15, 80, 30),
                confidence=1.0,
                text=str(result.sanity),
                value=result.sanity.current
            )
            elements.append(elem)

        # 添加资源信息
        for res_id, res in result.resources.items():
            elem = UIElement(
                name=res.name,
                element_type=UIElementType.RESOURCE,
                position=(res.position[0] - 50, res.position[1] - 15, 100, 30),
                confidence=1.0,
                text=f"{res.name}: {res.amount:,}"
            )
            elements.append(elem)

        return self.draw_elements(vis, elements, save_path)


# =============================================================================
# 主界面分析器
# =============================================================================

class MainMenuAnalyzer:
    """主界面分析器"""

    def __init__(self, use_gpu: bool = True):
        self.ocr = EnhancedOCRMatcher(use_gpu=use_gpu)
        self.visualizer = Visualizer()

    def initialize(self) -> bool:
        """初始化"""
        return self.ocr.initialize()

    def shutdown(self):
        """关闭"""
        self.ocr.shutdown()

    def analyze(
        self,
        image: np.ndarray,
        save_debug_images: bool = False,
        debug_dir: Optional[Path] = None
    ) -> MainMenuResult:
        """分析主界面

        Args:
            image: 输入图像
            save_debug_images: 是否保存调试图像
            debug_dir: 调试图像保存目录

        Returns:
            MainMenuResult对象
        """
        result = MainMenuResult()

        # 创建调试目录
        if save_debug_images and debug_dir:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"调试图像将保存到: {debug_dir}")

        # 1. 识别按钮
        logger.debug("识别按钮...")
        result.buttons = self.ocr.find_main_menu_buttons(image)

        # 2. 识别终端
        logger.debug("识别终端...")
        result.terminal = self.ocr.recognize_terminal(image)

        # 3. 识别活动
        logger.debug("识别活动...")
        result.activities = self.ocr.recognize_activities(image)

        # 4. 识别理智（带调试输出）
        logger.debug("识别理智...")
        result.sanity = self.ocr.recognize_sanity(
            image,
            save_debug_images=save_debug_images,
            debug_dir=debug_dir
        )

        # 5. 识别资源（龙门币、合成玉、源石）
        logger.debug("识别资源...")
        result.resources = self.ocr.recognize_resources(
            image,
            save_debug_images=save_debug_images,
            debug_dir=debug_dir
        )

        return result

    def visualize(
        self,
        image: np.ndarray,
        result: MainMenuResult,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """可视化结果"""
        return self.visualizer.draw_main_menu_result(image, result, save_path)


# =============================================================================
# 便捷函数
# =============================================================================

def analyze_main_menu(image_path: Union[str, Path]) -> MainMenuResult:
    """便捷函数：分析主界面"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    analyzer = MainMenuAnalyzer()
    if not analyzer.initialize():
        raise RuntimeError("OCR初始化失败")

    try:
        return analyzer.analyze(image)
    finally:
        analyzer.shutdown()


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("增强版GUI匹配器")
    print("=" * 60)

    # 测试图像预处理
    print("\n测试图像预处理...")
    test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    preprocessor = ImagePreprocessor()
    processed = preprocessor.enhance_for_ocr(test_image)
    print(f"  输入: {test_image.shape}")
    print(f"  输出: {processed.shape}")

    # 测试OCR（如果有图像）
    test_path = Path("主界面.png")
    if test_path.exists():
        print(f"\n测试主界面分析: {test_path}")

        analyzer = MainMenuAnalyzer()
        if analyzer.initialize():
            image = cv2.imread(str(test_path))
            result = analyzer.analyze(image)

            print("\n按钮识别结果:")
            for name, elem in result.buttons.items():
                if elem:
                    print(f"  ✓ {name:12s}: {elem.center} 置信度={elem.confidence:.3f} 文字='{elem.text}'")
                else:
                    print(f"  ✗ {name:12s}: 未找到")

            if result.terminal:
                print(f"\n终端: {result.terminal.name} @ {result.terminal.position}")

            if result.sanity:
                print(f"\n理智: {result.sanity}")

            print(f"\n活动数量: {len(result.activities)}")
            for act in result.activities[:3]:
                print(f"  - {act.name} @ {act.position}")

            # 可视化
            vis = analyzer.visualize(image, result, "main_menu_result.jpg")
            print("\n可视化结果已保存: main_menu_result.jpg")

            analyzer.shutdown()
        else:
            print("OCR初始化失败")
    else:
        print(f"\n测试图像不存在: {test_path}")

    print("\n测试完成!")
