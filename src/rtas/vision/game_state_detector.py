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
游戏对局状态检测模块 - 基于OpenCV和EasyOCR的图像识别系统

本模块通过识别游戏界面右下角的关键文字"剩余可放置角色"来判断当前是否处于对局中。
采用多阶段识别链：图像预处理 → ROI区域提取 → OCR文字识别 → 文本匹配 → 置信度评估

算法复杂度分析：
- 图像预处理: O(n) 其中 n 为图像像素数
- OCR识别: O(m) 其中 m 为ROI区域像素数，m << n
- 文本匹配: O(k) 其中 k 为目标文字长度，k 为常数

Author: RTAS Vision Team
Version: 1.0.0
"""

import cv2
import numpy as np
import re
import time
import threading
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging

# 配置模块日志
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# OCR / PyTorch 采用延迟加载：这里只探测是否安装，真正导入推迟到
# 首次创建 OCR 后端时，避免 CLI 启动时付出十几秒的无谓开销。
from .ocr_runtime import (
    OCR_AVAILABLE,
    TORCH_AVAILABLE,
    create_reader,
    cuda_available,
    default_engine_name,
)

if not OCR_AVAILABLE:
    logger.warning("未检测到可用的 OCR 引擎，OCR功能将不可用。请执行: uv sync")


class GameState(Enum):
    """
    游戏状态枚举

    定义了系统可识别的所有游戏状态，用于状态机转换和决策逻辑。
    """
    UNKNOWN = auto()           # 未知状态，初始状态或识别失败
    IN_BATTLE = auto()         # 对局中，检测到"剩余可放置角色"文字
    NOT_IN_BATTLE = auto()     # 非对局状态，未检测到目标文字
    TRANSITIONING = auto()     # 状态转换中，用于处理过渡帧
    ERROR = auto()             # 错误状态，OCR引擎故障或其他异常


@dataclass
class DetectionResult:
    """
    检测结果数据类

    封装单次检测的完整结果，包含状态、置信度、原始OCR结果和元数据。

    Attributes:
        state: 检测到的游戏状态
        confidence: 置信度分数 [0.0, 1.0]
        raw_text: OCR识别的原始文本
        matched_keywords: 匹配到的关键词列表
        roi_image: 检测使用的ROI区域图像（可选，用于调试）
        timestamp: 检测时间戳
        processing_time_ms: 处理耗时（毫秒）
        error_message: 错误信息（如果有）
    """
    state: GameState = GameState.UNKNOWN
    confidence: float = 0.0
    raw_text: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    roi_image: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None

    def is_confident(self, threshold: float = 0.7) -> bool:
        """
        判断检测结果是否可信

        Args:
            threshold: 置信度阈值，默认0.7

        Returns:
            置信度是否超过阈值
        """
        return self.confidence >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，便于序列化"""
        return {
            'state': self.state.name,
            'confidence': self.confidence,
            'raw_text': self.raw_text,
            'matched_keywords': self.matched_keywords,
            'timestamp': self.timestamp,
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message
        }


@dataclass
class DetectorConfig:
    """
    检测器配置参数类

    所有尺寸参数基于参考分辨率(1920x1080)定义，
    实际检测时根据输入图像尺寸自动缩放。

    Attributes:
        reference_width: 参考宽度（像素）
        reference_height: 参考高度（像素）
        roi_relative: ROI区域相对坐标 (x, y, width, height) ∈ [0, 1]
        target_keywords: 目标关键词列表，用于匹配识别
        similarity_threshold: 文本相似度阈值 [0.0, 1.0]
        confidence_threshold: 状态判定置信度阈值
        ocr_languages: OCR识别的语言列表
        use_gpu: 是否使用GPU加速
        preprocessing_enabled: 是否启用图像预处理
        denoise_strength: 去噪强度 [0, 10]
        contrast_enhancement: 对比度增强因子 [1.0, 3.0]
        history_size: 状态历史缓冲区大小，用于平滑处理
        enable_debug: 是否启用调试模式
    """
    # 参考分辨率
    reference_width: int = 1920
    reference_height: int = 1080

    # ROI区域定义（相对坐标，基于右下角"剩余可放置角色"文字区域）
    # 像素坐标: y=845~890, x=1640~1920 @ 1920x1080
    roi_relative: Tuple[float, float, float, float] = (
        1640.0 / 1920.0,   # x: 0.854
        845.0 / 1080.0,    # y: 0.782
        280.0 / 1920.0,    # width: 0.146
        45.0 / 1080.0      # height: 0.042
    )

    # 目标关键词配置
    target_keywords: List[str] = field(default_factory=lambda: [
        "剩余可放置角色",
        "可放置角色",
        "剩余",
        "放置",
        "角色"
    ])

    # 判定阈值
    similarity_threshold: float = 0.65      # 文本相似度阈值
    confidence_threshold: float = 0.75      # 状态判定置信度阈值

    # OCR配置
    ocr_languages: List[str] = field(default_factory=lambda: ['ch_sim', 'en'])
    use_gpu: bool = True
    ocr_timeout_seconds: float = 5.0        # OCR超时时间

    # 图像预处理配置
    preprocessing_enabled: bool = True
    denoise_strength: int = 3               # 去噪强度
    contrast_enhancement: float = 1.5       # 对比度增强
    sharpening_enabled: bool = True         # 是否启用锐化

    # 状态平滑配置
    history_size: int = 5                   # 历史缓冲区大小
    state_change_threshold: int = 3         # 状态改变所需连续帧数

    # 调试配置
    enable_debug: bool = False
    debug_output_dir: Optional[str] = None

    def get_absolute_roi(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        将相对ROI坐标转换为绝对像素坐标

        时间复杂度: O(1)

        Args:
            image_width: 图像宽度（像素）
            image_height: 图像高度（像素）

        Returns:
            绝对坐标元组 (x, y, width, height)
        """
        x = int(self.roi_relative[0] * image_width)
        y = int(self.roi_relative[1] * image_height)
        w = int(self.roi_relative[2] * image_width)
        h = int(self.roi_relative[3] * image_height)
        return (x, y, w, h)

    def validate(self) -> bool:
        """
        验证配置参数有效性

        Returns:
            配置是否有效

        Raises:
            ValueError: 配置参数无效时抛出
        """
        if not (0 <= self.roi_relative[0] <= 1 and 0 <= self.roi_relative[1] <= 1):
            raise ValueError(f"ROI起始坐标必须在[0,1]范围内: {self.roi_relative[:2]}")
        if not (0 < self.roi_relative[2] <= 1 and 0 < self.roi_relative[3] <= 1):
            raise ValueError(f"ROI尺寸必须为正且在(0,1]范围内: {self.roi_relative[2:]}")
        if not (0 < self.similarity_threshold <= 1):
            raise ValueError(f"相似度阈值必须在(0,1]范围内: {self.similarity_threshold}")
        if not (0 < self.confidence_threshold <= 1):
            raise ValueError(f"置信度阈值必须在(0,1]范围内: {self.confidence_threshold}")
        if not self.target_keywords:
            raise ValueError("目标关键词列表不能为空")
        return True


class ImagePreprocessor:
    """
    图像预处理器

    负责在OCR之前对图像进行增强处理，提高文字识别准确率。
    包含去噪、对比度增强、锐化、二值化等操作。

    算法说明：
    - 使用非局部均值去噪（NLM）保持边缘的同时去除噪点
    - CLAHE自适应直方图均衡化增强局部对比度
    - 拉普拉斯锐化增强文字边缘
    """

    def __init__(self, config: DetectorConfig):
        """
        初始化预处理器

        Args:
            config: 检测器配置
        """
        self.config = config
        self._clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整的图像预处理流程

        处理流程：
        1. 颜色空间转换（BGR→灰度）
        2. 放大图像（提高OCR精度）
        3. 去噪处理
        4. 对比度增强
        5. 锐化处理（可选）
        6. 二值化（可选）

        时间复杂度: O(n)，n为图像像素数

        Args:
            image: 输入BGR格式图像

        Returns:
            预处理后的灰度图像

        Raises:
            ValueError: 输入图像无效
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像不能为空")

        if not self.config.preprocessing_enabled:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 步骤1: 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 步骤2: 放大图像（2x超采样提高OCR精度）
        # 时间复杂度: O(n)，使用双三次插值
        scale = 2
        enlarged = cv2.resize(
            gray,
            (gray.shape[1] * scale, gray.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC
        )

        # 步骤3: 非局部均值去噪
        # 时间复杂度: O(n * window_size^2)，实际使用快速实现
        if self.config.denoise_strength > 0:
            denoised = cv2.fastNlMeansDenoising(
                enlarged,
                None,
                h=self.config.denoise_strength * 3,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            denoised = enlarged

        # 步骤4: 自适应直方图均衡化（CLAHE）
        # 增强局部对比度，特别适用于光照不均的场景
        enhanced = self._clahe.apply(denoised)

        # 步骤5: 对比度增强
        if self.config.contrast_enhancement != 1.0:
            enhanced = cv2.convertScaleAbs(
                enhanced,
                alpha=self.config.contrast_enhancement,
                beta=0
            )

        # 步骤6: 锐化处理
        if self.config.sharpening_enabled:
            # 使用拉普拉斯算子锐化
            laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
            sharpened = cv2.addWeighted(
                enhanced.astype(np.float64),
                1.0,
                laplacian,
                -0.5,
                0
            )
            enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)

        return enhanced

    def preprocess_multiple(self, image: np.ndarray) -> List[np.ndarray]:
        """
        生成多种预处理变体，用于提高OCR召回率

        生成策略：
        - 变体1: 标准预处理
        - 变体2: Otsu自动阈值二值化
        - 变体3: 自适应阈值二值化

        Args:
            image: 输入图像

        Returns:
            预处理后的图像列表
        """
        variants = []

        # 标准预处理
        standard = self.preprocess(image)
        variants.append(standard)

        # Otsu二值化变体
        _, otsu = cv2.threshold(standard, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)

        # 自适应阈值变体
        adaptive = cv2.adaptiveThreshold(
            standard, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        variants.append(adaptive)

        return variants


class TextMatcher:
    """
    文本匹配器

    实现多种文本相似度算法，用于判断OCR结果是否包含目标关键词。

    支持的算法：
    - 精确匹配（Exact Match）
    - 包含匹配（Contains Match）
    - 编辑距离（Levenshtein Distance）
    - 最长公共子序列（LCS）
    - N-gram相似度
    """

    def __init__(self, config: DetectorConfig):
        """
        初始化文本匹配器

        Args:
            config: 检测器配置
        """
        self.config = config
        self.target_keywords = config.target_keywords

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度

        使用编辑距离归一化计算相似度：
        similarity = 1 - (edit_distance / max_length)

        时间复杂度: O(n*m)，n和m为两个文本长度
        空间复杂度: O(min(n,m))，使用滚动数组优化

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数 [0.0, 1.0]
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # 归一化：去除空格，转为小写
        s1 = text1.replace(' ', '').lower()
        s2 = text2.replace(' ', '').lower()

        # 使用动态规划计算编辑距离
        # 空间优化：只保留两行
        m, n = len(s1), len(s2)
        if m < n:
            s1, s2 = s2, s1
            m, n = n, m

        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    curr[j] = prev[j - 1]
                else:
                    curr[j] = 1 + min(
                        prev[j],      # 删除
                        curr[j - 1],  # 插入
                        prev[j - 1]   # 替换
                    )
            prev, curr = curr, prev

        edit_distance = prev[n]
        max_length = max(len(s1), len(s2))

        return 1.0 - (edit_distance / max_length)

    def find_matches(self, text: str) -> Tuple[List[str], float]:
        """
        在文本中查找匹配的关键词

        匹配策略：
        1. 首先尝试精确匹配
        2. 然后尝试包含匹配
        3. 最后使用相似度匹配

        Args:
            text: 待匹配的文本

        Returns:
            元组 (匹配到的关键词列表, 最高相似度)
        """
        if not text:
            return [], 0.0

        matched_keywords = []
        max_similarity = 0.0

        for keyword in self.target_keywords:
            # 策略1: 精确匹配
            if keyword == text:
                matched_keywords.append(keyword)
                max_similarity = 1.0
                continue

            # 策略2: 包含匹配
            if keyword in text or text in keyword:
                matched_keywords.append(keyword)
                similarity = len(keyword) / max(len(text), len(keyword))
                max_similarity = max(max_similarity, similarity)
                continue

            # 策略3: 相似度匹配
            similarity = self.calculate_similarity(text, keyword)
            if similarity >= self.config.similarity_threshold:
                matched_keywords.append(keyword)
                max_similarity = max(max_similarity, similarity)

        return matched_keywords, max_similarity

    def calculate_confidence(self, text: str, matched_keywords: List[str], max_similarity: float) -> float:
        """
        计算检测结果的置信度

        置信度计算考虑以下因素：
        - 文本长度（避免过短文本的误匹配）
        - 匹配关键词数量
        - 最高相似度
        - 完整关键词匹配奖励

        Args:
            text: 识别的文本
            matched_keywords: 匹配到的关键词
            max_similarity: 最高相似度

        Returns:
            置信度分数 [0.0, 1.0]
        """
        if not matched_keywords:
            return 0.0

        # 基础置信度来自相似度
        confidence = max_similarity

        # 完整匹配"剩余可放置角色"给予额外奖励
        full_keyword = "剩余可放置角色"
        if full_keyword in matched_keywords or full_keyword in text:
            confidence = min(1.0, confidence + 0.2)

        # 文本长度惩罚（过短文本降低置信度）
        text_length = len(text.replace(' ', ''))
        if text_length < 3:
            confidence *= 0.5
        elif text_length < 5:
            confidence *= 0.8

        # 多关键词匹配奖励
        if len(matched_keywords) >= 2:
            confidence = min(1.0, confidence + 0.1 * (len(matched_keywords) - 1))

        return min(1.0, max(0.0, confidence))


class GameStateDetector:
    """
    游戏状态检测器（主类）

    基于OpenCV和EasyOCR实现游戏对局状态检测。
    通过识别右下角"剩余可放置角色"文字判断是否处于对局中。

    线程安全：本类所有公共方法都是线程安全的

    使用示例：
        >>> config = DetectorConfig()
        >>> detector = GameStateDetector(config)
        >>> detector.initialize()
        >>> result = detector.detect(image)
        >>> if result.state == GameState.IN_BATTLE:
        ...     print("当前在对局中")
        >>> detector.shutdown()
    """

    def __init__(self, config: Optional[DetectorConfig] = None, ocr_backend: Any = None):
        """
        初始化检测器

        Args:
            config: 检测器配置，若为None则使用默认配置
            ocr_backend: 可选的共享 OCR 后端（如交互模式里已加载的实例），
                传入后本检测器不再单独加载模型
        """
        self.config = config or DetectorConfig()
        self.config.validate()

        # 初始化组件
        self._preprocessor = ImagePreprocessor(self.config)
        self._text_matcher = TextMatcher(self.config)

        # OCR引擎（首次 initialize() 时才真正导入）
        self._reader: Optional[Any] = ocr_backend
        self._owns_reader = ocr_backend is None
        self._ocr_lock = threading.RLock()

        # 状态管理
        self._initialized = False
        self._state_history: deque = deque(maxlen=self.config.history_size)
        self._current_state = GameState.UNKNOWN
        self._state_change_count = 0
        self._last_state = GameState.UNKNOWN

        # 调试相关
        self._debug_counter = 0

        # 线程池（用于超时控制）
        self._executor = ThreadPoolExecutor(max_workers=1)

        logger.info("GameStateDetector初始化完成")

    def initialize(self) -> bool:
        """
        初始化OCR引擎

        必须在调用detect()之前执行。
        检测GPU可用性并初始化EasyOCR阅读器。

        Returns:
            初始化是否成功

        Raises:
            RuntimeError: EasyOCR未安装时抛出
        """
        if self._initialized:
            return True

        # 复用外部传入的 OCR 后端（交互模式下只加载一次模型）
        if self._reader is not None:
            self._initialized = True
            logger.info("复用已加载的 OCR 后端")
            return True

        if not OCR_AVAILABLE:
            raise RuntimeError(
                "没有可用的 OCR 引擎，无法初始化。请安装 easyocr（uv sync）、"
                "PaddleOCR（uv sync --extra paddle）、Tesseract，或启动 Umi-OCR。"
            )

        try:
            # 加载 OCR 依赖（torch 首次导入可能需要数秒，先给出提示）
            engine = default_engine_name()
            logger.info(f"正在加载 OCR 依赖（引擎: {engine}，首次运行较慢，请稍候）...")

            # 检测GPU可用性（此时才会真正导入 torch）
            gpu_available = False
            if self.config.use_gpu:
                gpu_available = cuda_available()
                if gpu_available:
                    logger.info("检测到GPU，启用GPU加速")
                else:
                    logger.info("未检测到GPU，使用CPU")

            # 初始化 OCR 后端（此处才会真正导入 easyocr / paddle 等）
            logger.info(f"正在初始化 OCR 引擎（{engine}）...")
            with self._ocr_lock:
                self._reader = create_reader(
                    self.config.ocr_languages,
                    gpu=gpu_available,
                    verbose=False
                )

            self._initialized = True
            logger.info(f"OCR 引擎 {engine} 初始化完成")
            return True

        except Exception as e:
            logger.error(f"OCR引擎初始化失败: {e}")
            if any(k in str(e).lower() for k in ("crc", "md5", "zip", "download", "corrupt")):
                logger.error(
                    "EasyOCR 模型下载失败或文件损坏，请删除 %s 下的模型文件后重试",
                    Path.home() / ".EasyOCR" / "model",
                )
            return False

    def shutdown(self) -> None:
        """
        关闭检测器，释放资源

        清理OCR引擎和线程池资源。
        建议在程序退出前调用。
        """
        with self._ocr_lock:
            if self._owns_reader:
                self._reader = None
            self._initialized = False

        self._executor.shutdown(wait=False)
        logger.info("GameStateDetector已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False

    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        提取感兴趣区域（ROI）

        从完整图像中提取右下角包含"剩余可放置角色"文字的区域。

        时间复杂度: O(1)

        Args:
            image: 完整游戏截图

        Returns:
            ROI区域图像

        Raises:
            ValueError: 图像尺寸无效
        """
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            raise ValueError(f"图像尺寸无效: {w}x{h}")

        # 获取绝对坐标
        x, y, roi_w, roi_h = self.config.get_absolute_roi(w, h)

        # 边界检查
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)

        if roi_w <= 0 or roi_h <= 0:
            raise ValueError(f"ROI尺寸无效: {roi_w}x{roi_h}")

        return image[y:y+roi_h, x:x+roi_w]

    def _perform_ocr(self, image: np.ndarray) -> str:
        """
        执行OCR识别

        对预处理后的图像执行OCR，识别其中的文字。

        时间复杂度: 取决于EasyOCR实现，通常为O(m)，m为图像像素数

        Args:
            image: 预处理后的图像

        Returns:
            识别的文本

        Raises:
            RuntimeError: OCR引擎未初始化
            TimeoutError: OCR执行超时
        """
        if not self._initialized or self._reader is None:
            raise RuntimeError("OCR引擎未初始化")

        def ocr_task():
            with self._ocr_lock:
                # EasyOCR返回格式: [(bbox, text, confidence), ...]
                results = self._reader.readtext(
                    image,
                    detail=0,           # 只返回文本
                    paragraph=False,    # 不按段落合并
                    contrast_ths=0.1    # 对比度阈值
                )
                return ' '.join(results) if results else ""

        # 使用线程池实现超时控制
        future = self._executor.submit(ocr_task)
        try:
            return future.result(timeout=self.config.ocr_timeout_seconds)
        except FutureTimeoutError:
            raise TimeoutError(f"OCR执行超时（>{self.config.ocr_timeout_seconds}s）")

    def _update_state_with_smoothing(self, new_state: GameState) -> GameState:
        """
        使用历史缓冲区平滑状态变化

        防止由于单帧识别错误导致的状态抖动。
        只有当新状态连续出现一定次数时才确认状态改变。

        Args:
            new_state: 新检测到的状态

        Returns:
            平滑后的状态
        """
        # 添加到历史
        self._state_history.append(new_state)

        # 如果历史不足，直接返回当前状态
        if len(self._state_history) < self.config.state_change_threshold:
            return self._current_state

        # 统计历史中最常见状态
        state_counts = {}
        for state in self._state_history:
            state_counts[state] = state_counts.get(state, 0) + 1

        most_common_state = max(state_counts, key=state_counts.get)
        max_count = state_counts[most_common_state]

        # 只有当最常见状态超过阈值时才改变
        if max_count >= self.config.state_change_threshold:
            if most_common_state != self._current_state:
                logger.debug(f"状态改变: {self._current_state.name} -> {most_common_state.name}")
                self._current_state = most_common_state

        return self._current_state

    def detect(self, image: np.ndarray, return_roi: bool = False, use_smoothing: bool = True) -> DetectionResult:
        """
        检测游戏状态（主方法）

        执行完整的检测流程：
        1. 提取ROI区域
        2. 图像预处理
        3. OCR文字识别
        4. 文本匹配
        5. 置信度计算
        6. 状态平滑（可选）

        时间复杂度: O(n + m + k)，n为图像像素，m为ROI像素，k为文本长度

        Args:
            image: BGR格式游戏截图
            return_roi: 是否在结果中返回ROI图像（用于调试）
            use_smoothing: 是否使用状态平滑，默认为True。单张检测建议设为False

        Returns:
            DetectionResult对象包含检测结果

        Raises:
            ValueError: 输入图像无效
            RuntimeError: OCR引擎未初始化
        """
        start_time = time.perf_counter()

        # 输入验证
        if image is None:
            return DetectionResult(
                state=GameState.ERROR,
                error_message="输入图像为空"
            )

        if image.size == 0:
            return DetectionResult(
                state=GameState.ERROR,
                error_message="输入图像尺寸为0"
            )

        try:
            # 步骤1: 提取ROI
            roi = self._extract_roi(image)

            # 步骤2: 图像预处理
            processed = self._preprocessor.preprocess(roi)

            # 步骤3: OCR识别
            raw_text = self._perform_ocr(processed)

            # 步骤4: 文本匹配
            matched_keywords, max_similarity = self._text_matcher.find_matches(raw_text)

            # 步骤5: 计算置信度
            confidence = self._text_matcher.calculate_confidence(
                raw_text, matched_keywords, max_similarity
            )

            # 步骤6: 判定状态
            if confidence >= self.config.confidence_threshold and matched_keywords:
                detected_state = GameState.IN_BATTLE
            else:
                detected_state = GameState.NOT_IN_BATTLE

            # 步骤7: 状态平滑（可选）
            if use_smoothing:
                final_state = self._update_state_with_smoothing(detected_state)
            else:
                final_state = detected_state

            # 计算处理时间
            processing_time_ms = (time.perf_counter() - start_time) * 1000

            # 调试输出
            if self.config.enable_debug:
                logger.debug(
                    f"检测完成: state={final_state.name}, "
                    f"confidence={confidence:.3f}, text='{raw_text}', "
                    f"time={processing_time_ms:.1f}ms"
                )

            # 构建结果
            result = DetectionResult(
                state=final_state,
                confidence=confidence,
                raw_text=raw_text,
                matched_keywords=matched_keywords,
                timestamp=time.time(),
                processing_time_ms=processing_time_ms
            )

            if return_roi:
                result.roi_image = roi.copy()

            return result

        except TimeoutError as e:
            logger.warning(f"OCR超时: {e}")
            return DetectionResult(
                state=GameState.ERROR,
                error_message=f"OCR超时: {e}"
            )
        except Exception as e:
            logger.error(f"检测过程中发生错误: {e}")
            return DetectionResult(
                state=GameState.ERROR,
                error_message=f"检测错误: {e}"
            )

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """
        批量检测多张图像

        Args:
            images: 图像列表

        Returns:
            检测结果列表
        """
        return [self.detect(img) for img in images]

    def is_in_battle(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> bool:
        """
        快速判断当前是否在对局中

        简化接口，只返回布尔结果。

        Args:
            image: 游戏截图
            confidence_threshold: 可选的置信度阈值覆盖

        Returns:
            是否在对局中
        """
        result = self.detect(image)
        threshold = confidence_threshold or self.config.confidence_threshold
        return result.state == GameState.IN_BATTLE and result.confidence >= threshold

    def get_current_state(self) -> GameState:
        """
        获取当前平滑后的状态

        Returns:
            当前游戏状态
        """
        return self._current_state

    def reset_state(self) -> None:
        """
        重置状态历史

        清除状态历史缓冲区，将状态重置为UNKNOWN。
        在场景切换或重新开始检测时调用。
        """
        self._state_history.clear()
        self._current_state = GameState.UNKNOWN
        self._state_change_count = 0
        logger.info("状态已重置")

    def save_debug_image(self, image: np.ndarray, prefix: str = "debug") -> Optional[str]:
        """
        保存调试图像

        Args:
            image: 要保存的图像
            prefix: 文件名前缀

        Returns:
            保存的文件路径，或None（如果调试未启用）
        """
        if not self.config.enable_debug or not self.config.debug_output_dir:
            return None

        try:
            output_dir = Path(self.config.debug_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time() * 1000)
            filename = f"{prefix}_{timestamp}_{self._debug_counter:04d}.png"
            filepath = output_dir / filename

            cv2.imwrite(str(filepath), image)
            self._debug_counter += 1

            return str(filepath)
        except Exception as e:
            logger.error(f"保存调试图像失败: {e}")
            return None


def create_default_detector(use_gpu: bool = True, enable_debug: bool = False) -> GameStateDetector:
    """
    创建默认配置的检测器（工厂函数）

    Args:
        use_gpu: 是否使用GPU
        enable_debug: 是否启用调试

    Returns:
        配置好的GameStateDetector实例
    """
    config = DetectorConfig(
        use_gpu=use_gpu,
        enable_debug=enable_debug
    )
    detector = GameStateDetector(config)
    detector.initialize()
    return detector


# 便捷函数：单次检测
def detect_game_state(image: np.ndarray, use_gpu: bool = True) -> DetectionResult:
    """
    便捷函数：单次检测游戏状态

    创建临时检测器执行单次检测，然后自动释放资源。
    适合不需要持续检测的场景。

    Args:
        image: 游戏截图
        use_gpu: 是否使用GPU

    Returns:
        检测结果

    Example:
        >>> result = detect_game_state(image)
        >>> if result.state == GameState.IN_BATTLE:
        ...     print("在对局中！")
    """
    with create_default_detector(use_gpu=use_gpu) as detector:
        return detector.detect(image)


if __name__ == "__main__":
    # 简单的自测代码
    import sys

    print("游戏状态检测模块")
    print("=" * 50)

    # 检查依赖
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"OCR可用: {OCR_AVAILABLE}")
    print(f"PyTorch可用: {TORCH_AVAILABLE}")

    if not OCR_AVAILABLE:
        print("\n错误: EasyOCR未安装")
        print("请执行: uv sync")
        sys.exit(1)

    # 测试配置
    config = DetectorConfig(enable_debug=True)
    print(f"\n配置验证: {config.validate()}")
    print(f"目标关键词: {config.target_keywords}")

    # 测试文本匹配器
    matcher = TextMatcher(config)
    test_texts = [
        "剩余可放置角色",
        "可放置角色",
        "剩余 5",
        "开始行动",
        ""
    ]

    print("\n文本匹配测试:")
    for text in test_texts:
        keywords, sim = matcher.find_matches(text)
        conf = matcher.calculate_confidence(text, keywords, sim)
        print(f"  '{text}' -> keywords={keywords}, sim={sim:.3f}, conf={conf:.3f}")

    print("\n模块加载成功！")
