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
视觉识别模块 - RTAS（Rhodeisland Tactical Autonomous System）

包含以下子模块：
- game_state_detector: 游戏状态检测（对局/非对局）
- nav_detector: 主界面/子菜单判定（主页按钮模板匹配）
- detector: 核心检测器（干员、敌人识别）
- ui_detector: UI信息检测
- colors: 颜色配置
- entities: 实体定义
- ocr: 可插拔 OCR 后端（EasyOCR / PaddleOCR / Umi-OCR / Tesseract）

使用示例：
    >>> from rtas.vision import GameStateDetector, GameState
    >>> detector = GameStateDetector()
    >>> detector.initialize()
    >>> result = detector.detect(image)
    >>> if result.state == GameState.IN_BATTLE:
    ...     print("当前在对局中")
"""

from .game_state_detector import (
    GameStateDetector,
    DetectorConfig,
    DetectionResult,
    GameState,
    ImagePreprocessor,
    TextMatcher,
    detect_game_state,
    create_default_detector,
)
from .nav_detector import (
    NavMenuButtonDetector,
    NavMenuButtonMatch,
    NavigationState,
    detect_navigation,
)
from .ocr_runtime import (
    EASYOCR_AVAILABLE,
    OCR_AVAILABLE,
    available_backend_name,
    create_reader,
    list_backends,
)

__all__ = [
    'GameStateDetector',
    'DetectorConfig',
    'DetectionResult',
    'GameState',
    'ImagePreprocessor',
    'TextMatcher',
    'detect_game_state',
    'create_default_detector',
    'EASYOCR_AVAILABLE',
    'OCR_AVAILABLE',
    'available_backend_name',
    'create_reader',
    'list_backends',
    'NavMenuButtonDetector',
    'NavMenuButtonMatch',
    'NavigationState',
    'detect_navigation',
]

__version__ = '1.0.0'
