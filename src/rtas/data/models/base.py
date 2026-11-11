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
基础数据模型定义

Author: RTAS Data Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List
import json


class DataSource(Enum):
    """数据来源类型"""
    GITHUB = auto()      # ArknightsGameData GitHub仓库
    PRTS_WIKI = auto()   # PRTS Wiki MediaWiki API
    LOCAL_CACHE = auto() # 本地缓存
    MANUAL = auto()      # 手动输入


@dataclass
class DataVersion:
    """数据版本信息"""
    version: str
    source: DataSource
    updated_at: datetime
    commit_hash: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version': self.version,
            'source': self.source.name,
            'updated_at': self.updated_at.isoformat(),
            'commit_hash': self.commit_hash,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """从字典创建"""
        return cls(
            version=data['version'],
            source=DataSource[data['source']],
            updated_at=datetime.fromisoformat(data['updated_at']),
            commit_hash=data.get('commit_hash'),
            description=data.get('description', '')
        )


@dataclass
class ArkDataModel:
    """
    明日方舟数据模型基类

    所有数据模型都继承此类，提供统一的序列化和反序列化接口
    """
    id: str
    name: str
    source: DataSource = DataSource.LOCAL_CACHE
    version: Optional[DataVersion] = None
    raw_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典表示"""
        result = {
            'id': self.id,
            'name': self.name,
            'source': self.source.name,
            'metadata': self.metadata
        }
        if self.version:
            result['version'] = self.version.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArkDataModel':
        """从字典创建实例"""
        version = None
        if 'version' in data:
            version = DataVersion.from_dict(data['version'])

        return cls(
            id=data['id'],
            name=data['name'],
            source=DataSource[data.get('source', 'LOCAL_CACHE')],
            version=version,
            metadata=data.get('metadata', {})
        )

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'ArkDataModel':
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))

    def __hash__(self) -> int:
        """基于ID的哈希"""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """基于ID的相等性比较"""
        if not isinstance(other, ArkDataModel):
            return NotImplemented
        return self.id == other.id


class DataCache:
    """
    数据缓存管理器

    提供内存缓存和本地文件缓存功能
    """

    def __init__(self, cache_dir: str = "cache"):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录路径
        """
        self._cache_dir = cache_dir
        self._memory_cache: Dict[str, Any] = {}
        self._cache_metadata: Dict[str, datetime] = {}

    def get(self, key: str, max_age: Optional[int] = None) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键
            max_age: 最大缓存时间（秒），None表示不过期

        Returns:
            缓存数据或None
        """
        # 先检查内存缓存
        if key in self._memory_cache:
            if max_age is None or self._is_valid(key, max_age):
                return self._memory_cache[key]
            else:
                del self._memory_cache[key]
                if key in self._cache_metadata:
                    del self._cache_metadata[key]

        return None

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值
        """
        self._memory_cache[key] = value
        self._cache_metadata[key] = datetime.now()

    def clear(self) -> None:
        """清空所有缓存"""
        self._memory_cache.clear()
        self._cache_metadata.clear()

    def _is_valid(self, key: str, max_age: int) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache_metadata:
            return False

        age = (datetime.now() - self._cache_metadata[key]).total_seconds()
        return age <= max_age

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'memory_entries': len(self._memory_cache),
            'total_size': sum(len(str(v)) for v in self._memory_cache.values())
        }
