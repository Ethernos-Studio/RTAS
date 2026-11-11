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
PRTS Wiki数据提供者

通过MediaWiki API查询PRTS Wiki数据

Author: RTAS Data Team
Version: 1.0.0
"""

import json
import logging
import re
import time
import html
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from urllib.parse import quote, urlencode
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models.base import DataSource, DataVersion

logger = logging.getLogger(__name__)


@dataclass
class PRTSConfig:
    """PRTS配置"""
    base_url: str = "https://prts.wiki"
    api_endpoint: str = "/api.php"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit: float = 0.5  # 请求间隔（秒）
    user_agent: str = "RTAS/1.0"


class PRTSDataProvider:
    """
    PRTS Wiki数据提供者

    通过MediaWiki API查询明日方舟Wiki数据
    """

    def __init__(self, config: Optional[PRTSConfig] = None):
        """
        初始化PRTS数据提供者

        Args:
            config: PRTS配置，使用默认配置如果为None
        """
        self.config = config or PRTSConfig()
        self._session: Optional[requests.Session] = None
        self._last_request_time: float = 0
        self._version: Optional[DataVersion] = None
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> bool:
        """
        初始化数据提供者

        Returns:
            是否初始化成功
        """
        try:
            logger.info("初始化PRTS数据提供者...")

            # 创建HTTP会话
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': self.config.user_agent,
                'Accept': 'application/json',
                'Accept-Language': 'zh-CN,zh;q=0.9'
            })

            # 配置重试策略
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # 测试连接
            if not self._test_connection():
                logger.error("无法连接到PRTS Wiki")
                return False

            # 初始化版本信息
            self._version = DataVersion(
                version="live",
                source=DataSource.PRTS_WIKI,
                updated_at=datetime.now(),
                description="PRTS Wiki MediaWiki API"
            )

            self._initialized = True
            logger.info("PRTS数据提供者初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False

    def _test_connection(self) -> bool:
        """测试API连接"""
        try:
            result = self._api_request({
                'action': 'query',
                'meta': 'siteinfo',
                'siprop': 'general',
                'format': 'json'
            })
            return result is not None and 'query' in result
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

    def _rate_limit(self) -> None:
        """速率限制"""
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_request_time
            if elapsed < self.config.rate_limit:
                time.sleep(self.config.rate_limit - elapsed)
            self._last_request_time = time.time()

    def _api_request(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        发送API请求

        Args:
            params: API参数

        Returns:
            API响应数据或None
        """
        if not self._session:
            logger.error("会话未初始化")
            return None

        self._rate_limit()

        url = f"{self.config.base_url}{self.config.api_endpoint}"

        try:
            response = self._session.get(
                url,
                params=params,
                timeout=self.config.timeout
            )
            response.raise_for_status()

            data = response.json()

            # 检查API错误
            if 'error' in data:
                logger.error(f"API错误: {data['error']}")
                return None

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            return None

    def search(self, query: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        搜索Wiki页面

        Args:
            query: 搜索关键词
            limit: 返回结果数量限制

        Returns:
            搜索结果列表
        """
        result = self._api_request({
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'format': 'json'
        })

        if not result or 'query' not in result:
            return []

        search_results = result['query'].get('search', [])
        return [
            {
                'title': item['title'],
                'snippet': html.unescape(item['snippet']),
                'pageid': item['pageid']
            }
            for item in search_results
        ]

    def get_page_content(self, title: str) -> Optional[str]:
        """
        获取页面Wikitext内容

        Args:
            title: 页面标题

        Returns:
            页面内容或None
        """
        result = self._api_request({
            'action': 'query',
            'prop': 'revisions',
            'titles': title,
            'rvprop': 'content',
            'rvslots': 'main',
            'format': 'json'
        })

        if not result or 'query' not in result:
            return None

        pages = result['query'].get('pages', {})
        for page_id, page_data in pages.items():
            if 'revisions' in page_data:
                revision = page_data['revisions'][0]
                content = revision['slots']['main']['*']
                return content

        return None

    def get_page_html(self, title: str) -> Optional[str]:
        """
        获取页面HTML内容

        Args:
            title: 页面标题

        Returns:
            页面HTML或None
        """
        result = self._api_request({
            'action': 'parse',
            'page': title,
            'prop': 'text',
            'format': 'json'
        })

        if not result or 'parse' not in result:
            return None

        return result['parse']['text']['*']

    def get_operator_info(self, operator_name: str) -> Optional[Dict[str, Any]]:
        """
        获取干员详细信息

        Args:
            operator_name: 干员名称

        Returns:
            干员信息字典或None
        """
        # 尝试获取干员页面
        content = self.get_page_content(operator_name)
        if not content:
            return None

        # 解析Wikitext中的数据模板
        info = {
            'name': operator_name,
            'wiki_content': content,
            'parsed_data': self._parse_operator_wikitext(content)
        }

        # 获取HTML渲染内容
        html_content = self.get_page_html(operator_name)
        if html_content:
            info['html_content'] = html_content

        return info

    def _parse_operator_wikitext(self, content: str) -> Dict[str, Any]:
        """
        解析干员Wikitext内容

        Args:
            content: Wikitext内容

        Returns:
            解析后的数据
        """
        data = {}

        # 解析干员数据模板
        char_data_match = re.search(
            r'\{\{干员数据\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        if char_data_match:
            template_content = char_data_match.group(1)
            # 解析模板参数
            for line in template_content.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

        # 解析技能信息
        skill_matches = re.findall(
            r'\{\{技能\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        data['skills'] = []
        for match in skill_matches:
            skill_data = {}
            for line in match.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    skill_data[key.strip()] = value.strip()
            data['skills'].append(skill_data)

        return data

    def get_stage_info(self, stage_code: str) -> Optional[Dict[str, Any]]:
        """
        获取关卡详细信息

        Args:
            stage_code: 关卡代码，如 "1-7"

        Returns:
            关卡信息字典或None
        """
        # 搜索关卡页面
        search_results = self.search(f"{stage_code} 关卡", limit=5)

        for result in search_results:
            if stage_code in result['title']:
                content = self.get_page_content(result['title'])
                if content:
                    return {
                        'title': result['title'],
                        'code': stage_code,
                        'wiki_content': content,
                        'parsed_data': self._parse_stage_wikitext(content)
                    }

        return None

    def _parse_stage_wikitext(self, content: str) -> Dict[str, Any]:
        """
        解析关卡Wikitext内容

        Args:
            content: Wikitext内容

        Returns:
            解析后的数据
        """
        data = {}

        # 解析关卡数据模板
        stage_data_match = re.search(
            r'\{\{关卡数据\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        if stage_data_match:
            template_content = stage_data_match.group(1)
            for line in template_content.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

        # 解析敌人配置
        enemy_matches = re.findall(
            r'\{\{关卡敌人\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        data['enemies'] = []
        for match in enemy_matches:
            enemy_data = {}
            for line in match.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    enemy_data[key.strip()] = value.strip()
            data['enemies'].append(enemy_data)

        return data

    def get_item_info(self, item_name: str) -> Optional[Dict[str, Any]]:
        """
        获取物品详细信息

        Args:
            item_name: 物品名称

        Returns:
            物品信息字典或None
        """
        content = self.get_page_content(item_name)
        if not content:
            return None

        return {
            'name': item_name,
            'wiki_content': content,
            'parsed_data': self._parse_item_wikitext(content)
        }

    def _parse_item_wikitext(self, content: str) -> Dict[str, Any]:
        """
        解析物品Wikitext内容

        Args:
            content: Wikitext内容

        Returns:
            解析后的数据
        """
        data = {}

        # 解析物品数据模板
        item_data_match = re.search(
            r'\{\{材料数据\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        if item_data_match:
            template_content = item_data_match.group(1)
            for line in template_content.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

        # 解析掉落信息
        drop_matches = re.findall(
            r'\{\{材料掉落\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        data['drops'] = []
        for match in drop_matches:
            drop_data = {}
            for line in match.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    drop_data[key.strip()] = value.strip()
            data['drops'].append(drop_data)

        return data

    def get_enemy_info(self, enemy_name: str) -> Optional[Dict[str, Any]]:
        """
        获取敌人详细信息

        Args:
            enemy_name: 敌人名称

        Returns:
            敌人信息字典或None
        """
        content = self.get_page_content(enemy_name)
        if not content:
            return None

        return {
            'name': enemy_name,
            'wiki_content': content,
            'parsed_data': self._parse_enemy_wikitext(content)
        }

    def _parse_enemy_wikitext(self, content: str) -> Dict[str, Any]:
        """
        解析敌人Wikitext内容

        Args:
            content: Wikitext内容

        Returns:
            解析后的数据
        """
        data = {}

        # 解析敌人数据模板
        enemy_data_match = re.search(
            r'\{\{敌人数据\|([^}]+)\}\}',
            content,
            re.DOTALL
        )
        if enemy_data_match:
            template_content = enemy_data_match.group(1)
            for line in template_content.split('\n|'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

        return data

    def get_category_members(self, category: str, limit: int = 50) -> List[str]:
        """
        获取分类成员

        Args:
            category: 分类名称
            limit: 返回数量限制

        Returns:
            页面标题列表
        """
        result = self._api_request({
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': limit,
            'format': 'json'
        })

        if not result or 'query' not in result:
            return []

        members = result['query'].get('categorymembers', [])
        return [item['title'] for item in members]

    def get_all_operators(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        获取所有干员列表

        Args:
            progress_callback: 进度回调函数

        Returns:
            干员信息列表
        """
        # 从分类获取干员列表
        operator_titles = self.get_category_members('干员', limit=500)

        operators = []
        total = len(operator_titles)

        for idx, title in enumerate(operator_titles):
            try:
                if progress_callback:
                    progress_callback(idx + 1, total)

                info = self.get_operator_info(title)
                if info:
                    operators.append(info)

            except Exception as e:
                logger.warning(f"获取干员 {title} 信息失败: {e}")
                continue

        return operators

    def get_version(self) -> Optional[DataVersion]:
        """获取当前数据版本"""
        return self._version

    def get_stats(self) -> Dict[str, Any]:
        """获取提供者统计信息"""
        return {
            'initialized': self._initialized,
            'base_url': self.config.base_url,
            'api_endpoint': self.config.api_endpoint,
            'rate_limit': self.config.rate_limit,
            'version': self._version.to_dict() if self._version else None
        }
