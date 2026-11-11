# RTAS

**RTAS (Rhodeisland Tactical Autonomous System)** 是一个面向《明日方舟》的战术分析与自动化工具集，
提供游戏状态识别、编队/干员识别、关卡地图分析，以及结构化的游戏数据同步与查询能力。

> 本项目以 **GNU Affero General Public License v3.0 or later（AGPL-3.0-or-later）** 发布。

## 功能概览

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| CLI | `src/rtas/cli.py` | 统一命令行入口：检测、监控、GUI 匹配、编队识别、数据管理 |
| 数据层 | `src/rtas/data/` | GitHub 数据源、PRTS Wiki、SQLite 结构化数据库、领域模型 |
| 地图分析 | `src/rtas/map/` | 关卡解析、敌人路径、波次分析、地图可视化 |
| 视觉分析 | `src/rtas/vision/` | 游戏状态检测、GUI 元素匹配、编队与干员识别、文本定位 |

## 环境要求

- Python `>=3.10,<3.14`
- [uv](https://docs.astral.sh/uv/)（唯一的包管理与运行工具）
- Windows（视觉与屏幕捕获功能目前主要面向 Windows）

## 安装

```powershell
# 安装 uv（若尚未安装）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 创建虚拟环境并安装项目及依赖
uv sync

# 查看 CLI 帮助
uv run rtas --help
```

`uv sync` 会依据 `pyproject.toml` 与 `uv.lock` 创建 `.venv`，确保依赖可复现。
默认会同时安装 `dev` 依赖组（pytest、ruff）；只安装运行依赖可使用：

```powershell
uv sync --no-dev
```

## 使用

```powershell
# 检测单张截图
uv run rtas detect -i screenshot.png

# 批量检测并导出 JSON
uv run rtas detect -b .\screenshots -o results.json -f json

# 实时监控屏幕 60 秒
uv run rtas monitor -d 60 -n 0.5

# 主界面 / GUI 元素分析
uv run rtas main-menu -i screenshot.png -o result.jpg

# 编队识别
uv run rtas squad -i squad.png -j report.json

# 数据管理
uv run rtas data sync
uv run rtas data stats
uv run rtas data operator -n "阿米娅"
```

也可以通过模块方式运行：

```powershell
uv run python -m rtas --help
```

运行后产生的配置、缓存与日志默认位于 `~/.rtas/`（Windows 为 `%USERPROFILE%\.rtas\`）。

## 测试截图工具

`rtas-capture` 会自动查找《明日方舟》窗口，用 **Windows Graphics Capture**
直接读取窗口自身的画面并保存为 PNG，默认写入项目根目录的 `__temp__/`，
文件名形如 `arknights_20261112_001334.png`。

因为抓取的是窗口合成表面而不是屏幕，所以：

- **不需要把游戏切到前台**，终端挡在游戏前面也能截到干净画面；
- 其它置顶窗口、弹窗、鼠标光标不会混进截图；
- 被遮挡或最小化的游戏窗口也可以截取。

```powershell
# 自动查找并截图到 __temp__
uv run rtas-capture

# 查看当前所有可见窗口（找不到游戏时先用这个确认标题）
uv run rtas-capture --list

# 模拟器标题不含“明日方舟”时，用关键字或窗口句柄指定
uv run rtas-capture -k MuMu
uv run rtas-capture --hwnd 0x00123456

# 截取所有匹配窗口 / 指定输出目录 / 包含标题栏
uv run rtas-capture --all -o .\__temp__
uv run rtas-capture --full-window

# 确实需要先把游戏切到前台时
uv run rtas-capture --activate
```

也可以使用主命令的等价子命令：

```powershell
uv run rtas capture
```

实现位于 [src/rtas/tools/window_capture.py](src/rtas/tools/window_capture.py)，
仅支持 Windows 10 1903+。WGC 组件来自 MIT 许可的
[wgc-python](https://github.com/XuanChenxuan/wgc_python)，已内置到
`src/rtas/tools/_vendor/wgc/`，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
`--list` 不需要安装 OCR 依赖即可运行。

## 项目结构

```text
RTAS/
├── LICENSE                 # AGPL-3.0 许可证全文
├── THIRD_PARTY_NOTICES.md  # 内置 WGC 组件的 MIT 许可说明
├── README.md
├── pyproject.toml          # uv / PEP 621 项目与依赖配置
├── uv.lock                 # uv 锁文件（提交到版本库）
├── src/
│   └── rtas/
│       ├── __init__.py     # 版本与许可证元数据
│       ├── __main__.py     # python -m rtas 入口
│       ├── cli.py          # 命令行实现
│       ├── data/           # 数据模型、提供者与数据库
│       ├── map/            # 关卡与地图分析
│       ├── tools/          # 测试辅助工具（WGC 窗口截图）
│       └── vision/         # OCR / OpenCV 视觉分析
└── tests/                  # 许可证与包元数据检查
```

## 数据来源

- [Kengxxiao/ArknightsGameData](https://github.com/Kengxxiao/ArknightsGameData)：游戏数据仓库，
  由 `rtas data sync` 在运行时拉取到本地，本项目不随附游戏数据。
- [PRTS Wiki](https://prts.wiki/)：通过 MediaWiki API 查询干员、关卡与物品资料。
- 游戏素材、名称与相关内容的版权归上海鹰角网络科技有限公司（Hypergryph）所有。

## 开发

```powershell
# 代码检查
uv run ruff check src

# 语法编译检查
uv run python -m compileall -q src/rtas

# 构建分发包
uv build
```

## 许可证

Copyright (C) 2026 Ethernos Studio

本程序是自由软件：你可以在自由软件基金会发布的
**GNU Affero General Public License** 条款下重新分发和/或修改它，
可以选择许可证的 **第 3 版** 或（按你的意愿）**任何更新版本**。

本程序基于“希望它有用”的目的分发，但**不提供任何担保**，
甚至不提供适销性或特定用途适用性的默示担保。
详见 GNU Affero General Public License。

你应该已经随本程序收到 GNU Affero General Public License 的副本；
如果没有，请访问 <https://www.gnu.org/licenses/>。

许可证全文见 [LICENSE](LICENSE)。

### 网络使用与源码提供（AGPL 第 13 条）

如果你修改本项目并通过计算机网络向用户提供服务，你必须按照
AGPL-3.0 第 13 条的要求，向这些用户提供对应修改后版本的完整源码。
本项目的官方源码仓库为：

<https://github.com/Ethernos-Studio/RTAS>

### 第三方组件

项目依赖的第三方 Python 包遵循各自的许可证（例如 EasyOCR、OpenCV、PyTorch、
NumPy、Pillow、Requests 等）。这些依赖不会改变本项目自身的 AGPL-3.0-or-later 授权。
