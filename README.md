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
| 关卡推演 | `src/rtas/sim/` | 敌人走位/阻挡/输出推演、位置图与推演视频 |
| 视觉分析 | `src/rtas/vision/` | 游戏状态检测、GUI 元素匹配、编队与干员识别、文本定位 |
| OCR 后端 | `src/rtas/vision/ocr/` | EasyOCR / PaddleOCR / Umi-OCR / Tesseract 统一接口 |
| 导航判定 | `src/rtas/vision/nav_detector.py` | 主界面/子菜单判定、顶部导航栏按钮坐标 |

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
uv run rtas squad                      # 省略 -i：直接抓当前游戏窗口

# 数据管理
uv run rtas data sync
uv run rtas data stats
uv run rtas data operator -n "阿米娅"

# 判断截图在主界面还是子菜单，并给出「打开顶部导航栏」按钮坐标
uv run rtas nav -i screenshot.png

# 查看/试用 OCR 引擎
uv run rtas ocr --list
uv run rtas ocr -i screenshot.png --engine paddleocr

# 关卡推演（详见「关卡推演（sim）」章节）
uv run rtas sim map -l ArknightsGameData/zh_CN/gamedata/levels/obt/main/level_main_01-07.json -t 30 -o t30.png
```

也可以通过模块方式运行：

```powershell
uv run python -m rtas --help
```

运行后产生的配置、缓存与日志默认位于 `~/.rtas/`（Windows 为 `%USERPROFILE%\.rtas\`）。
结构化数据库统一位于当前工作目录下的 `cache/rtas_structured.db`。

## 直接截图分析

图像类命令（`detect`、`nav`、`ocr`、`gui`、`main-menu`、`squad`）**省略 `-i` 就会现场抓游戏窗口**
再分析，不用先手动截图：

```powershell
# 游戏开着就能直接跑，截到的图会存进 __temp__ 并写进日志
uv run rtas nav
uv run rtas squad --json report.json
uv run rtas detect
uv run rtas main-menu -o main.jpg

# 窗口标题/进程不含「明日方舟 / arknights」时，用关键字指定
uv run rtas squad --window MuMu
uv run rtas nav --window BlueStacks --activate
```

- 抓图走 Windows Graphics Capture，读窗口自身的画面，**被其它窗口遮挡也能截**，不需要游戏获得焦点。
- 同时匹配到多个窗口时，取客户区最大的那个（通常是游戏本体，其余多为启动器/工具栏）。
- 每次都会在 `__temp__` 下留一张 `arknights_<时间戳>.png`，方便回看当时的画面；
  `--json` 输出里的 `file` 字段就是这张图的路径。
- `--activate` 会在截图前把窗口切到前台（某些模拟器需要）；默认不抢焦点。
- `ocr --list` 仍然只列引擎，不会触发截图。
- 想先确认能匹配到哪些窗口：`uv run rtas capture --list`。

## OCR 引擎

视觉命令默认使用 EasyOCR，也可以切换或混用以下 4 种引擎（全部按需加载，
不影响 `rtas --help`、`rtas capture`、`rtas nav` 的启动速度）：

| 引擎 | 安装方式 | 说明 |
| --- | --- | --- |
| `easyocr` | 默认依赖（`uv sync`） | 中文/英文稳定，CPU 可用 |
| `paddleocr` | `uv sync --extra paddle` | PP-OCR 系列，中文准确率高，首次安装较大 |
| `umi` | 安装 [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) 并开启 HTTP 服务 | 走本机 HTTP（默认 `http://127.0.0.1:1224`），无需在项目里装 Paddle |
| `tesseract` | 安装 Tesseract-OCR 并加入 PATH | 轻量，需要 `chi_sim` 语言包 |

选择顺序：`--ocr` 参数 > 配置文件 `ocr_engine` > 环境变量 `RTAS_OCR_ENGINE` >
`auto`（按 easyocr → paddleocr → umi → tesseract 取第一个可用的）。

```powershell
# 查看各引擎是否可用
uv run rtas ocr --list

# 临时用某个引擎跑一张图
uv run rtas --ocr paddleocr ocr -i screenshot.png
uv run rtas ocr -i screenshot.png --engine umi

# 持久化到配置文件（~/.rtas/config.json）
uv run rtas config --set ocr_engine paddleocr
```

Umi-OCR 地址可用 `--ocr-url http://127.0.0.1:1224` 或环境变量 `RTAS_UMI_URL` 指定；
Tesseract 可执行文件可用 `RTAS_TESSERACT_CMD` 指定。

## 主界面 / 子菜单判定

子菜单（终端、干员、采购中心、基建……）左上角有一个「打开顶部导航栏」按钮
（网状球体 + 房子图标，位于返回箭头 `<` 与 `i` 信息按钮之间）。点击它会弹出包含
首页 / 编队 / 干员 / 档案 / 终端 / 基建 / 公开招募 / 干员寻访 / 采购中心 的顶部导航栏，
再点其中的「首页」才回到主界面。它不是「回到主界面」按钮。

`rtas nav` 用模板匹配判断截图处在哪一层：

- 主界面：没有该按钮（顶部导航栏本身可见）→ `state = main_menu`
- 任意子菜单：左上角有该按钮 → `state = sub_menu`，同时给出按钮中心坐标
  （点击它即可打开顶部导航栏）

这正是 MAA 的通用导航思路：任意子菜单 → 点导航按钮 → 点「首页」，
不必为每个界面单独写识别。

```powershell
uv run rtas nav -i screenshot.png            # 文本输出
uv run rtas nav -i screenshot.png --json     # JSON（含坐标）
uv run rtas nav -i screenshot.png -o nav.jpg # 画框可视化
```

模板来自游戏 UI 资源里的 `btn_topmenu_home`（只取房子高亮部分），存放在
`src/rtas/vision/templates/ui/home_btn_house.png`，用多尺度灰度匹配适配
720p～4K 分辨率。当前实测：1264×759 采购中心截图相似度 0.96，主界面最高 0.61。

## 后台点击与一键进入界面

`click` / `goto` / `key` / `shell` 使用**不抢鼠标**的后台点击：既不动系统光标，
也不切换前台窗口，你可以同时继续用电脑。

```powershell
# 在截图坐标点击（坐标与 rtas capture / nav 输出的截图一致）
uv run rtas click 640 360

# 一键进入某个界面：先判断层级，必要时打开顶部导航栏再点目标项
uv run rtas goto 编队
uv run rtas goto 首页
uv run rtas goto 采购中心 --dry-run   # 只打印计划，不点击

# 发送按键
uv run rtas key esc
```

点击方式由 `--method` 选择，默认 `window`：

| 方式 | 行为 | 适用 |
| --- | --- | --- |
| `window`（默认） | 把窗口移到目标点正好位于真实光标下方，再投递鼠标消息，结束后还原窗口位置 | Unity 客户端（明日方舟 PC 端），对应 MAA 的 `PostMessageWithWindowPos` |
| `post` | 纯 `PostMessage`，窗口完全不动 | 模拟器（Unity 客户端会忽略这类点击） |
| `send` | `SendMessage`，同步等窗口处理 | 调试 |
| `cursor` | 真实移动光标并点击 | 兜底，会抢鼠标 |

> **必须以管理员身份运行。** 游戏进程的完整性级别高于普通进程，Windows 的
> UIPI 会拦截低权限进程的 `SetWindowPos` / `PostMessage`（表现为
> `SetWindowPos` 返回失败、`GetLastError() == 5`，点击没有任何效果）。
> 这与 MAA 的要求一致；非管理员运行时 CLI 会打印警告。

```powershell
# 自动弹 UAC 以管理员身份重新启动同一条命令
uv run rtas goto 编队 --elevate
```

## 交互式模式

`rtas shell` 只加载一次窗口/OCR 资源，之后连续执行命令，避免反复启动的开销。

**shell 里的命令与 CLI 完全一致**：命令行怎么写，会话里就怎么写。参数由同一个
`argparse` 解析器解析、由同一个执行函数处理，所以选项、输出、退出码都相同；
区别只是配置、日志、检测器在会话内复用，不会再初始化一次 OCR。

```powershell
# 人类可读的 REPL
uv run rtas shell

# quiet 模式：stdin 一行一条命令，stdout 一行一条 JSON（日志走 stderr）
uv run rtas shell -q
```

```text
rtas> detect -i __temp__\shot.png -f json      # 与 rtas detect 完全相同
rtas> monitor -d 30 -n 0.5
rtas> data operator -n "阿米娅"
rtas> squad -i squad.png -j report.json
rtas> ocr --list
rtas> config --show
rtas> --help                                   # 完整 CLI 帮助
rtas> help detect                              # 单个子命令的帮助
```

常用窗口操作还有不带参数的快捷写法，直接作用于会话已加载的窗口：

| 快捷命令 | 说明 |
| --- | --- |
| `status` | 窗口 / 权限 / 点击方式 |
| `nav` / `detect` / `ocr [top]` | 直接截当前窗口做判定或识别 |
| `goto <菜单>` / `click <x> <y>` / `key <按键>` | 复用会话里的 GameAgent，不必重新找窗口 |
| `capture [路径]` | 截图保存 |
| `sleep <秒>` / `help` / `quit` | 会话控制 |

规则很简单：**不带 `-` 选项时走上面的快捷方式；一旦出现 `-x` / `--xxx`
就交给 CLI 解析**（例如 `nav -i shot.png --json`、`goto 编队 --dry-run`）。
`--config` / `--no-gpu` / `--log-level` 只影响它所在的那一条命令，
其余命令沿用启动 shell 时的配置；`--ocr` 可以逐条切换引擎。

quiet 模式适合被其它程序（UI、脚本）当作 stdio 服务调用，协议示例：

```text
> status
{"ok": true, "command": "status", "window": {...}, "admin": false, "method": "window"}
> goto 编队
{"ok": true, "command": "goto", "target": "编队", "steps": [{"action": "open_nav_bar", ...}]}
> click 640 360
{"ok": true, "command": "click", "x": 640, "y": 360}
> config --show
{"ok": true, "command": "config", "exit_code": 0, "output": "当前配置:\n{...}"}
> quit
{"ok": true, "command": "quit", "exit": true}
```

转发的 CLI 命令把标准输出放在 `output`、退出码放在 `exit_code`；失败时额外给出
`{"ok": false, "error": "..."}`（`error` 取自 stderr 或日志里的告警/错误）。

## 关卡推演（sim）

`rtas sim` 直接读关卡数据做「走位 → 阻挡 → 输出」推演，不需要游戏在运行：

| 子命令 | 作用 |
| --- | --- |
| `sim map` | 某个时刻场上敌人都在哪（不模拟阻挡），输出图片或 JSON |
| `sim run` | 完整推演（可部署干员），导出视频 + 数据 |
| `sim tiles` | 列出可部署地块，生成计划模板 |

支持两种关卡格式：

- **RTAS 简化格式**：`tiles` / `routes` / `waves`（`rtas level` 用的那种），敌人属性写在 `enemies` 里；
- **明日方舟原版关卡**：`ArknightsGameData/zh_CN/gamedata/levels/**.json`，
  会自动读取同级的 `enemydata/enemy_database.json` 拿到血量 / 防御 / 移速 / 阻挡权重，
  路线缺少 `checkpoints` 时用 A* 在可通行地块上补全，门格不是 `tile_start` / `tile_end`
  时按路线端点补成红门 / 蓝门。

```powershell
# 关卡路径
$LV = "ArknightsGameData/zh_CN/gamedata/levels/obt/main/level_main_01-07.json"

# 1) 第 30 秒的敌人位置图（无阻挡），也可以 -t 30,45,60 一次出多张
uv run rtas sim map -l $LV -t 30 -o t30.png

# 2) 红门 → 蓝门的无阻挡走位视频
uv run rtas sim run -l $LV -t 60 -o noblock.mp4 --fps 20

# 3) 5 秒放一个阻挡 1 的近卫（600 伤害 / 1 秒，只打自己挡住的敌人）
uv run rtas sim run -l $LV -o run.mp4 --json run.json `
  --op "name=近卫,t=5,tile=3:2,block=1,damage=600,interval=1.0,mode=melee"

# 4) 不知道能放哪：先列地块、生成计划模板，改完直接跑
uv run rtas sim tiles -l $LV --plan-template plan.json
uv run rtas sim run -l $LV -p plan.json -o run.mp4
```

`--op` 支持的字段：`name`、`t`（部署时刻）、`tile=行:列`、`block`、`damage`、
`interval`（攻击间隔）、`mode=melee|ranged`、`range`（远程射程，格）、`retreat`（撤退时刻）。

推演规则（简化、可解释）：

- 敌人按 `waves` 计算出的绝对时间在红门生成，按「格 / 秒」沿路线移动，
  原版关卡的速度会乘以关卡的 `moveMultiplier`；
- 地面敌人走到干员所在格时，若干员还有阻挡余量（`block` ≥ 已占权重 + 该敌人 `weight`）就被挡住，
  飞行敌人无视阻挡；
- 干员按 `attack_interval` 出手：`melee` 只打自己阻挡的敌人，`ranged` 打射程内最靠前的敌人；
  伤害先扣防御，最低保底 5%；
- 血量归零 = 击杀（干员记击杀数 / 总伤害），走到蓝门 = 漏怪并扣生命值。

输出物：`.mp4` 视频（OpenCV `mp4v`）、`.gif`（PIL）、`.json`（含逐帧 `frames` 与 `events`，
可用 `--no-frames` 只留汇总）。视频顶部 HUD 实时显示时间、生命、击杀、漏怪与阻挡数。

### 作为库调用

```python
from rtas.sim import OperatorSpec, Plan, Simulator, load_level

level = load_level("ArknightsGameData/zh_CN/gamedata/levels/obt/main/level_main_01-07.json")

# 只看走位：某一时刻的敌人位置 / 数据
result = Simulator(level).run(duration=90)
result.save_snapshot("t30.png", time=30)
for enemy in result.frame_at(30).enemies:
    print(enemy.name, round(enemy.row, 2), round(enemy.col, 2), enemy.hp)

# 放一个阻挡 1 的近卫，看击杀/漏怪并导出视频
plan = Plan(duration=120, fps=20)
plan.add(OperatorSpec(name="近卫", tile=(3, 2), deploy_time=5, block=1,
                      damage=600, attack_interval=1.0))
result = Simulator(level, plan).run()
print(result.summary)            # 击杀 / 漏怪 / 生命 / 最高同时阻挡 / 清场时间
result.export_video("run.mp4")   # 也可以 .gif
result.save_json("run.json")     # 逐帧数据 + 事件
```

只想快速跑一次时可以用一行式入口：`simulate(level, [op1, op2], duration=120)`。

`SimulationResult` 还提供 `frame_at(t)`、`events_of("kill")`、`to_dict()`、`render(t)`；
`SimLevel` 可被其它工具直接消费（`grid` / `routes` / `spawns` / `enemies`）。

## 平台支持

窗口查找、截图与输入注入都在 `rtas/platform/` 下按平台实现：

- `rtas/platform/base.py` 定义平台无关的 `InputBackend`（查找窗口、点击、按键、截图、
  权限检查）与 `ClickMethod`；
- `rtas/platform/windows/` 是当前的 Windows 实现（Win32 输入 + Windows Graphics Capture）。

`rtas.agent.GameAgent`、`rtas.shell`、OCR 与视觉模块都只依赖这层抽象，
新增 macOS / Linux 后端时实现 `InputBackend` 并在 `get_backend()` 里注册即可，
上层代码无需改动。非 Windows 平台目前会抛出 `UnsupportedPlatform`。

## 首次运行与 OCR 模型

需要 OCR 的命令（`detect`、`monitor`、`gui`、`main-menu`、`squad`、`text`）会在首次运行时
自动下载约 100 MB 的 EasyOCR 模型到 `~/.EasyOCR/model/`（Windows 为
`%USERPROFILE%\.EasyOCR\model\`）。下载与加载期间日志会依次显示
「正在加载 OCR 依赖」和「正在初始化 OCR 引擎」。

OCR 与 PyTorch 采用延迟加载：`rtas --help`、`rtas capture`、`rtas nav`、`rtas data`
等命令不会导入这些重量级依赖，启动时间约 1～2 秒。

如果模型下载被网络中断导致文件损坏（日志出现 `Bad CRC-32`、`MD5 hash mismatch`
或「模型下载失败或文件损坏」），删除模型目录后重跑即可自动重新下载：

```powershell
Remove-Item "$env:USERPROFILE\.EasyOCR\model\*" -Force
uv run rtas detect -i screenshot.png
```

## 测试截图工具

`rtas-capture` 会自动查找《明日方舟》窗口，用 **Windows Graphics Capture**
直接读取窗口自身的画面并保存为 PNG，默认写入项目根目录的 `__temp__/`，
文件名形如 `arknights_20261112_001334.png`。

因为抓取的是窗口合成表面而不是屏幕，所以：

- **不需要把游戏切到前台**，终端挡在游戏前面也能截到干净画面；
- 其它置顶窗口、弹窗、鼠标光标不会混进截图；
- 被遮挡或后台窗口可以直接截取；最小化时会临时恢复窗口再截，
  若系统限制恢复则提示先手动打开。

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
│       ├── agent.py        # 游戏操作代理（截图 → 识别 → 点击）
│       ├── shell.py        # 交互式模式（含 quiet stdio 协议）
│       ├── cli.py          # 命令行实现
│       ├── data/           # 数据模型、提供者与数据库
│       ├── map/            # 关卡与地图分析
│       ├── sim/            # 关卡推演：模型 / 加载 / 引擎 / 渲染 / 子命令
│       │   ├── model.py    # 地块、路线、敌人档案、干员配置、帧快照
│       │   ├── loader.py   # 简化格式 + 原版关卡（含 A* 补路线、敌人数据库）
│       │   ├── engine.py   # 移动 / 阻挡 / 攻击 / 漏怪 的固定步长推演
│       │   ├── render.py   # 底图、逐帧绘制、mp4/gif 导出
│       │   └── commands.py # rtas sim map / run / tiles
│       ├── platform/       # 平台相关的窗口/输入/截图后端
│       │   ├── base.py     # InputBackend 抽象与 ClickMethod
│       │   └── windows/    # Win32 输入、窗口查找、WGC 截图
│       ├── tools/          # 测试辅助工具（WGC 窗口截图）
│       └── vision/         # OCR / OpenCV 视觉分析
│           ├── ocr/        # 可插拔 OCR 后端（easyocr/paddleocr/umi/tesseract）
│           ├── nav_detector.py  # 主界面/子菜单判定
│           └── templates/ui/    # 导航按钮等 UI 模板
└── tests/                  # 许可证、OCR、导航判定、代理/shell、关卡推演测试
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
