# EYESOFPRIESTESS · 数据与模型管线

> 项目原名「方舟雷达」，因实体周边已占用该名字，2026-11-14 更名为 **EYESOFPRIESTESS**。
> 给 human-v1.0 装雷达：识别战场上的敌人/友军位置、身份、数量，输出场景图 JSON 给 RTAS。
> 本期只做 **L1 雷达辅助**，不做自动战斗决策。设计决策见 [HANDOFF.md](HANDOFF.md) 与
> [eyesofpriestess_plan.md](eyesofpriestess_plan.md)，**改任何一处前先读它们**。

命令仍然叫 `radar`（也在 `pyproject.toml` 里注册了 `eop` 别名），Python 包名是 `eyesofpriestess`。

这个目录是一个独立子工程（有自己的 `pyproject.toml` / venv），不依赖 RTAS 的代码；
RTAS 只消费它的 `scene.jsonl`。

## 它替你做了什么

```
raw/*.mp4 ──radar extract──> frames/ + manifest.jsonl
                                   │
                      radar ls-tasks / ls-interface
                                   │
                        Label Studio 点选（A/S/D/F…）
                                   │ 导出 JSON
                       radar import-ls ──> clicks/*.jsonl
                                   │
                    radar build（点 → SAM 出框 → YOLO 数据集）
                                   │
                    radar train → radar export / radar val
```

一条命令都不用手打参数，`radar --help` 都有说明。

## 环境

Windows 11 + NVIDIA GPU 实测（本机 GTX 1650 4 GB / CUDA 12.6，8 GB 以上更稳）。

```powershell
cd radar
uv sync --extra train          # ultralytics + CUDA 版 torch（约 3 GB，首次）
uv run radar doctor            # 自检：cv2 / ultralytics / torch+CUDA / 数据状态
```

标注平台单独装（体积大，且和训练环境解耦更省心）：

```powershell
uv tool install label-studio   # 或 pip install label-studio
```

**权重离线放置**：`mobile_sam.pt`（SAM 出框）和 `yolov8s.pt`（预训练）默认从 GitHub 拉，
国内经常连不上。手动下好丢进 `weights/` 即可，`radar build` / `radar train` 会自动优先用它们：

```powershell
# 走本机代理下载（把 7897 换成你的端口）
Invoke-WebRequest -Uri 'https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt' `
  -OutFile weights\mobile_sam.pt -Proxy http://127.0.0.1:7897 -UseBasicParsing
```

## 用法：从视频到可训练数据集

### 1) 抽帧

把录像丢进 `raw/`（OBS 原片，禁止 B 站二压；分辨率锁死不要混），然后：

```powershell
uv run radar extract                                 # 默认读 raw/，输出 frames/ + manifest.jsonl
uv run radar extract --max-frames 400 --sample-fps 2 # 要更快：候选帧率降到 2fps
uv run radar extract --jobs 4                        # 视频多时并行（默认最多 4 线程）
```

- 解码走 **ffmpeg 管道**（`imageio-ffmpeg` 自带静态 exe，`uv sync` 即装，无需管理员）：
  先用 `fps` 滤镜把候选帧降到 `--sample-fps`（默认 3），Python 只处理这 1/N 的帧；
- 原分辨率 JPEG(95)，**不缩放**（敌人最小约 40px，缩了就没了）；
- 感知哈希 + 帧差去重 + `--min-gap` 限流；黑屏（`--min-brightness`）和
  加载页/编队页（`--min-edges`，边缘密度低）自动跳过；
- `--max-frames` 是「这个视频总共要多少帧」：重跑同一条命令是幂等的，调大才会继续抽。

实测（本机 1080p 素材，GTX 1650）：单条 100s@30fps 约 22s（旧 cv2 逐帧 ≥43s，
单解码就要 43s）；三条共 348s 约 43s。`--hwaccel` 走 NVDEC 在这张卡上反而更慢
（25ms/帧 vs 8ms），默认不开。

### 2) 生成标注界面与任务

```powershell
uv run radar ls-interface     # configs/enemies.json -> labelstudio/interface.xml
uv run radar ls-tasks         # frames/ -> tasks/ls_tasks.json
```

**标签用游戏图鉴里的编号**（`B1` 源石虫、`B2` 源石虫·α、`A3`、`D1`、`19`…），
不用手写兵种名：跨语言、跨版本稳定，图鉴里一眼就能抄。编号表在
[`configs/enemies.json`](configs/enemies.json)，加一个敌人就是加一行，再跑一次
`ls-interface`。编号最终都落到 **3 个 YOLO 粗类**：

| 粗类 | 标什么 |
|---|---|
| `enemy` | 敌人（图鉴编号各自成标签，最终并到这一类） |
| `ally` | **可自由移动的非玩家友方单位**（友军 NPC、召唤物） |
| `tile` | 离散、不动的非敌非友战场物体（路障/装置/箱子）；**地面格子不标** |

**玩家干员不标**：部署位置和编成 RTAS 侧本来就知道，模型再猜一遍是重复劳动，
还会被技能/撤退动画误导。兵种身份也不训分类器，留给 embedding 图库。

**血条 / 技能条也不标**：它们是"半透明黑底 + 纯色填充"的规则图形，用颜色阈值 CV 直接拿
（[`src/eyesofpriestess/cv/bars.py`](src/eyesofpriestess/cv/bars.py)，`find_bars()`）。
让模型学只会被"干员开技能血条变红 / 技能条绿→橙"这类同色不同义的情况带偏，
标注成本还要翻倍。

### 3) Label Studio 标注

LS 必须用**本地文件存储**（走 Cloud 转存会让导出 URL 和 manifest 对不上，转换器会大面积 skip）。
`ls-serve` 已经把环境变量配好了：

```powershell
uv run radar ls-serve --background   # 后台起，等它就绪后打印地址（日志 ls_serve.log）
# 浏览器打开 http://127.0.0.1:8080
```

第一次进 LS 要注册一个本地账号（邮箱随便填，密码记住；数据存在 `radar/ls_data/`）。
然后：

1. **Create Project**，名字随意（如 `eyes-of-priestess`）；
2. 进项目 → 右上 **Settings → Labeling Interface → Code**，
   把 `labelstudio/interface.xml` 的内容整个粘进去 → **Save**。
   切到别的标签页应该能看到 18 个标签（编号 A/S/D/F… + ally/tile/血条/ignore）；
3. 回项目 → **Import** → 选 `tasks/ls_tasks.json` → 确认，203 个任务进列表；
4. **标注**：点开一个任务，
   - 先按热键选标签（`a`=B1 源石虫、`s`=B2、`d`=1、`f`=02、`g`=A3…），**再点目标**；
   - 一个目标只点一下，别画框（框是 SAM 的事）；
   - 遮挡/糊到看不清的，按 `z`（遮挡）或 `x`（模糊）点一下，**不要删帧**；
   - `Ctrl+Enter`/`Submit` 提交，`Skip` 跳过整帧（比如没战场的画面）；
   - 右侧标签面板可以点 × 删掉点错的点；
5. 标完一批 → 项目页 **Export** → 格式选 **JSON** → 下载，存成 `radar/ls_export.json`；
6. 转成 clicks：

```powershell
uv run radar import-ls --export ls_export.json
```

LS 的坐标是百分比，这里按 manifest 里的真实分辨率乘回像素，写 `clicks/<帧>.jsonl`。
转换完会报「转换 N 点 / 跳过 M 任务」，跳过数不为 0 就是存储配置或帧名对不上。

> 键位表随时可以看：`uv run radar ls-interface` 会打印「键位 / 标签 / 类别 / 中文名」。

### 4) 导出转 clicks

```powershell
uv run radar import-ls --export ls_export.json
```

LS 的坐标是百分比，这里按 manifest 里的真实分辨率乘回像素，写 `clicks/<帧>.jsonl`。
转换完会报「转换 N 点 / 跳过 M 任务」，跳过数不为 0 就是存储配置或帧名对不上。

### 5) 点 → 框 → YOLO 数据集

```powershell
uv run radar build                       # 默认 clicks/ + frames/ -> dataset/
uv run radar build --region 0.05,0.15,0.98,0.95 --downscale 0.6   # 大图提速
```

产出：

| 路径 | 内容 |
|---|---|
| `dataset/images/{train,val}` | 帧副本（**按视频划分**，同一视频不会横跨 train/val） |
| `dataset/labels/{train,val}` | YOLO 标签 `cls cx cy w h`（ignore 不进来） |
| `dataset/labels_ignore/` | ignore 点备查（`99 cx cy w h`），永远保留 |
| `dataset/crops/<编号>/` | 身份图库素材，按图鉴编号分目录 |
| `dataset/data.yaml` | 训练配置（绝对路径，换 cwd 不影响） |
| `dataset/report.json` | 每帧统计 / 类别直方图 / 出框率 / ignore 率 / 报警 |

`report.json` 里 `box_ratio` 低于 80% 会报警：说明 SAM 没框住点（目标太小或被特效盖住），
先确认点是否点在目标上，再考虑 `--downscale` 或换 SAM 权重。

### 6) 训练 / 导出 / 验证 / 预测

```powershell
uv run radar train --epochs 50            # 冒烟：imgsz=1280 freeze=10 mosaic=1.0
uv run radar val                          # val 上的 mAP
uv run radar export --fmt onnx            # TensorRT 用 --fmt engine --half
uv run radar predict --src raw/1-7.mp4    # 检测 + ByteTrack -> scene.jsonl
```

### 6.1) 框的质量：三个必须一起用的设置

输出要给"读不了坐标的 VLM"用，框错了后面全错，所以这三条是默认值、别随手改：

| 设置 | 值 | 为什么 |
|---|---|---|
| `--imgsz` | **1280** | 960 时"两怪合一"的合并框明显更多（val：合并框 5 个/吞 11 个 GT，1280 只有 3 个/6 个） |
| `--iou`（NMS） | **0.8** | 默认 0.45~0.7 会把挨着走的两个敌人压成一个 |
| 合并框抑制 | **自动开** | 一个大框套住 ≥2 个更小的框 → 丢大框（`postprocess.MergeSuppressor`） |

实测（val 17 帧 / 56 个敌人）：

```
imgsz= 960         : 合并框 5 个，吞掉 11/56 个 GT，漏检 10
imgsz= 960 +合并抑制: 合并框 3 个，吞掉  7/56 个 GT，漏检 10
imgsz=1280         : 合并框 3 个，吞掉  6/56 个 GT，漏检  5   ← 默认
imgsz=1280 +合并抑制: 合并框 3 个，吞掉  6/56 个 GT，漏检  5
```

代价是速度：1280 在 GTX 1650 上约 20 FPS（960 是 34 FPS）。**框正确 > 帧率**，所以默认 1280。

`train` 固定了 HANDOFF §2.8 的三件套：`imgsz=1280`、`freeze=10`、mosaic 拉满（后 20 轮关）。

`scene.jsonl` 一行一帧：

```json
{"t": 3.2, "frame_idx": 96, "units": [{"track_id": 12, "cls": "enemy", "conf": 0.71, "box": [812, 430, 46, 52]}]}
```

坐标是纯截图像素，单应 / tile 映射归 RTAS，这里不碰。

## 标注规矩（摘要）

完整版在 [`labelstudio/labeling_rules.md`](labelstudio/labeling_rules.md)，也会同步进 LS 的
Instructions 面板（`uv run radar ls-sync-instruction --email ... --password ...`）。
注意 LS 的 Instructions 面板按 **HTML** 渲染，不认 markdown：推送时会把 `#`/`-`/`**` 自动转成
`<h2>/<ul>/<b>`，所以仓库里存 markdown、LS 里显示正常排版。

- **IGNORE-occluded**：可见部分 < 2/3、中心点被盖、或两个目标叠到分不出边界；
- **IGNORE-blurry**：认不出兵种，或 < 20px 且边缘发糊；
- **受击变红/闪白、小目标、边缘被裁一角** → 照常点编号，**不** ignore；
- **`tile`** = 离散不动的非敌非友物体（路障/装置/箱子）；**地面格子不标**，装置血条也不单独标；
- 编号不认识 → `other_enemy`，别 ignore；
- 一帧 ignore 超过 1/3 就直接 Skip 整帧。

## 目录

```
radar/
├── configs/enemies.json     # 图鉴编号 -> YOLO 粗类（标签的唯一来源）
├── configs/arknights.yaml   # 数据集配置参考（build 会生成 dataset/data.yaml）
├── labelstudio/interface.xml# 由 ls-interface 生成，别手改
├── src/eyesofpriestess/
│   ├── schema.py            # 点选 schema（字段锁定，append-only JSONL）
│   ├── frames.py            # 抽帧 + manifest
│   ├── catalog.py           # 编号目录 + LS 界面生成
│   ├── ls.py                # LS 任务 / 导出 -> clicks
│   ├── sam.py               # 点 -> 掩码 -> 框（支持 region / 降采样）
│   ├── dataset.py           # clicks -> YOLO 数据集 + report
│   ├── train.py             # 训练 / 导出 / 验证
│   ├── predict.py           # 检测 + ByteTrack -> scene.jsonl
│   └── cli.py               # `radar` 子命令
├── tests/                   # 24 个用例，不依赖 ultralytics（假 SAM/假视频）
├── raw/ frames/ clicks/ dataset/ runs/ tasks/   # 数据与产物，全在 .gitignore
└── pyproject.toml
```

## 自检

```powershell
uv run radar doctor        # 依赖 / CUDA / 帧数 / 点数 / 数据集 / 权重
uv sync --extra train      # 环境不对就重来
uv run pytest -q           # 25 passed
```

## 本机已验证到哪一步

下面这些是真跑过的（2026-11-13，1-7 实战录像 1920x1080@30fps）：

| 步骤 | 命令 | 结果 |
|---|---|---|
| 抽帧 | `radar extract` | 3 条录像（348s）→ 203 帧，43s；跳过黑屏 58 帧、加载/编队页 15 帧 |
| 任务 | `radar ls-tasks` | 50 个任务，`d=frames/...` 本地存储 URL |
| 出框 | `radar build`（真 SAM） | 3 点 3 框，`box_ratio=1.0`，框贴合目标 |
| 训练 | `radar train --epochs 1 --imgsz 640` | `runs/arknights/smoke/weights/best.pt` |
| 导出 | `radar export --fmt onnx` | `best.onnx` |
| 验证 | `radar val` | 5 类 names 正确 |

还没做的：Label Studio 实标（要你先装 LS）、`predict` 出场景图（要有训练好的权重）。
`HANDOFF.md` 的 T2/T4 验收就等你的第一批标注。

## 已知坑（血泪换的）

1. **录制黑屏**：如果视频里出现整段纯黑，多半是录制时抓的是被遮挡/独占全屏的窗口。
   `extract` 会跳过黑帧并打印「跳过黑屏 N 帧」，N 很大就说明录像废了，重录。
2. **LS 用了 Cloud 转存** → 导出 URL 与 manifest 对不上 → 转换器 skip。必须本地存储 + `DOCUMENT_ROOT`。
3. **只有一条视频**：val 只能按时间切，`report.json` 会报警。补录第二条视频后重建，
   这才是唯一正确的 val（相邻帧几乎相同，按帧随机 = 泄漏）。
4. **极小目标**：mobile_sam 对 <30px 的目标经常整片 mask 失败，表现为 `box_ratio` 掉。
   先按 `--region` 只跑战场、`--downscale` 提速，再不行换 sam2 权重。
5. **类别 id 顺序**：`dataset/data.yaml` 的 `names` 由 `configs/enemies.json` 的 `classes` 生成，
   顺序是 enemy/ally/tile/hp_bar_red/hp_bar_ally，别手改。
6. **ignore 不删帧**：YOLO 没有 ignore 语义，但 `labels_ignore/` 必须留着——它是"此处有不可判定目标"
   的事实记录，删了会让数据集对密集怪群产生系统性盲区。
7. **LS 1.23 起图片 404**：光设 `LOCAL_FILES_SERVING_ENABLED` 不够，
   `/data/local-files/?d=...` 还会查项目里有没有 **Local files 存储连接**
   （Settings → Cloud Storage → Local files，path 指到 `frames/`）。没有这条记录就一律 404，
   前端表现为 `There was an issue loading URL from $image value`。
   `radar ls-init` 会自动挂好这条连接（幂等，已存在就跳过）。
