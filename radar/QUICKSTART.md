# 十分钟上手（给要跑起来的人）

只跑推理/标注、不碰游戏资源的话，**不需要** Node、ArkUnpacker、Spine 那一套。

## 0. 你需要

- Windows 10/11，**NVIDIA 显卡**（CPU 也能跑，慢 10 倍）
- [uv](https://docs.astral.sh/uv/)（Python 包管理器，一条命令装好）
- ffmpeg（`uv sync` 会装 `imageio-ffmpeg`，自带一份，不用单独装）

## 1. 装依赖（约 3 分钟，首次会下 torch ~2.5GB）

```powershell
cd E:\RTAS\radar
uv sync --extra train          # 检测/训练
uv sync --extra annotate       # 需要标注时装 Label Studio（可选）
```

自检：

```powershell
uv run radar doctor
```

## 2. 想要现成模型

把 `runs/arknights/v1/weights/best.pt`（2 类：enemy / ally）拷到
`runs/arknights/` 下任意位置，或者直接用自己的训练结果。

权重**不入库**（AGPL + 体积原因），找我要或者自己训。

## 3. 跑推理：视频 → 场景 JSONL → 给 VLM 的图文对

```powershell
# 整段视频
uv run radar predict --src raw/R8-6.mp4 --out out/scene.jsonl

# 只跑第 31 秒起的 8 秒（快速验证）
uv run radar predict --src raw/R8-6.mp4 --start 31 --seconds 8 --out out/scene.jsonl

# 转成"图 + 带坐标的文字"（VLM 的输入）
uv run radar describe --scene out/scene.jsonl --src raw/R8-6.mp4 --out-dir out/vlm
```

产物：

- `out/scene.jsonl`：每帧一行，`{"t":31.0,"units":[{"track_id":1,"cls":"enemy","code":"O1","conf":0.92,"box":[x,y,w,h]}]}`
- `out/vlm/0001.jpg`：**图上画好框和 `#编号`**
- `out/vlm/0001.txt`：

  ```
  t=31.1s：敌人 3（B2 源石虫·α×2、1 士兵×1）；左上 2、中上 1
    #1 B2 源石虫·α(0.94) conf=0.94 center=(980,240) size=138x70 中上
  ```

  `#编号` 在图上和文字里是**同一个**，VLM 不需要做坐标换算。
- `out/vlm/index.jsonl`：`{"t":31.08,"image":"0001.jpg","text":"..."}`

## 4. 想自己标注 + 训练

```powershell
# 1) 抽帧（自动去重、跳黑屏/加载页）
uv run radar extract --src raw/

# 2) 生成任务并起 Label Studio
uv run radar ls-tasks
uv run radar ls-init --email 你的邮箱 --password '你的密码' --project "RTAS model"
uv run radar ls-serve --background            # http://127.0.0.1:8080

# 3) 标完之后回导
uv run radar ls-export --email ... --password ...
uv run radar import-ls --export ls_export.json
uv run radar build                            # 点 -> SAM 框 -> YOLO 数据集
uv run radar train --epochs 50                # imgsz=1280
```

标注规矩在 `labelstudio/labeling_rules.md`，也会同步进 Label Studio 的
Instructions 面板（`uv run radar ls-sync-instruction --email ... --password ...`）。

**只需点三样东西**：敌人本体（选图鉴编号）、非玩家友军（`ally`）、看不清就 ignore。
血条、技能条、路障/装置、地面格子**一律不点**。

## 5. 常见问题

| 现象 | 原因/处理 |
|---|---|
| `ModuleNotFoundError: lap` | `uv sync --extra train` |
| 图片在 Label Studio 里 404 | 必须有 `LocalFilesImportStorage`（`ls-init` 会自动建） |
| 训练好慢 | GTX 1650 约 20 FPS@1280；960 会快一倍，但"两怪合一"会变多 |
| 身份编号经常错 | 已知问题（top-1 约 55%），主要混在 源石虫 / 源石虫·α |
| 想换节点/公网访问 | `start_tunnel.cmd`（frp），配置里的 token 自己换 |

## 6. 那个可选的"从游戏资源渲染参考图"

只有在你想重建 `reference/enemies/*.png` 时才需要：

1. 装 Node（本项目用 fnm，`fnm install 26`）
2. 下载 [ArkUnpacker](https://github.com/isHarryh/Ark-Unpacker) 到游戏 `StreamingAssets/AB/`
3. `cd tools/spine_render && npm install`
4. `uv run python tools/render_reference.py`

许可证说明见 [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md)（Spine Runtimes 有单独要求）。
