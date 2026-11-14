# 第三方许可证与说明

本仓库主体（`radar/` 下的 Python 代码、配置、文档）采用 **AGPL-3.0-or-later**
（与上层 RTAS 工程一致）。下面列出用到的第三方组件，以及各自对本仓库的约束。

> 本文档不是法律意见。分发前请自行复核；尤其是 Spine Runtimes，它对"每个使用者"
> 都有单独要求，见下文第 2 节。

---

## 1. Python 运行时依赖（通过 pyproject 安装，不随仓库分发）

| 组件 | 许可 | 用途 | 说明 |
|---|---|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | **AGPL-3.0** | YOLO 检测/训练 | AGPL 与本仓库一致；**闭源分发或 SaaS 化必须开源**，或购买 Ultralytics 商业许可 |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | 推理/训练后端 | |
| [torchvision](https://github.com/pytorch/vision) | BSD-3-Clause | resnet18 身份编码器 | 预训练权重来自 ImageNet 训练的公开权重 |
| [OpenCV](https://opencv.org/) | Apache-2.0 | 图像处理 | `opencv-python` 轮子自带 OpenCV 许可 |
| [NumPy](https://numpy.org/) | BSD-3-Clause | 数值计算 | |
| [Label Studio](https://labelstud.io/) | Apache-2.0 | 标注 UI | 仅作外部服务运行；`labelstudio` 是可选 extra，不分发 |
| [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) | BSD-2-Clause | 自带 ffmpeg 二进制 | ffmpeg 本体为 LGPL/GPL，见其自带说明 |
| [PyYAML](https://pyyaml.org/) | MIT | 配置读取 | |
| [requests](https://requests.readthedocs.io/) | Apache-2.0 | Label Studio API | |
| [lap](https://github.com/gatagat/lap) | BSD-2-Clause | ByteTrack 匹配 | |

## 2. Spine Runtimes（**仓库内有文件**，务必阅读）

`radar/tools/spine_render/vendor/` 下有两个文件：

- `spine-core.js`
- `spine-canvas.js`

来源：<https://github.com/EsotericSoftware/spine-runtimes>，分支 `3.8`。
许可证全文见同目录 `LICENSE-spine-runtimes.txt`（Spine Runtimes License Agreement，
Copyright (c) 2013-2019, Esoteric Software LLC）。

**关键约束**：该许可要求"每个使用者必须自行获得 Spine Editor 许可"，并且再分发时必须
附带该许可与版权声明（本仓库已附带）。因此：

- 这两个文件**只服务于一个可选的离线工具**（把游戏 Spine 模型渲染成参考图），
  不属于 Python 包，也不参与检测/训练链路；
- 不想承担该许可的话，直接删掉 `radar/tools/spine_render/` 即可，
  其余功能（标注、训练、推理、描述）完全不受影响；
- 该许可与 AGPL-3.0 不兼容（AGPL 禁止附加限制），所以本仓库把它们隔离在
  `tools/spine_render/` 里，并在此明确声明**它们不受本仓库 AGPL 覆盖**。

## 3. 外部工具（只调用、不分发代码）

| 工具 | 许可 | 用途 | 说明 |
|---|---|---|---|
| [ArkUnpacker](https://github.com/isHarryh/Ark-Unpacker) | BSD-3-Clause | 解包游戏 `.ab` 资源 | 以**独立可执行文件**调用（`subprocess`），本仓库不包含其代码；许可全文见 `radar/tools/LICENSE-arkunpacker.txt` |

## 4. 仅作调研、未采用其代码

| 项目 | 许可 | 情况 |
|---|---|---|
| [KokodayoData](https://github.com/KKDYData/KokodayoData) | **GPL-3.0** | 调研阶段看过它的 `src/utils/Spine/skel2Json.js`（明日方舟 Spine 二进制格式的 JS 解析器）。最终改用 **Esoteric Software 官方 Spine 3.8 运行时**，**没有复制、改写或链接它的任何代码**，本仓库不包含其代码，也不构成衍生作品。 |

## 5. 游戏资源

`reference/enemies/*.png`、`frames/`、`raw/` 等来自《明日方舟》（上海鹰角网络）。
这些资源**仅用于本地的模型训练与研究**，不随代码一起分发（已在 `.gitignore` 中排除）。
如需公开分发渲染后的参考图，请自行评估版权风险。

## 6. 数据集与模型权重

- `dataset/`（含人工标注坐标）、`runs/`（训练产物）、`weights/*.pt` 均由本仓库脚本生成，
  默认不入库；
- 若你要发布训练好的权重，注意其中可能间接包含游戏美术的视觉特征，
  以及 ultralytics 的 AGPL 约束（见第 1 节）。
