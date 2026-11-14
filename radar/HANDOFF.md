# EYESOFPRIESTESS — 交接简报（给执行 Agent）

> 你接手的是一个游戏画面感知子系统的环境搭建与管线接通任务。
> 本文档是唯一权威依据。标【不可违背】的决策带有充分理由，不要"优化"它们。
> 有疑问先问，不要自行更改设计。

## 1. 背景（三分钟版）

- 目标游戏：明日方舟（塔防，俯视 3D 地图 + 2D 角色序列帧，画面含透视、技能特效）。
- 本期目标（MVP）：**L1 雷达辅助**——实时识别战场敌人/友军的位置、身份、数量，输出场景图 JSON 给下游 RTAS 做 overlay。**不做自动战斗决策**。
- 技术路线：YOLO 检测（粗类）+ ByteTrack 跟踪 + embedding 图库身份匹配。
- 已有教训（前两个项目死于此）：
  1. 纯手工画框标注 → 1000 张计划只完成 100 张 → 项目死亡。所以本期标注是**点选制**，框由 SAM 派生。
  2. 传统 CV（固定 UI mask + 前景提取 + 手工特征）维护成本爆炸。所以感知全部模型化。

## 2. 不可违背的决策清单

1. 【不可违背】感知链全程**纯像素坐标**。YOLO 不知道"格子"的存在；单应/tile 映射归 RTAS（下游工程），本仓库不做、不预留。
2. 【不可违背】**永远不 warp/透视矫正输入图像**。游戏地面与角色不共享投影约束，整帧矫正必撕裂。
3. 【不可违背】标注最小单位是**点**，schema 为 append-only JSONL：
   `{"frame","x","y","cls","type":"point|ignore","reason","op","uid","ts"}`。所有环节消费它。
4. 【不可违背】ignore 是**一等公民**：被遮挡/模糊到无法判定的目标标 `type:"ignore"`（reason=occluded|blurry），保留坐标，**不删帧**。
5. 【不可违背】train/val 划分**按视频切分**，绝不按帧随机（相邻帧几乎相同，按帧切=验证集泄漏）。
6. 【不可违背】Label Studio 的坐标是**百分比**，转换时必须乘 manifest 里的实际分辨率。
7. 【不可违背】检测类别只有 2 个：`enemy, ally`。具体兵种识别走 embedding 图库，**不训分类器**。
   - **玩家干员不标**：部署位置与编成由 RTAS 侧掌握，模型再猜一遍是重复劳动，还会被技能/撤退动画误导。
   - `ally` 指**可自由移动的非玩家友方单位**（友军 NPC、召唤物等），它们的位置是真正未知、需要雷达补的。
   - **血条不进检测类别**：它是"半透明黑底 + 纯色填充"的简单图形，颜色阈值 CV 就能稳定拿到，
     让模型学只会引入噪声（同类条会因受击/技能变色），也不该占用人工标注预算。
   - **场景物（tile）也不进**：可交互物（路障/装置）的位置在游戏的地图数据里本来就有，
     模型学它没有收益，只会把地图装饰误报成装置（1-9/1-10 实测就出现过）。
8. 【不可违背】训练输入 **imgsz=1280**（画面中敌人最小约 40px，640 会漏检）。
9. 【不可违背】身份判别不在遮挡帧做：tracklet 最佳帧裁全身匹配，标签回写轨迹。
10. 【不可违背】评估优先级：自建 GT 帧报数 > 一切 val 指标。

## 3. 附件文件清单（与本简报一起接收）

| 文件 | 作用 | 状态 |
|---|---|---|
| eyesofpriestess_plan.md | 完整方案 v1（架构/管线/里程碑） | 参考 |
| arknights.yaml | YOLO 数据集配置 | 直接可用 |
| train_yolo.sh | 训练/导出/评估命令 | 直接可用 |
| extract_frames.py | 视频→可标注帧（去重限流） | 直接可用 |
| labelstudio_interface.xml | LS 点选界面（KeyPointLabels + IGNORE 标签） | 直接可用 |
| ls_export2clicks.py | LS 导出→clicks JSONL | 直接可用 |
| nightly_sam2yolo.py | clicks→SAM 出框→YOLO 数据集+ignore+crops | 直接可用 |
| predict_track.py | 检测+ByteTrack→场景图 JSONL | 直接可用 |
| offline_copypaste.py | 透视一致 copy-paste 增强 | 直接可用 |
| server.py + annotator.html + tasks/demo.json | 自研点选页（备用，本期不用部署） | 备用 |
| arknights.yaml 中 names 顺序即 cls id | | |

## 4. 环境规格

- Python ≥ 3.10，Windows 11（主）/ Arch Linux（备），有 NVIDIA GPU（RTX 4060, CUDA 12.x）
- pip: `ultralytics>=8.3` `opencv-python` `numpy` `albumentations` `fastapi` `uvicorn` `label-studio`
- SAM 权重：`mobile_sam.pt`（ultralytics 首次运行自动下载）
- TensorRT 为可选项（仅导出阶段需要），环境装不上就导出 ONNX 兜底
- 目录约定：
    raw/                原始录像（用户后续提供）
    frames/             extract_frames.py 输出（帧 + manifest.jsonl）
    clicks/             标注输出（LS 经转换器）
    dataset/            nightly_sam2yolo.py 输出（YOLO 格式）

## 5. 任务清单（按序执行，每步有验收）

- **T1 环境**：创建 venv，装好全部 pip 依赖，import 自检（ultralytics/cv2/fastapi/label_studio）。验收：`python -c "import ultralytics, cv2; print(ultralytics.__version__)"` 通过。
- **T2 LS 部署**：启动 Label Studio，创建项目，labeling interface 粘贴 labelstudio_interface.xml，导入本地存储的 frames 目录（storage 用本地路径，不用 LS Cloud 转存）。验收：界面显示 A/S/D/F/G/H 键位标签。
- **T3 帧目录**：等用户提供 raw/ 录像后跑 extract_frames.py。验收：manifest.jsonl 行数>0，抽查帧无重复画面。
- **T4 转换器**：用 LS 标完的小批量导出 JSON，跑 ls_export2clicks.py。验收：clicks/*.jsonl 生成、坐标是像素值（最大不超过帧宽高）、ignore 记录 reason 正确。
- **T5 nightly**：跑 nightly_sam2yolo.py。验收：dataset/images|labels/{train,val} 均有内容，labels_ignore 有 ignore 备查，report.json 中 class_hist 合理，val 集与 train 集**不来自同一 video**。
- **T6 训练就绪**：数据放好后跑 train_yolo.sh 的首训（可先 50 epoch 冒烟）。验收：runs/arknights/v0/weights/best.pt 生成；能导出 ONNX。
- **T7 GT 评估**：用户提供 2 帧 GT（框标注），写 eval 脚本对比 best 模型在这两帧上的检出（IoU≥0.5 匹配，报漏检/误检数）。验收：输出两帧的 per-class 数字。
- **T8 文档**：把实际跑通的命令和环境版本写进 README（你的继任者只需要 README）。

## 6. 已知坑（血泪换的）

1. LS 图片若经 Cloud 转存，导出 URL 与 manifest 对不上，转换器会大面积 skip——存储必须本地化。
2. nightly 的 SAM 批量推理在大图上较慢，先对战场区域裁剪或降采样跑 SAM，框再映射回原坐标。
3. 录像必须 OBS 原片，严禁 B 站二压视频（水印+压缩率会让小目标消失）。
4. mobile_sam 对极小目标（<30px）可能整个 mask 失败，report.json 里 boxes 数明显少于点数时要报警。
5. YOLO 标签类别 id 必须与 arknights.yaml 的 names 顺序一致。
6. ignore 区域不进训练 labels（YOLO 不支持 ignore），但 labels_ignore/ 必须保留备查。
7. Windows 路径含中文/空格在部分工具链会炸，目录命名全 ASCII。

## 7. 完成定义（DoD）

- T1–T7 全部验收通过
- README 含：环境版本、复现命令、GT 两帧的当前报数
- 用户拿到手只需要：录视频 → 放进 raw/ → 跑两条命令 → LS 标注 → 跑两条命令 → 看报数

—— 简报完。设计决策的来龙去脉都在 plan.md 里，改任何一处前先读它。
