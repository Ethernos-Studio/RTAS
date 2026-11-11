# data - 数据层

## 目录说明

本目录包含推理层的数据访问和管理代码。

## 子目录

### cache/
本地文件缓存:
- 图片缓存
- 模型权重缓存

### database/
数据库管理:
- SQLite/Redis 连接
- 数据模型定义

### models/
领域模型:
- 干员模型
- 敌人模型
- 道具模型
- 关卡模型

### providers/
数据源适配器:
- GitHub 原始数据
- PRTS Wiki 爬取

## 相关目录

- [src/rtas/map/](../map/): 地图分析
- [src/rtas/vision/](../vision/): 视觉分析
