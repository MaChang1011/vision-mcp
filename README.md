# 全视觉 MCP 系统方案

## 1. 目标
建设一套可演进的视觉能力平台，并以 MCP 形式对外提供服务，覆盖：
- OCR
- 图片识别 / 识物
- 目标检测
- 分割
- 图像理解 / 看图问答
- 图像预处理

## 2. 系统定位
视觉能力中台 + MCP 接入层。

### 2.1 视觉核心服务层
职责：
- 模型加载
- 推理执行
- 后端路由
- 统一输出
- 缓存
- 日志
- 配置
- 监控

### 2.2 MCP 接入层
职责：
- 对外暴露 tool
- 参数校验
- 权限控制
- 面向 Agent 的语义化接口
- 与 Hermes 对接

## 3. 架构分层
### 3.1 接入层
- MCP Server
- 可选 HTTP API（后期）

### 3.2 编排层
- Vision Orchestrator
- 按任务类型选择模型链路
- 处理 fallback / timeout / retry / 结果合并

### 3.3 能力层
- OCR
- Detection
- Grounding
- Segmentation
- VLM
- Image Ops

### 3.4 基础设施层
- 模型缓存
- 文件缓存
- 结果缓存
- 日志
- 指标
- 配置
- 设备管理（CPU/GPU）

## 4. 推荐底座
- OCR: PaddleOCR
- 通用识物: YOLO
- 按提示找目标: Grounding DINO
- 精细分割: SAM / MobileSAM
- 图像理解: LLaVA 或 Florence-2
- 图像处理胶水: OpenCV

## 5. 模块边界
### OCR 模块
负责：文字检测、识别、文字框、表格/文档结构扩展。

### Detection 模块
负责：目标检测、分类、可选实例分割。

### Grounding 模块
负责：按提示词定位目标、开放词汇检测。

### Segmentation 模块
负责：给 box / prompt / point 生成 mask。

### VLM 模块
负责：图片描述、图片问答、高层语义理解。

### Image Ops 模块
负责：裁剪、缩放、去噪、deskew、ROI 提取、画框。

## 6. 长期目标
### 6.1 统一数据协议
统一定义：ImageRef、BBox、MaskRef、OCRLine、DetectedObject、VisionAnswer。

### 6.2 任务编排能力
支持复合任务：
- analyze_document_image
- find_and_extract_object
- understand_scene

### 6.3 评测能力
建立 OCR、检测、中文截图、文档场景、UI 截图样本集，跟踪准确率、延迟、失败率、资源占用。

### 6.4 多环境部署
- 本地轻量版
- 单机增强版
- 远程高性能版

## 7. 优先级规划
### P0
先做成可用产品：
- ocr_image
- ocr_image_with_boxes
- detect_objects
- preprocess_image
- crop_image

底座：PaddleOCR、YOLO、OpenCV。

### P1
补智能定位能力：
- detect_by_prompt
- find_region

底座：Grounding DINO。

### P2
补精细分割能力：
- segment_by_box
- segment_by_prompt

底座：SAM / MobileSAM。

### P3
补高层图像理解：
- describe_image
- answer_about_image
- compare_images

底座：LLaVA 或 Florence-2。

### P4
补场景化工作流：
- analyze_document_image
- extract_ui_text_and_objects
- find_and_cut_object
- screen_understanding

## 8. 阶段计划
### 阶段 1（2~3 周）
- 单机跑通
- Hermes 可接入
- 输出格式稳定
- 集成 PaddleOCR / YOLO / OpenCV

### 阶段 2（3~6 周）
- 引入 Grounding DINO / SAM
- 编排层初版
- lazy load / timeout / cache

### 阶段 3（6~12 周）
- 引入 LLaVA / Florence-2
- 评测样本
- 指标与监控
- 多环境部署脚本

## 9. 关键技术决策
- 先模块化，不先全能化
- 统一 schema 优先于统一模型
- 优先本地闭环，再考虑云 fallback
- 先做同步工具，再做异步任务

## 10. 推荐路线
### P0 先做
- PaddleOCR
- YOLO
- OpenCV
- 统一 MCP 接口
- 统一输出 schema

### P1 再做
- Grounding DINO

### P2 再做
- SAM

### P3 再做
- LLaVA / Florence-2

### P4 再做
- 业务工作流封装
