# Roadmap

## 目标
把全视觉 MCP 从规划文档逐步推进为可运行、可评测、可演进的系统。

---

## 阶段划分

### Phase 0：方案固化
目标：明确架构、边界、数据协议、优先级。
交付：
- architecture.md
- roadmap.md
- tool-spec.md
- implementation-plan.md

完成标准：
- 系统边界清晰
- P0/P1/P2 路线明确
- Tool schema 初稿稳定

---

### Phase 1：最小可用版本（MVP）
目标：建立可运行的本地视觉 MCP 基础能力。
范围：
- PaddleOCR
- YOLO
- OpenCV
- MCP server 初版

能力：
- ocr_image
- ocr_image_with_boxes
- detect_objects
- preprocess_image
- crop_image

完成标准：
- Hermes 可调用
- 统一输出 schema 生效
- 本地单机可运行
- 基础样例可验证

风险控制：
- 不引入 VLM
- 不引入多节点部署
- 不引入复杂异步体系

---

### Phase 2：增强定位与分割
目标：增加更智能的定位与切图能力。
范围：
- Grounding DINO
- SAM / MobileSAM
- 编排层初版

能力：
- detect_by_prompt
- find_region
- segment_by_box
- segment_by_prompt

完成标准：
- 支持按提示找目标
- 支持输出 mask_path
- 支持多模型链路编排

风险控制：
- 先不追求复杂多语言 prompt 优化
- 先做英文桥接+基本中文映射

---

### Phase 3：高层视觉理解
目标：增加图像理解与问答能力。
范围：
- LLaVA 或 Florence-2
- VLM 管理模块

能力：
- describe_image
- answer_about_image
- compare_images
- understand_scene（编排任务）

完成标准：
- 图像描述能力稳定可用
- 基础问答可运行
- 与 OCR / detection 可联合输出

风险控制：
- VLM 不作为基础依赖
- 单独设置资源限制与超时

---

### Phase 4：平台化与工程化
目标：从可用系统升级为长期维护平台。
范围：
- 评测样本集
- 指标采集
- 缓存策略
- 多环境部署
- 任务编排增强

能力：
- analyze_document_image
- extract_ui_text_and_objects
- find_and_extract_object
- screen_understanding

完成标准：
- 有评测基线
- 有指标与日志
- 支持轻量/增强/远程部署

---

## 优先级原则

### P0
必须先完成：
- OCR
- 常见物体检测
- 图像预处理

### P1
高价值增强：
- detect_by_prompt
- segmentation

### P2
高层语义理解：
- describe_image
- answer_about_image

### P3
平台化增强：
- 评测
- 监控
- 缓存
- 多环境部署

---

## 成功标准

### 短期成功
- 有可调用的 MCP 工具
- 文本识别和识物稳定
- 输出格式清晰一致

### 中期成功
- 支持多模型任务链路
- 分割与 prompt 检测可用
- 模型切换成本低

### 长期成功
- 有评测、监控、路线图
- 能支撑更多视觉任务
- 能持续维护与演进
