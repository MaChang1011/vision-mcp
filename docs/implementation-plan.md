# Implementation Plan

## 1. 执行顺序
按 A → C → B 顺序推进：
- A：先完善设计文档
- C：再补实施计划与任务拆解
- B：最后创建代码骨架

这样做的目的：
- 先固化边界，避免代码骨架返工
- 先明确阶段目标和优先级，保证后续开发有里程碑
- 最后再落代码结构，保证目录和模块命名更稳定

---

## 2. 当前阶段状态

### A：详细设计文档
状态：已完成第一版
已完成文档：
- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/tool-spec.md`

验收标准：
- 架构分层清晰
- 模块边界明确
- 工具列表明确
- P0/P1/P2 范围明确

后续补充项：
- 增加数据结构定义文档
- 增加错误码/异常约定
- 增加配置项说明

---

## 3. C：实施计划与任务拆解
目标：把方案变成可执行任务，而不是停留在设计层。

### C1. 里程碑

#### M1：项目初始化
目标：建好代码仓结构与基础工程约束
输出：
- src/tests/scripts 目录
- requirements / pyproject
- .gitignore
- 基础 README

#### M2：P0 能力打通
目标：完成 OCR + detection + image ops 的本地闭环
输出：
- PaddleOCR backend
- YOLO backend
- OpenCV backend
- MCP tools：ocr_image / ocr_image_with_boxes / detect_objects / preprocess_image / crop_image

#### M3：统一协议与编排初版
目标：避免 tool 返回结构发散
输出：
- schema 定义
- response wrapper
- backend base interface
- orchestrator 初版

#### M4：P1 能力增强
目标：增加 detect_by_prompt / segmentation
输出：
- Grounding DINO backend
- SAM backend
- tools: detect_by_prompt / segment_by_box / segment_by_prompt

#### M5：P2 视觉理解
目标：加入高层图像理解
输出：
- VLM backend
- tools: describe_image / answer_about_image

#### M6：工程化增强
目标：提升长期维护能力
输出：
- 评测样本
- 指标
- 缓存
- 配置文档
- 部署脚本

---

### C2. 任务拆解

#### 阶段 M1 任务
- 确定 Python 版本
- 确定依赖管理方式
- 创建 src 目录结构
- 建立 tests 骨架
- 建立基础配置文件

#### 阶段 M2 任务
- 封装 OCR backend 接口
- 接入 PaddleOCR
- 封装 detection backend 接口
- 接入 YOLO
- 封装 image ops backend 接口
- 接入 OpenCV
- 实现 P0 tools
- 增加基础测试样例

#### 阶段 M3 任务
- 定义 schema
- 定义统一错误格式
- 增加 meta 字段规范
- 加入 orchestrator 初版
- 抽象 backend base class

#### 阶段 M4 任务
- 接入 Grounding DINO
- 接入 SAM / MobileSAM
- 实现 detect_by_prompt
- 实现 segment_by_box
- 实现 segment_by_prompt
- 增加组合任务测试

#### 阶段 M5 任务
- 选型 LLaVA 或 Florence-2
- 实现 describe_image
- 实现 answer_about_image
- 加入资源限制和超时控制

#### 阶段 M6 任务
- 设计评测样本目录
- 增加 benchmark 脚本
- 增加日志/指标输出
- 增加本地/增强版部署说明

---

## 4. 优先级

### P0（必须先完成）
- 项目骨架
- OCR backend
- detection backend
- image ops backend
- MCP server 初版
- 统一 schema 初版

### P1（高价值增强）
- Grounding DINO
- SAM / MobileSAM
- 多模型编排

### P2（可延后）
- VLM
- 高层视觉理解

### P3（平台化）
- 评测
- 监控
- 缓存
- 多环境部署

---

## 5. 风险与控制

### 风险 1：骨架先建错，后续返工
控制：A 完成后再做 B。

### 风险 2：没有任务拆解，开发会发散
控制：C 阶段先明确里程碑和验收标准。

### 风险 3：P0 范围失控
控制：P0 只做 PaddleOCR / YOLO / OpenCV，不提前上 VLM。

---

## 6. 下一步
按 A → C → B 顺序，当前 A 已完成第一版，C 已完成第一版。
下一步进入 B：创建 Python 项目骨架。
