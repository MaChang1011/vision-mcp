# Architecture

## 1. 目标

建设一套可演进的全视觉能力平台，并通过 MCP 对外提供统一工具接口，覆盖：
- OCR
- 图片识别 / 识物
- 通用目标检测
- 按提示定位目标
- 分割
- 图像理解 / 看图问答
- 图像预处理

设计原则：
- 先稳定可用，再逐步增强
- 模型可替换，接口尽量稳定
- 统一输出协议优先于追求单模型全能
- 接入层与核心能力层解耦
- 本地闭环优先，后续支持远程/云 fallback

---

## 2. 总体架构

系统分为四层：

### 2.1 接入层
职责：
- 提供 MCP Server
- 参数校验
- 输入标准化
- 返回统一响应结构
- 后续可扩展 HTTP API / batch API

### 2.2 编排层
职责：
- 根据任务选择后端
- 管理模型调用顺序
- 处理超时、重试、fallback
- 合并多模型结果

典型链路：
- `segment_by_prompt` = Grounding DINO + SAM
- `analyze_document_image` = OpenCV + PaddleOCR + 可选 VLM
- `understand_scene` = YOLO + OCR + VLM

### 2.3 能力层
按能力拆分，而不是按模型拆分：
- OCR
- Detection
- Grounding
- Segmentation
- VLM
- ImageOps

每个能力模块内部支持多个 backend。

### 2.4 基础设施层
职责：
- 模型缓存
- 文件缓存
- 结果缓存
- 配置加载
- 日志与指标
- CPU/GPU 设备管理
- 任务资源限制

---

## 3. 模块边界

### 3.1 OCR 模块
职责：
- 文本检测
- 文本识别
- 文本框输出
- 版面/表格能力的后续扩展

默认 backend：PaddleOCR
备选 backend：Tesseract / Surya

### 3.2 Detection 模块
职责：
- 常见物体检测
- 分类
- 可选实例分割

默认 backend：YOLO

### 3.3 Grounding 模块
职责：
- 开放词汇检测
- 根据自然语言提示找目标

默认 backend：Grounding DINO
说明：中文 prompt 建议先做内部翻译桥接。

### 3.4 Segmentation 模块
职责：
- 基于 box / point / prompt 做 mask 生成

默认 backend：SAM / MobileSAM

### 3.5 VLM 模块
职责：
- 图像描述
- 图像问答
- 高层视觉总结

候选 backend：LLaVA / Florence-2

### 3.6 ImageOps 模块
职责：
- 灰度化
- 去噪
- deskew
- resize / crop
- 画框
- ROI 提取

默认 backend：OpenCV

---

## 4. 数据流

### 4.1 单模型任务
例：`ocr_image`
1. 接入层解析 image_path / image_url / base64
2. ImageOps 做必要预处理
3. OCR backend 执行推理
4. 统一封装输出

### 4.2 多模型任务
例：`segment_by_prompt`
1. 接入层接收 prompt + image
2. Grounding 模块输出 bbox
3. Segmentation 模块根据 bbox 产出 mask
4. 返回 mask_path + bbox + meta

### 4.3 高层理解任务
例：`understand_scene`
1. Detection 输出 objects
2. OCR 输出 visible_text
3. VLM 生成 scene summary
4. 编排层合并结果

---

## 5. 统一数据协议

建议定义核心对象：
- `ImageRef`
- `BBox`
- `MaskRef`
- `OCRLine`
- `DetectedObject`
- `VisionAnswer`
- `ToolResponse`

统一返回结构建议：

```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": {},
  "meta": {
    "latency_ms": 120,
    "image_width": 1280,
    "image_height": 720
  },
  "error": null
}
```

失败结构：

```json
{
  "ok": false,
  "engine": "yolo",
  "data": null,
  "meta": {},
  "error": "model unavailable"
}
```

---

## 6. 非功能性要求

### 6.1 可维护性
- tool 层与 backend 层分离
- backend 可替换
- 配置驱动模型选择

### 6.2 可观测性
- 记录模型加载时间
- 记录请求耗时
- 记录失败率 / 超时率
- 记录命中缓存比例

### 6.3 性能
- 模型 lazy load
- 支持 CPU/GPU 切换
- 大模型任务设置独立超时

### 6.4 安全与资源控制
- 限制输入文件大小
- 控制支持的图片格式
- 路径访问白名单/规范化
- 控制同时推理并发数

---

## 7. 演进路径

### 阶段 1
- PaddleOCR
- YOLO
- OpenCV
- MCP server 初版

### 阶段 2
- Grounding DINO
- SAM / MobileSAM
- Orchestrator 初版

### 阶段 3
- LLaVA / Florence-2
- 评测体系
- 指标监控
- 多环境部署

---

## 8. 风险点

### 风险 1：模型太多，系统复杂度失控
策略：P0 只上 PaddleOCR / YOLO / OpenCV。

### 风险 2：VLM 引入过早，资源和稳定性失控
策略：VLM 放到 P3。

### 风险 3：输出格式混乱
策略：先定义 schema，再接入 backend。

### 风险 4：Prompt grounding 中文效果波动
策略：增加 prompt 规范化与英文桥接层。

---

## 9. 推荐结论

推荐把系统建设为：
- 一个统一的 Vision Core
- 一个薄的 MCP 接入层
- 多 backend 的模块化能力体系

这是长期可维护、可替换、可演进的最稳方案。
