# Vision MCP

面向 Agent 的全视觉能力平台，通过 MCP (Model Context Protocol) 对外提供统一工具接口。

## P0 当前能力 (已交付)

| 工具 | 功能 | 后端 | 状态 |
|------|------|------|------|
| `vision_ocr_image` | 提取图片文字 | PaddleOCR | ✅ |
| `vision_ocr_image_with_boxes` | 提取文字 + 文本框坐标 | PaddleOCR | ✅ |
| `vision_detect_objects` | 通用目标检测 | YOLOv8 | ✅ |
| `vision_preprocess_image` | 灰度化/去噪/缩放 | OpenCV | ✅ |
| `vision_crop_image` | 按 bbox 裁切 | OpenCV | ✅ |

中文 OCR 已知限制：`lang=ch` 的简体中文模型在当前环境存在 SIGILL 兼容问题，已做 subprocess 隔离 + 自动降级到 `chinese_cht` 兜底，服务不会崩溃。

## 快速开始

### 1. 安装

```bash
cd vision-mcp-plan
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ocr,opencv,detection,dev]"
```

### 2. 启动 MCP Server

```bash
source .venv/bin/activate
python -m vision_mcp
```

或使用脚本：

```bash
bash scripts/start.sh
```

服务通过 stdio 运行，标准 MCP 协议。

### 3. 接入 Hermes

在 `~/.hermes/config.yaml` 中添加：

```yaml
mcp_servers:
  vision:
    command: /home/machang/vision-mcp-plan/.venv/bin/python
    args:
      - -m
      - vision_mcp
```

然后重启 Hermes 或执行 `/reload-mcp`。

接入后可用工具名：
- `mcp_vision_vision_ocr_image`
- `mcp_vision_vision_ocr_image_with_boxes`
- `mcp_vision_vision_detect_objects`
- `mcp_vision_vision_preprocess_image`
- `mcp_vision_vision_crop_image`

### 4. 在 Hermes 中调用示例

```
ocr_image: /tmp/photo.jpg, lang_hint=en
detect_objects: /tmp/photo.jpg, labels=["person","car"]
preprocess_image: /tmp/photo.jpg, operations=["grayscale","resize"], resize_width=800
crop_image: /tmp/photo.jpg, bbox=[100,100,300,300]
```

## 项目结构

```
src/vision_mcp/
├── server.py          # MCP Server 入口
├── schema.py          # 统一数据协议
├── orchestrator.py    # 编排层 (P1)
├── core/
│   └── io.py          # 图像 IO 工具
├── tools/
│   ├── ocr_tools.py   # OCR 工具
│   ├── detection.py   # 目标检测工具
│   └── image_ops.py   # 图像预处理工具
└── backends/
    ├── base.py              # Backend 基类
    ├── paddleocr_backend.py # PaddleOCR 后端
    ├── yolo_backend.py      # YOLO 后端
    └── opencv_backend.py    # OpenCV 后端
```

## 统一返回格式

所有工具返回 JSON 字符串，结构：

```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": { "text": "...", "confidence": 0.98 },
  "meta": { "image_width": 640, "image_height": 480, "latency_ms": 1340 },
  "error": null
}
```

## 测试

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## 演进路线

| 阶段 | 能力 | 后端 | 状态 |
|------|------|------|------|
| P0 | OCR / 检测 / 预处理 | PaddleOCR, YOLO, OpenCV | ✅ 已完成 |
| P1 | 按提示找目标 | Grounding DINO | 🔜 进行中 |
| P2 | 分割 | SAM / MobileSAM | 📋 计划中 |
| P3 | 图像理解 | LLaVA / Florence-2 | 📋 计划中 |
| P4 | 业务工作流 | 编排层 | 📋 计划中 |
