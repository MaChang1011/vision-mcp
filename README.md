# Vision MCP

面向 Agent 的全视觉能力平台，通过 MCP (Model Context Protocol) 对外提供统一工具接口。

## 当前能力

| 工具 | 功能 | 后端 | 阶段 |
|------|------|------|------|
| `vision_ocr_image` | 提取图片文字 | PaddleOCR | P0 ✅ |
| `vision_ocr_image_with_boxes` | 文字 + 文本框坐标 | PaddleOCR | P0 ✅ |
| `vision_detect_objects` | 通用目标检测 | YOLOv8 | P0 ✅ |
| `vision_preprocess_image` | 灰度化/去噪/缩放 | OpenCV | P0 ✅ |
| `vision_crop_image` | 按 bbox 裁切 | OpenCV | P0 ✅ |
| `vision_detect_by_prompt` | 按描述找目标 | Grounding DINO | P1 ✅ |
| `vision_segment_by_box` | 按 bbox 生成 mask | SAM | P2 ✅ |

## 快速开始

### 1. 安装

```bash
cd vision-mcp-plan
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ocr,opencv,detection,grounding,segmentation,dev]"
```

### 2. 下载模型权重

SAM 需要手动下载一次权重（~375MB）：

```bash
wget -P ~/.cache/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 3. 启动

```bash
bash scripts/start.sh
# 或 source .venv/bin/activate && python -m vision_mcp
```

### 4. 接入 Hermes

`~/.hermes/config.yaml`：

```yaml
mcp_servers:
  vision:
    command: /home/machang/vision-mcp-plan/.venv/bin/python
    args: [-m, vision_mcp]
```

然后 `/reload-mcp`。

## 项目结构

```
src/vision_mcp/
├── server.py                    # MCP Server 入口
├── schema.py                    # 统一数据协议
├── tools/                       # 工具层
│   ├── ocr_tools.py             # OCR
│   ├── detection.py             # 目标检测 (YOLO)
│   ├── grounding.py             # 开放词汇检测 (Grounding DINO)
│   ├── segmentation.py          # 分割 (SAM)
│   └── image_ops.py             # 图像预处理 (OpenCV)
└── backends/                    # 后端实现
    ├── paddleocr_backend.py
    ├── yolo_backend.py
    ├── grounding_backend.py
    ├── sam_backend.py
    └── opencv_backend.py
```

## 演进路线

| 阶段 | 能力 | 状态 |
|------|------|------|
| P0 | OCR / 检测 / 预处理 | ✅ |
| P1 | 按提示找目标 | ✅ |
| P2 | 分割 | ✅ |
| P3 | 图像理解 (LLaVA / Florence-2) | 📋 |
| P4 | 业务工作流 | 📋 |

## 统一返回格式

```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": { "text": "..." },
  "meta": { "latency_ms": 1340 },
  "error": null
}
```

## 测试

```bash
python -m pytest tests/ -v
```
