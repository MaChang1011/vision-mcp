# Vision MCP → Hermes 集成指南

## 快速接入

### 1. 添加 MCP Server 配置

编辑 `~/.hermes/config.yaml`，在 `mcp_servers` 段添加：

```yaml
mcp_servers:
  vision:
    command: /home/machang/vision-mcp-plan/.venv/bin/python
    args:
      - -m
      - vision_mcp
```

### 2. 重启或重载

```bash
# 方式 A: 在 Hermes 对话中输入
/reload-mcp

# 方式 B: 重启 Hermes gateway
hermes gateway restart
```

### 3. 验证接入

在 Hermes 中输入：

```
list available vision tools
```

Hermes 会自动发现以下工具并可用：
- `mcp_vision_vision_ocr_image`
- `mcp_vision_vision_ocr_image_with_boxes`
- `mcp_vision_vision_detect_objects`
- `mcp_vision_vision_preprocess_image`
- `mcp_vision_vision_crop_image`

## MCP 工具详细说明

### vision_ocr_image
提取图片中所有文字。

**参数：**
- `image_path` (必填): 图片绝对路径
- `lang_hint` (可选): 语言提示，`zh`/`en`/`ja`/`ko`/`fr`/`de`，默认 `zh`

**返回：**
```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": {
    "text": "Hello OCR\n123456",
    "confidence": 0.9824
  },
  "meta": {
    "image_width": 640,
    "image_height": 240,
    "lang": "en",
    "line_count": 2,
    "latency_ms": 1340
  }
}
```

### vision_ocr_image_with_boxes
提取文字并返回每行坐标。

**参数：**
- `image_path` (必填)
- `lang_hint` (可选)

**返回：**
```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": {
    "lines": [
      {
        "text": "Hello OCR",
        "confidence": 0.99,
        "bbox": {"x1": 31, "y1": 33, "x2": 330, "y2": 64}
      }
    ]
  }
}
```

### vision_detect_objects
检测图片中的常见物体（COCO 80 类）。

**参数：**
- `image_path` (必填)
- `labels` (可选): 只返回指定类别，如 `["person", "car", "cup"]`
- `confidence_threshold` (可选): 置信度阈值，默认 `0.25`
- `max_detections` (可选): 最大检出数，默认 `100`

**返回：**
```json
{
  "ok": true,
  "engine": "yolo",
  "data": {
    "objects": [
      {
        "label": "person",
        "confidence": 0.87,
        "bbox": {"x1": 49, "y1": 399, "x2": 245, "y2": 903}
      }
    ]
  },
  "meta": {
    "model_name": "yolov8n.pt",
    "object_count": 5,
    "latency_ms": 230
  }
}
```

### vision_preprocess_image
图像预处理：灰度化、去噪、缩放。

**参数：**
- `image_path` (必填)
- `operations` (必填): 操作列表，支持 `grayscale` `denoise` `resize`
- `output_path` (可选): 输出路径，默认自动生成
- `resize_width` (可选): 缩放目标宽度
- `resize_height` (可选): 缩放目标高度
- `denoise_strength` (可选): 去噪强度，默认 `10`

**返回：**
```json
{
  "ok": true,
  "data": {
    "output_path": "/tmp/image_preprocessed.jpg",
    "applied": ["grayscale", "resize"]
  },
  "meta": {
    "image_width": 500,
    "image_height": 300,
    "output_width": 250,
    "output_height": 150
  }
}
```

### vision_crop_image
按坐标裁切图像区域。

**参数：**
- `image_path` (必填)
- `bbox` (必填): `[x1, y1, x2, y2]`
- `output_path` (可选)

**返回：**
```json
{
  "ok": true,
  "data": {
    "output_path": "/tmp/image_cropped.jpg",
    "bbox": [30, 200, 200, 270]
  },
  "meta": {
    "crop_width": 170,
    "crop_height": 70
  }
}
```

## 典型调用示例

### 示例 1: 图片 OCR 后提取关键信息
```
1. ocr_image: /path/to/receipt.jpg, lang_hint=zh
2. 根据返回的 text 内容提取金额、日期等信息
```

### 示例 2: 检测 → 裁切 → OCR 链路
```
1. detect_objects: /path/to/photo.jpg, labels=["person"]
2. crop_image: /path/to/photo.jpg, bbox=<从步骤1获取的bbox>
3. 根据需要继续分析裁切后的图
```

### 示例 3: 预处理提高 OCR 质量
```
1. preprocess_image: /path/to/blurry.jpg, operations=["grayscale","denoise"]
2. ocr_image: /path/to/blurry_preprocessed.jpg
```

## 已知限制与注意事项

### 中文 OCR 兼容性
当前环境的 `lang=ch` 模型存在 SIGILL 兼容问题，已做 subprocess 隔离 + 自动降级。中文场景建议：
- 文案简短的用 `en` 也能识别部分内容
- 纯中文可试试 `zh-hant`（繁体模型）
- 或直接请求 `ocr_image` 时传 `lang_hint=zh`，系统会自动降级

### YOLO 识别范围
当前 `yolov8n.pt` 只能识别 COCO 80 类标准物体（人、车、杯子、鼠标等），无法识别排插、充电器、数据线等细粒度电子配件。这类需求建议等 P1 Grounding DINO 上线后使用 `detect_by_prompt`。

### 路径要求
所有 `image_path` **必须是绝对路径**，因为 Hermes 调用时工作目录可能不同。

## 故障排查

### 工具未出现
```bash
# 确认配置正确
hermes mcp list

# 强制重载 MCP
/reload-mcp

# 检查 server 能正常启动
/home/machang/vision-mcp-plan/.venv/bin/python -m vision_mcp --help
```

### 调用返回错误
```bash
# 确认图片路径存在且有效
python -c "from vision_mcp.core.io import load_image; load_image('/path/to/img.jpg')"

# 单独测试后端
python -c "
from vision_mcp.tools.ocr_tools import ocr_image
print(ocr_image('/path/to/img.jpg'))
"
```
