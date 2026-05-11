# Tool Specification

## 设计原则
- tool 名称稳定，backend 可替换
- 参数尽量简单，复杂逻辑放编排层
- 输出统一使用 ToolResponse 包装
- 路径、URL、base64 三种输入后续兼容，MVP 先以 image_path 为主

---

## 1. ocr_image
职责：提取图片中的完整文字。

输入：
- image_path: string
- lang_hint?: string

输出 data：
- text: string
- confidence?: number

默认 backend：PaddleOCR

---

## 2. ocr_image_with_boxes
职责：提取文字并返回文本框。

输入：
- image_path: string
- lang_hint?: string

输出 data：
- lines: OCRLine[]

OCRLine:
- text: string
- confidence: number
- bbox: [x1, y1, x2, y2]

默认 backend：PaddleOCR

---

## 3. detect_objects
职责：识别图片中的常见目标。

输入：
- image_path: string
- labels?: string[]

输出 data：
- objects: DetectedObject[]

DetectedObject:
- label: string
- confidence: number
- bbox: [x1, y1, x2, y2]

默认 backend：YOLO

---

## 4. preprocess_image
职责：做图像预处理，提高后续 OCR / detection 效果。

输入：
- image_path: string
- operations: string[]

支持操作（MVP）：
- grayscale
- denoise
- deskew
- resize

输出 data：
- output_path: string
- applied: string[]

默认 backend：OpenCV

---

## 5. crop_image
职责：按 bbox 裁切图像区域。

输入：
- image_path: string
- bbox: [x1, y1, x2, y2]

输出 data：
- output_path: string
- bbox: [x1, y1, x2, y2]

默认 backend：OpenCV

---

## 6. detect_by_prompt
职责：根据自然语言提示定位目标。

输入：
- image_path: string
- prompt: string

输出 data：
- matches: DetectedObject[]

默认 backend：Grounding DINO

备注：建议内部支持中文 prompt 规范化和英文桥接。

---

## 7. segment_by_box
职责：根据 bbox 生成 mask。

输入：
- image_path: string
- bbox: [x1, y1, x2, y2]

输出 data：
- mask_path: string
- bbox: [x1, y1, x2, y2]

默认 backend：SAM / MobileSAM

---

## 8. segment_by_prompt
职责：根据 prompt 先定位，再分割。

输入：
- image_path: string
- prompt: string

输出 data：
- segments: array
  - label: string
  - bbox: [x1, y1, x2, y2]
  - mask_path: string

默认链路：Grounding DINO + SAM

---

## 9. describe_image
职责：生成图片内容摘要。

输入：
- image_path: string
- detail_level?: string

输出 data：
- summary: string

候选 backend：LLaVA / Florence-2

---

## 10. answer_about_image
职责：针对图片做问答。

输入：
- image_path: string
- question: string

输出 data：
- answer: string

候选 backend：LLaVA / Florence-2

---

## 统一响应结构

```json
{
  "ok": true,
  "engine": "paddleocr",
  "data": {},
  "meta": {
    "latency_ms": 123,
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
  "error": "invalid image path"
}
```

---

## MVP 范围
P0 工具：
- ocr_image
- ocr_image_with_boxes
- detect_objects
- preprocess_image
- crop_image

P1 工具：
- detect_by_prompt
- segment_by_box
- segment_by_prompt

P2 工具：
- describe_image
- answer_about_image
