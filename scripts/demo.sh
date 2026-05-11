#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
source .venv/bin/activate

DEMO_DIR="tmp/demo"
mkdir -p "$DEMO_DIR"

python - "$DEMO_DIR" << 'PYEOF'
import json, os, sys
import cv2
import numpy as np

out = sys.argv[1]

# 1. Create test image
img = np.full((400, 600, 3), 245, dtype=np.uint8)
cv2.rectangle(img, (10, 10), (590, 390), (60, 60, 60), thickness=2)
cv2.putText(img, "Vision MCP Demo", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 120), 2)
cv2.putText(img, "Line 2: 12345", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 30, 30), 2)
cv2.putText(img, "Line 3: ABC xyz", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 120, 30), 2)
cv2.circle(img, (500, 120), 60, (200, 160, 0), -1)
cv2.rectangle(img, (30, 220), (200, 350), (0, 120, 200), -1)
cv2.putText(img, "ROI", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite(f"{out}/demo.png", img)
print(f"Created {out}/demo.png")

# 2. OCR
os.chdir(os.path.dirname(out) + "/..")
from vision_mcp.tools.ocr_tools import ocr_image, ocr_image_with_boxes
print("\n=== ocr_image (en) ===")
r = ocr_image(f"{out}/demo.png", lang_hint="en")
print(json.dumps(r, indent=2, ensure_ascii=False))

print("\n=== ocr_image_with_boxes (en) ===")
r = ocr_image_with_boxes(f"{out}/demo.png", lang_hint="en")
print(json.dumps(r, indent=2, ensure_ascii=False))

# 3. Preprocess
from vision_mcp.tools.image_ops import preprocess_image, crop_image
print("\n=== preprocess_image ===")
r = preprocess_image(f"{out}/demo.png", operations=["grayscale", "resize"], resize_width=300)
print(json.dumps(r, indent=2, ensure_ascii=False))

# 4. Crop + OCR chain
print("\n=== crop → ocr chain ===")
r = crop_image(f"{out}/demo.png", bbox=[30, 220, 200, 350])
print("crop:", json.dumps(r, indent=2, ensure_ascii=False))
if r["ok"]:
    r2 = ocr_image(r["data"]["output_path"], lang_hint="en")
    print("ocr on crop:", json.dumps(r2, indent=2, ensure_ascii=False))

# 5. YOLO detection
from vision_mcp.tools.detection import detect_objects
print("\n=== detect_objects ===")
r = detect_objects(f"{out}/demo.png", confidence_threshold=0.10)
print(json.dumps(r, indent=2, ensure_ascii=False))
PYEOF

echo ""
echo "=== Demo complete ==="
echo "Output files in: $DEMO_DIR"
ls -la "$DEMO_DIR/"
