"""Vision MCP Server — exposes vision tools via MCP protocol."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from vision_mcp.tools.detection import detect_objects
from vision_mcp.tools.grounding import detect_by_prompt
from vision_mcp.tools.image_ops import crop_image, preprocess_image
from vision_mcp.tools.ocr_tools import ocr_image, ocr_image_with_boxes
from vision_mcp.tools.segmentation import segment_by_box
from vision_mcp.tools.vlm_tools import answer_about_image, describe_image

logger = logging.getLogger("vision_mcp")
mcp = FastMCP("vision-mcp", log_level="INFO")


@mcp.tool()
def vision_ocr_image(image_path: str, lang_hint: str | None = None) -> str:
    """Extract all text from an image using PaddleOCR."""
    result = ocr_image(image_path=image_path, lang_hint=lang_hint)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_ocr_image_with_boxes(image_path: str, lang_hint: str | None = None) -> str:
    """Extract text with bounding boxes from an image using PaddleOCR."""
    result = ocr_image_with_boxes(image_path=image_path, lang_hint=lang_hint)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_preprocess_image(
    image_path: str,
    operations: list[str],
    output_path: str | None = None,
    resize_width: int | None = None,
    resize_height: int | None = None,
    denoise_strength: int = 10,
) -> str:
    """Preprocess an image using OpenCV. ops: grayscale, denoise, resize."""
    result = preprocess_image(
        image_path=image_path, operations=operations, output_path=output_path,
        resize_width=resize_width, resize_height=resize_height, denoise_strength=denoise_strength,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_crop_image(image_path: str, bbox: list[int], output_path: str | None = None) -> str:
    """Crop an image by bbox [x1, y1, x2, y2]."""
    result = crop_image(image_path=image_path, bbox=bbox, output_path=output_path)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_detect_objects(
    image_path: str, labels: list[str] | None = None,
    confidence_threshold: float = 0.25, max_detections: int = 100,
) -> str:
    """Detect common objects (COCO 80 classes) using YOLO."""
    result = detect_objects(image_path=image_path, labels=labels, confidence_threshold=confidence_threshold, max_detections=max_detections)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_detect_by_prompt(
    image_path: str, prompt: str,
    box_threshold: float = 0.35, text_threshold: float = 0.25,
) -> str:
    """Detect objects by natural language description using Grounding DINO."""
    result = detect_by_prompt(image_path=image_path, prompt=prompt, box_threshold=box_threshold, text_threshold=text_threshold)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_segment_by_box(
    image_path: str, bbox: list[int], output_path: str | None = None,
) -> str:
    """Generate a segmentation mask for a bbox region using SAM.

    Returns mask_path and confidence score.
    """
    result = segment_by_box(image_path=image_path, bbox=bbox, output_path=output_path)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_describe_image(
    image_path: str,
    length: str = "normal",
) -> str:
    """Generate a textual description of an image using Moondream VLM.

    Args:
        image_path: Path to the image file.
        length: Detail level — 'short', 'normal' (default), or 'long'.
    """
    result = describe_image(image_path=image_path, length=length)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_answer_about_image(
    image_path: str,
    question: str,
) -> str:
    """Answer a natural-language question about an image using Moondream VLM.

    Args:
        image_path: Path to the image file.
        question: The question to answer (e.g., "What color is the car?").
    """
    result = answer_about_image(image_path=image_path, question=question)
    return json.dumps(result, ensure_ascii=False)


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Vision MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
