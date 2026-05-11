"""Vision MCP Server — exposes vision tools via MCP protocol."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from vision_mcp.tools.image_ops import crop_image, preprocess_image
from vision_mcp.tools.ocr_tools import ocr_image, ocr_image_with_boxes

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
    """Preprocess an image using OpenCV.

    Supported operations in current MVP:
    - grayscale
    - denoise
    - resize
    """
    result = preprocess_image(
        image_path=image_path,
        operations=operations,
        output_path=output_path,
        resize_width=resize_width,
        resize_height=resize_height,
        denoise_strength=denoise_strength,
    )
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_crop_image(
    image_path: str,
    bbox: list[int],
    output_path: str | None = None,
) -> str:
    """Crop an image by bbox [x1, y1, x2, y2] using OpenCV."""
    result = crop_image(image_path=image_path, bbox=bbox, output_path=output_path)
    return json.dumps(result, ensure_ascii=False)


def main():
    """Run the MCP server via stdio transport."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Vision MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
