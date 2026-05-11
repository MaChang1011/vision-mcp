"""Vision MCP Server — exposes vision tools via MCP protocol."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from vision_mcp.tools.ocr_tools import ocr_image, ocr_image_with_boxes

logger = logging.getLogger("vision_mcp")
mcp = FastMCP("vision-mcp", log_level="INFO")


@mcp.tool()
def vision_ocr_image(image_path: str, lang_hint: str | None = None) -> str:
    """Extract all text from an image using PaddleOCR.

    Args:
        image_path: Absolute path to the image file.
        lang_hint: Language hint (zh, en, ja, ko, fr, de, ...). Defaults to Chinese.

    Returns:
        JSON string with the extracted text and confidence.
    """
    result = ocr_image(image_path=image_path, lang_hint=lang_hint)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def vision_ocr_image_with_boxes(image_path: str, lang_hint: str | None = None) -> str:
    """Extract text with bounding boxes from an image using PaddleOCR.

    Args:
        image_path: Absolute path to the image file.
        lang_hint: Language hint (zh, en, ja, ko, fr, de, ...). Defaults to Chinese.

    Returns:
        JSON string with lines containing text, confidence, and bbox.
    """
    result = ocr_image_with_boxes(image_path=image_path, lang_hint=lang_hint)
    return json.dumps(result, ensure_ascii=False)


def main():
    """Run the MCP server via stdio transport."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Vision MCP Server...")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
