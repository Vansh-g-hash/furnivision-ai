"""
FastAPI backend for AI Furniture Detector.

Provides REST API endpoints for:
- Health checks
- Product link generation
- Image upload and detection
"""

from __future__ import annotations

import base64
import logging
import socket
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    build_amazon_link,
    default_model_path,
    load_model,
    run_detection,
)
from .logging_config import configure_logging, get_logger

logger = get_logger(__name__)


app = FastAPI(
    title="AI Interior Furniture Detection API",
    description="Production-grade REST API for furniture detection in room images using YOLOv8 World.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=3)
def get_model(model_path: str):
    """Load and cache YOLO model (LRU cache with 3 slots)."""
    logger.info(f"Loading model: {model_path}")
    return load_model(Path(model_path).expanduser())


def decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes to BGR numpy array.

    Args:
        file_bytes: Image file bytes.

    Returns:
        Image as numpy array in BGR format.

    Raises:
        ValueError: If image is empty or invalid.
    """
    if not file_bytes:
        logger.error("Empty file uploaded")
        raise ValueError("Uploaded image is empty")

    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("Invalid image file format")
        raise ValueError("Uploaded file is not a valid image")

    logger.debug(f"Decoded image: shape={image.shape}")
    return image


def encode_image_base64(image_bgr: np.ndarray) -> str:
    """
    Encode BGR image to base64 JPEG string.

    Args:
        image_bgr: Image in BGR format.

    Returns:
        Base64-encoded JPEG string.

    Raises:
        ValueError: If encoding fails.
    """
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        logger.error("Failed to encode image to JPEG")
        raise ValueError("Failed to encode annotated image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.get("/")
def root() -> dict[str, object]:
    """API root endpoint - returns service information."""
    return {
        "name": "AI Interior Furniture Detection API",
        "status": "running",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "detect": "/api/detect",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    logger.debug("Health check requested")
    return {"status": "ok"}


@app.get("/api/product-link")
def product_link(name: str) -> dict[str, str]:
    """
    Generate Amazon product search link for given item name.

    Args:
        name: Item name to search for.

    Returns:
        Dictionary with name and Amazon search link.

    Raises:
        HTTPException: If name is empty.
    """
    if not name.strip():
        logger.warning("Product link requested with empty name")
        raise HTTPException(status_code=400, detail="name is required")

    link = build_amazon_link(name.strip())
    logger.debug(f"Generated product link for: {name}")
    return {"name": name.strip(), "link": link}


@app.post("/api/detect")
async def detect_image(
    image: Annotated[UploadFile, File(description="Room/interior image file (JPEG, PNG, etc.)")],
    confidence: Annotated[float, Form()] = DEFAULT_CONFIDENCE,
    iou: Annotated[float, Form()] = DEFAULT_IOU,
    model_path: Annotated[str, Form()] = str(default_model_path()),
    include_image: Annotated[bool, Form()] = True,
) -> JSONResponse:
    """
    Upload image and receive furniture detections with product links.

    Args:
        image: Image file to process.
        confidence: Detection confidence threshold (0-1).
        iou: NMS IoU threshold (0-1).
        model_path: Path to YOLO model weights.
        include_image: Whether to include base64-encoded annotated image in response.

    Returns:
        JSON with detections, item counts, and optional annotated image.

    Raises:
        HTTPException: If processing fails (400 for invalid input, 404 for missing file, 500 for errors).
    """
    try:
        logger.info(f"Detection request: conf={confidence}, iou={iou}, include_image={include_image}")

        contents = await image.read()
        image_bgr = decode_upload_to_bgr(contents)
        model = get_model(model_path)
        result = run_detection(model, image_bgr, confidence=confidence, iou=iou)

        payload: dict[str, object] = {
            "filename": image.filename,
            "total_boxes": len(result.all_items),
            "unique_count": len(result.unique_items),
            "unique_items": [item.to_dict() for item in result.unique_items],
            "all_items": [item.to_dict() for item in result.all_items],
        }

        if include_image:
            payload["annotated_image_base64"] = encode_image_base64(result.annotated_image)
            payload["annotated_image_mime"] = "image/jpeg"

        logger.info(f"Detection complete: {len(result.unique_items)} unique items")
        return JSONResponse(payload)

    except ValueError as exc:
        logger.warning(f"Invalid input: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        logger.error(f"File not found: {exc}")
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Detection failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


def find_available_port(start: int = 8000, attempts: int = 20) -> int:
    """
    Find an available local port.

    Args:
        start: Starting port number.
        attempts: Number of ports to try.

    Returns:
        First available port found.

    Raises:
        OSError: If no ports are available.
    """
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                logger.info(f"Found available port: {port}")
                return port
    raise OSError(f"No available local port found in range {start}-{start + attempts - 1}")


def main() -> None:
    """Start FastAPI server on available port."""
    configure_logging(level="INFO")
    import uvicorn

    port = find_available_port()
    logger.info(f"Starting API server on http://127.0.0.1:{port}")
    print(f"API documentation: http://127.0.0.1:{port}/docs")

    try:
        uvicorn.run(
            "ai_furniture_detector.api:app",
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.exception(f"Server failed: {e}")
        raise


if __name__ == "__main__":
    main()
