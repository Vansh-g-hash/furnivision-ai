from __future__ import annotations

import base64
import csv
import io
import json
import os
import socket
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

try:
    from .detector import (
        DEFAULT_CONFIDENCE,
        DEFAULT_IOU,
        build_amazon_link,
        category_for_item,
        default_model_path,
        display_name,
        load_model,
        run_detection,
    )
except ImportError:  # Allows: python web_app.py
    from detector import (  # type: ignore
        DEFAULT_CONFIDENCE,
        DEFAULT_IOU,
        build_amazon_link,
        category_for_item,
        default_model_path,
        display_name,
        load_model,
        run_detection,
    )

APP_TITLE = "FurniVision AI — Furniture Detection Studio"
MAX_UPLOAD_MB = 15
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

app = FastAPI(title=APP_TITLE, version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_industry_headers(request, call_next):
    """Add lightweight production-friendly headers for browser safety and tracing."""
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@lru_cache(maxsize=3)
def cached_model(model_path: str):
    return load_model(Path(model_path).expanduser())


def decode_upload_to_bgr(file_bytes: bytes) -> np.ndarray:
    if not file_bytes:
        raise ValueError("Uploaded image is empty")

    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            "Uploaded file is not a valid image. Use JPG, PNG, WEBP, or JPEG."
        )

    return image


def encode_jpeg_data_url(image_bgr: np.ndarray, quality: int = 94) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )

    if not ok:
        raise ValueError("Failed to encode image")

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def safe_bbox(raw_bbox: Any) -> list[int]:
    try:
        values = list(raw_bbox or [0, 0, 0, 0])[:4]

        while len(values) < 4:
            values.append(0)

        return [int(round(float(value))) for value in values]
    except Exception:
        return [0, 0, 0, 0]


def clean_item_dict(item: Any, index: int) -> dict[str, Any]:
    data = item.to_dict() if hasattr(item, "to_dict") else dict(item)

    raw_name = str(data.get("name", "unknown")).strip().lower()
    pretty_name = display_name(raw_name)
    category = str(
        data.get("category") or category_for_item(raw_name) or "Other"
    )
    link = str(data.get("link") or build_amazon_link(raw_name))
    confidence = float(data.get("confidence", 0) or 0)
    count = int(data.get("count", 1) or 1)
    bbox = safe_bbox(
        data.get("bbox")
        or data.get("coords")
        or [0, 0, 0, 0]
    )

    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)

    return {
        "id": f"det-{index}",
        "name": raw_name,
        "display_name": pretty_name,
        "category": category,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "count": count,
        "bbox": bbox,
        "area": area,
        "link": link,
    }


def build_summary(
    items: list[dict[str, Any]],
    all_items: list[dict[str, Any]],
) -> dict[str, Any]:
    category_counts: dict[str, int] = {}
    link_count = 0

    for item in items:
        category = item["category"]
        count = int(item.get("count", 1))

        category_counts[category] = (
            category_counts.get(category, 0) + count
        )

        if item.get("link"):
            link_count += 1

    top_category = (
        max(category_counts, key=category_counts.get)
        if category_counts
        else "None"
    )

    avg_confidence = (
        round(
            sum(
                item.get("confidence", 0)
                for item in all_items
            )
            / len(all_items)
            * 100,
            1,
        )
        if all_items
        else 0
    )

    return {
        "unique_items": len(items),
        "total_detections": len(all_items),
        "category_counts": category_counts,
        "top_category": top_category,
        "avg_confidence": avg_confidence,
        "shopping_links": link_count,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "app": APP_TITLE,
    }


@app.get("/robots.txt", response_class=PlainTextResponse)
def robots() -> str:
    return "User-agent: *\nDisallow:\n"


@app.post("/api/detect")
async def detect_image(
    image: UploadFile = File(...),
    confidence: float = Form(DEFAULT_CONFIDENCE),
    iou: float = Form(DEFAULT_IOU),
    model_path: str = Form(str(default_model_path())),
) -> JSONResponse:
    try:
        if confidence < 0 or confidence > 1:
            raise ValueError(
                "Confidence must be between 0 and 1"
            )

        if iou < 0 or iou > 1:
            raise ValueError(
                "IoU must be between 0 and 1"
            )

        if (
            image.content_type
            and image.content_type.lower()
            not in ALLOWED_IMAGE_TYPES
        ):
            raise ValueError(
                "Unsupported image type. "
                "Use JPG, PNG, JPEG, or WEBP."
            )

        started_at = time.perf_counter()

        file_bytes = await image.read()

        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"Image is too large. "
                f"Maximum allowed size is {MAX_UPLOAD_MB} MB."
            )

        image_bgr = decode_upload_to_bgr(file_bytes)

        model = cached_model(model_path)

        result = run_detection(
            model,
            image_bgr,
            confidence=confidence,
            iou=iou,
        )

        process_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

        items = [
            clean_item_dict(item, index)
            for index, item in enumerate(result.unique_items)
        ]

        all_items = [
            clean_item_dict(item, index)
            for index, item in enumerate(result.all_items)
        ]

        items.sort(
            key=lambda item: (
                item.get("category", ""),
                -float(item.get("confidence", 0)),
                item.get("display_name", ""),
            )
        )

        return JSONResponse(
            {
                "ok": True,
                "filename": image.filename or "uploaded_image",
                "original_image": encode_jpeg_data_url(image_bgr),
                "image": encode_jpeg_data_url(
                    result.annotated_image
                ),
                "image_width": int(
                    result.annotated_image.shape[1]
                ),
                "image_height": int(
                    result.annotated_image.shape[0]
                ),
                "unique_count": len(items),
                "all_count": len(all_items),
                "items": items,
                "all_items": all_items,
                "summary": build_summary(
                    items,
                    all_items,
                ),
                "process_ms": process_ms,
                "model": Path(model_path).name,
            }
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {exc}",
        ) from exc


HTML_PAGE = r'''
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta
    name="viewport"
    content="width=device-width, initial-scale=1"
/>
<title>FurniVision AI</title>

<style>
:root {
    --bg: #070914;
    --bg2: #0b1020;
    --panel: rgba(15, 23, 42, 0.82);
    --panel2: rgba(2, 6, 23, 0.88);
    --card: rgba(255, 255, 255, 0.07);
    --card2: rgba(255, 255, 255, 0.105);
    --line: rgba(148, 163, 184, 0.22);
    --text: #f8fafc;
    --muted: #94a3b8;
    --soft: #cbd5e1;
    --brand: #38bdf8;
    --brand2: #a78bfa;
    --good: #22c55e;
    --warn: #f59e0b;
    --danger: #fb7185;
    --shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
}

* {
    box-sizing: border-box;
}

html,
body {
    height: 100%;
}

body {
    margin: 0;
    color: var(--text);
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        "Segoe UI",
        sans-serif;
    background:
        radial-gradient(
            circle at 18% 5%,
            rgba(56, 189, 248, 0.18),
            transparent 26%
        ),
        radial-gradient(
            circle at 88% 14%,
            rgba(167, 139, 250, 0.16),
            transparent 24%
        ),
        linear-gradient(
            135deg,
            #030712,
            #0f172a 52%,
            #020617
        );
    overflow: hidden;
}

button,
input,
select {
    font: inherit;
}

.app {
    height: 100vh;
    display: grid;
    grid-template-columns:
        310px
        minmax(0, 1fr)
        390px;
    gap: 14px;
    padding: 14px;
}

.glass {
    background:
        linear-gradient(
            180deg,
            var(--panel),
            rgba(15, 23, 42, 0.68)
        );
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
    border-radius: 26px;
    overflow: hidden;
}

.left,
.right {
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.brand {
    padding: 20px;
    border-bottom: 1px solid var(--line);
}

.logo {
    display: flex;
    align-items: center;
    gap: 12px;
}

.mark {
    width: 44px;
    height: 44px;
    border-radius: 15px;
    background:
        linear-gradient(
            135deg,
            var(--brand),
            var(--brand2)
        );
    display: grid;
    place-items: center;
    font-size: 22px;
    box-shadow:
        0 12px 30px
        rgba(56, 189, 248, 0.25);
}

h1 {
    font-size: 18px;
    margin: 0;
    letter-spacing: 0.2px;
}

.sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
}

.upload {
    margin: 16px;
    padding: 18px;
    border:
        1.5px dashed
        rgba(148, 163, 184, 0.35);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.045);
    text-align: center;
    cursor: pointer;
    transition: 0.18s;
}

.upload:hover,
.upload.drag {
    border-color: var(--brand);
    background: rgba(56, 189, 248, 0.08);
    transform: translateY(-1px);
}

.upload .big {
    font-weight: 900;
    font-size: 15px;
}

.upload p {
    margin: 8px 0 0;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
}

.hidden {
    display: none !important;
}

.section {
    padding: 0 16px 16px;
}

.label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 900;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #e2e8f0;
    margin: 16px 0 10px;
}

.control {
    background: rgba(2, 6, 23, 0.42);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 13px;
    margin-bottom: 10px;
}

.control-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}

.control b {
    font-size: 13px;
}

.value {
    font-size: 12px;
    color: var(--brand);
    font-weight: 900;
}

.range {
    width: 100%;
    accent-color: var(--brand);
    margin-top: 10px;
}

.primary,
.secondary,
.ghost {
    border: 0;
    border-radius: 16px;
    padding: 12px 14px;
    color: var(--text);
    font-weight: 900;
    cursor: pointer;
    transition: 0.16s;
}

.primary {
    width: 100%;
    background:
        linear-gradient(
            135deg,
            var(--brand),
            var(--brand2)
        );
    box-shadow:
        0 18px 38px
        rgba(56, 189, 248, 0.23);
}

.primary:hover {
    transform: translateY(-1px);
}

.primary:disabled {
    opacity: 0.52;
    cursor: not-allowed;
    transform: none;
}

.secondary {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid var(--line);
}

.ghost {
    background: transparent;
    border: 1px solid var(--line);
    color: var(--soft);
}

.grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
}

.mini {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.5;
}

.center {
    display: grid;
    grid-template-rows:
        auto
        minmax(0, 1fr)
        auto;
    min-width: 0;
}

.topbar {
    min-height: 72px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--line);
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 12px;
    background: rgba(255, 255, 255, 0.075);
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    color: var(--soft);
}

.spacer {
    flex: 1;
}

.viewbtn.active {
    background: rgba(56, 189, 248, 0.16);
    border-color: rgba(56, 189, 248, 0.5);
    color: #e0f2fe;
}

.canvas-wrap {
    min-height: 0;
    display: grid;
    place-items: center;
    overflow: auto;
    padding: 22px;
    background:
        radial-gradient(
            circle at 50% 20%,
            rgba(255, 255, 255, 0.04),
            transparent 24%
        );
}

.stage {
    position: relative;
    max-width: 100%;
    max-height: 100%;
    transform-origin: center;
    transition: transform 0.12s ease;
}

.imgbox {
    position: relative;
    display: block;
}

.main-img {
    display: block;
    max-width: min(
        100%,
        calc(100vw - 760px)
    );
    max-height: calc(100vh - 160px);
    width: auto;
    height: auto;
    border-radius: 18px;
    box-shadow:
        0 24px 70px
        rgba(0, 0, 0, 0.45);
    background: #111;
}

.hotspot {
    position: absolute;
    border:
        2px solid
        rgba(56, 189, 248, 0);
    background: rgba(56, 189, 248, 0);
    border-radius: 10px;
    transition: 0.13s;
    pointer-events: auto;
    cursor: pointer;
}

.hotspot:hover,
.hotspot.active {
    border-color: rgba(56, 189, 248, 0.95);
    background: rgba(56, 189, 248, 0.16);
    box-shadow:
        0 0 0 4px rgba(56, 189, 248, 0.12),
        0 0 26px rgba(56, 189, 248, 0.35);
}

.hotspot-label {
    position: absolute;
    z-index: 120;
    display: inline-flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
    text-decoration: none;
    color: #f8fafc;
    background: #0b1b2f;
    border:
        1px solid
        rgba(56, 189, 248, 0.48);
    border-radius: 999px;
    padding: 5px 8px;
    font-size: 10.5px;
    font-weight: 1000;
    line-height: 1;
    box-shadow:
        0 8px 18px rgba(0, 0, 0, 0.3),
        inset 0 -2px 0 rgba(56, 189, 248, 0.18);
    cursor: pointer;
    transition: 0.14s;
}

.hotspot-label::after {
    content: "↗";
    color: #facc15;
    font-size: 11px;
    font-weight: 1000;
}

.hotspot-label:hover,
.hotspot-label.active {
    background: #12345a;
    border-color: rgba(250, 204, 21, 0.9);
    box-shadow:
        0 0 0 4px rgba(250, 204, 21, 0.14),
        0 12px 28px rgba(0, 0, 0, 0.45);
    transform: translateY(-1px);
}

.leader-line {
    position: absolute;
    height: 2px;
    background:
        linear-gradient(
            90deg,
            rgba(56, 189, 248, 0.15),
            rgba(56, 189, 248, 0.9)
        );
    transform-origin: 0 50%;
    border-radius: 99px;
    pointer-events: none;
    z-index: 80;
}

.leader-line.active {
    height: 3px;
    background:
        linear-gradient(
            90deg,
            rgba(250, 204, 21, 0.2),
            rgba(250, 204, 21, 0.95)
        );
    box-shadow:
        0 0 14px
        rgba(250, 204, 21, 0.5);
}

.empty {
    width: min(760px, 90%);
    min-height: 460px;
    display: grid;
    place-items: center;
    text-align: center;
    border:
        1px dashed
        rgba(148, 163, 184, 0.35);
    border-radius: 28px;
    background: rgba(255, 255, 255, 0.04);
    padding: 34px;
}

.empty h2 {
    margin: 0 0 8px;
    font-size: 32px;
}

.empty p {
    margin: 0;
    color: var(--muted);
    line-height: 1.6;
}

.footerbar {
    min-height: 60px;
    border-top: 1px solid var(--line);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    color: var(--muted);
    font-size: 12px;
}

.zoom {
    width: 160px;
    accent-color: var(--brand);
}

.right-head {
    padding: 18px;
    border-bottom: 1px solid var(--line);
}

.search {
    width: 100%;
    padding: 13px 14px;
    border-radius: 16px;
    border: 1px solid var(--line);
    background: rgba(2, 6, 23, 0.44);
    color: var(--text);
    outline: none;
}

.stats {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 12px;
}

.stat {
    background: rgba(255, 255, 255, 0.065);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 13px;
}

.stat span {
    display: block;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat b {
    font-size: 22px;
}

.chips {
    display: flex;
    gap: 8px;
    overflow: auto;
    padding: 12px 18px;
    border-bottom: 1px solid var(--line);
}

.chip {
    flex: 0 0 auto;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.06);
    color: var(--soft);
    border-radius: 999px;
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 900;
    cursor: pointer;
}

.chip.active {
    background: rgba(56, 189, 248, 0.18);
    border-color: rgba(56, 189, 248, 0.52);
    color: #e0f2fe;
}

.items {
    padding: 14px 14px 18px;
    overflow: auto;
    min-height: 0;
}

.card {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    margin-bottom: 10px;
    padding: 13px;
    border: 1px solid var(--line);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.055);
    cursor: pointer;
    transition: 0.14s;
}

.card:hover,
.card.active {
    transform: translateY(-1px);
    background: rgba(56, 189, 248, 0.11);
    border-color: rgba(56, 189, 248, 0.42);
}

.card h3 {
    margin: 0;
    font-size: 14px;
}

.meta {
    margin-top: 6px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.tag {
    font-size: 11px;
    color: var(--soft);
    background: rgba(255, 255, 255, 0.08);
    border:
        1px solid
        rgba(148, 163, 184, 0.18);
    border-radius: 999px;
    padding: 5px 8px;
}

.buy {
    align-self: start;
    text-decoration: none;
    color: #020617;
    background: #facc15;
    border-radius: 12px;
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 1000;
}

.toast {
    position: fixed;
    left: 50%;
    bottom: 24px;
    transform:
        translateX(-50%)
        translateY(20px);
    opacity: 0;
    background: #0f172a;
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    padding: 12px 16px;
    border-radius: 16px;
    transition: 0.2s;
    z-index: 20;
}

.toast.show {
    opacity: 1;
    transform:
        translateX(-50%)
        translateY(0);
}

.loader {
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.72);
    backdrop-filter: blur(12px);
    display: none;
    place-items: center;
    z-index: 30;
}

.loader.show {
    display: grid;
}

.load-card {
    width: min(420px, 90vw);
    padding: 26px;
    border-radius: 24px;
    background: #0f172a;
    border: 1px solid var(--line);
    text-align: center;
}

.spinner {
    width: 54px;
    height: 54px;
    border:
        5px solid
        rgba(255, 255, 255, 0.12);
    border-top-color: var(--brand);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 16px;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

.compare {
    position: relative;
    overflow: hidden;
    border-radius: 18px;
}

.compare img {
    display: block;
}

.compare .after {
    position: absolute;
    inset: 0;
    clip-path: inset(0 0 0 50%);
}

.compare input {
    position: absolute;
    inset: auto 18px 18px 18px;
    width: calc(100% - 36px);
    accent-color: var(--brand);
}

.compare-line {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 2px;
    background: #fff;
    box-shadow:
        0 0 14px
        rgba(255, 255, 255, 0.6);
}

@media (max-width: 1100px) {
    body {
        overflow: auto;
    }

    .app {
        height: auto;
        min-height: 100vh;
        grid-template-columns: 1fr;
    }

    .center {
        min-height: 680px;
    }

    .main-img {
        max-width: calc(100vw - 72px);
    }

    .left,
    .right {
        min-height: auto;
    }

    .topbar,
    .footerbar {
        height: auto;
        flex-wrap: wrap;
    }
}
</style>
</head>

<body>
<div class="app">
    <aside class="left glass">
        <div class="brand">
            <div class="logo">
                <div class="mark">◆</div>

                <div>
                    <h1>FurniVision AI</h1>

                    <div class="sub">
                        Market-ready furniture detection studio
                    </div>
                </div>
            </div>
        </div>

        <label
            id="dropZone"
            class="upload"
            for="fileInput"
        >
            <input
                id="fileInput"
                class="hidden"
                type="file"
                accept="image/*"
            >

            <div class="big">
                Drop room image here
            </div>

            <p>
                or click to upload JPG / PNG / WEBP.
                Your detector stays exactly the same.
            </p>
        </label>

        <div class="section">
            <div class="label">
                <span>Detection Settings</span>
                <span id="modelState">Ready</span>
            </div>

            <div class="control">
                <div class="control-row">
                    <b>Confidence</b>
                    <span
                        class="value"
                        id="confValue"
                    ></span>
                </div>

                <input
                    id="confidence"
                    class="range"
                    type="range"
                    min="0.05"
                    max="0.90"
                    step="0.01"
                    value="0.12"
                >
            </div>

            <div class="control">
                <div class="control-row">
                    <b>IoU / Overlap</b>

                    <span
                        class="value"
                        id="iouValue"
                    ></span>
                </div>

                <input
                    id="iou"
                    class="range"
                    type="range"
                    min="0.20"
                    max="0.90"
                    step="0.01"
                    value="0.50"
                >
            </div>

            <button
                id="detectBtn"
                class="primary"
                disabled
            >
                Detect Furniture
            </button>

            <div
                class="grid2"
                style="margin-top: 10px"
            >
                <button
                    id="resetBtn"
                    class="secondary"
                >
                    Reset
                </button>

                <button
                    id="demoTips"
                    class="secondary"
                >
                    Tips
                </button>
            </div>

            <p class="mini">
                Tip: for cleaner results, use a bright
                wide-angle room photo. Lower confidence
                catches more objects; higher confidence
                reduces false tags.
            </p>
        </div>
    </aside>

    <main class="center glass">
        <div class="topbar">
            <span
                class="pill"
                id="filePill"
            >
                No image selected
            </span>

            <span
                class="pill"
                id="sizePill"
            >
                —
            </span>

            <span
                class="pill"
                id="speedPill"
            >
                —
            </span>

            <div class="spacer"></div>

            <button
                class="secondary viewbtn active"
                data-view="detected"
            >
                Detected
            </button>

            <button
                class="secondary viewbtn"
                data-view="original"
            >
                Original
            </button>

            <button
                class="secondary viewbtn"
                data-view="compare"
            >
                Before / After
            </button>
        </div>

        <div
            class="canvas-wrap"
            id="canvasWrap"
        >
            <div
                class="empty"
                id="empty"
            >
                <div>
                    <h2>Upload. Detect. Shop.</h2>

                    <p>
                        Clean dashboard, real before/after,
                        clickable products, filters, exports,
                        hover highlights, upload validation,
                        and faster cached model serving.
                    </p>
                </div>
            </div>

            <div
                class="stage hidden"
                id="stage"
            >
                <div
                    class="imgbox"
                    id="imageBox"
                >
                    <img
                        id="mainImage"
                        class="main-img"
                        alt="Detection output"
                    >
                </div>
            </div>
        </div>

        <div class="footerbar">
            <button
                id="fitBtn"
                class="ghost"
            >
                Fit
            </button>

            <button
                id="zoomOut"
                class="ghost"
            >
                −
            </button>

            <input
                id="zoomRange"
                class="zoom"
                type="range"
                min="50"
                max="200"
                value="100"
            >

            <button
                id="zoomIn"
                class="ghost"
            >
                +
            </button>

            <span id="zoomLabel">
                100%
            </span>

            <div class="spacer"></div>

            <button
                id="downloadImg"
                class="ghost"
            >
                Download Image
            </button>

            <button
                id="copyLinks"
                class="ghost"
            >
                Copy Links
            </button>

            <button
                id="exportJson"
                class="ghost"
            >
                JSON
            </button>

            <button
                id="exportCsv"
                class="ghost"
            >
                CSV
            </button>
        </div>
    </main>

    <aside class="right glass">
        <div class="right-head">
            <input
                id="search"
                class="search"
                placeholder="Search detected item..."
            >

            <div class="stats">
                <div class="stat">
                    <span>Unique</span>
                    <b id="uniqueStat">0</b>
                </div>

                <div class="stat">
                    <span>Total</span>
                    <b id="totalStat">0</b>
                </div>

                <div class="stat">
                    <span>Avg Conf.</span>
                    <b id="confStat">0%</b>
                </div>

                <div class="stat">
                    <span>Top Cat.</span>

                    <b
                        id="catStat"
                        style="font-size: 15px"
                    >
                        —
                    </b>
                </div>
            </div>
        </div>

        <div
            class="chips"
            id="chips"
        >
            <button
                class="chip active"
                data-cat="All"
            >
                All
            </button>
        </div>

        <div
            class="items"
            id="items"
        >
            <p class="mini">
                Detected objects will appear here.
                Hover a card to highlight the object;
                click Buy to open product search.
            </p>
        </div>
    </aside>
</div>

<div
    class="loader"
    id="loader"
>
    <div class="load-card">
        <div class="spinner"></div>

        <h2>Detecting furniture...</h2>

        <p class="mini">
            Model is analysing your room image
            and preparing clickable results.
        </p>
    </div>
</div>

<div
    class="toast"
    id="toast"
></div>

<script>
const $ = selector => document.querySelector(selector);
const $$ = selector => Array.from(
    document.querySelectorAll(selector)
);

let selectedFile = null;
let lastData = null;
let currentView = "detected";
let activeCat = "All";
let zoom = 100;
let activeId = null;

const fileInput = $("#fileInput");
const dropZone = $("#dropZone");
const detectBtn = $("#detectBtn");
const loader = $("#loader");
const stage = $("#stage");
const empty = $("#empty");
const imageBox = $("#imageBox");
const mainImage = $("#mainImage");

function toast(message) {
    const element = $("#toast");

    element.textContent = message;
    element.classList.add("show");

    setTimeout(
        () => element.classList.remove("show"),
        2200
    );
}

function fmtPct(value) {
    return Math.round(value * 100) + "%";
}

function updateSliders() {
    $("#confValue").textContent = fmtPct(
        $("#confidence").value
    );

    $("#iouValue").textContent = fmtPct(
        $("#iou").value
    );
}

["confidence", "iou"].forEach(id => {
    $("#" + id).addEventListener(
        "input",
        updateSliders
    );
});

updateSliders();

function setFile(file) {
    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        toast("Please upload an image file");
        return;
    }

    selectedFile = file;
    detectBtn.disabled = false;

    $("#filePill").textContent = file.name;

    const url = URL.createObjectURL(file);

    mainImage.src = url;

    empty.classList.add("hidden");
    stage.classList.remove("hidden");

    clearHotspots();

    $("#items").innerHTML =
        '<p class="mini">' +
        "Image loaded. Press Detect Furniture." +
        "</p>";

    $("#sizePill").textContent = "Preview";
}

fileInput.addEventListener(
    "change",
    event => setFile(event.target.files[0])
);

["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.add("drag");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, event => {
        event.preventDefault();
        dropZone.classList.remove("drag");
    });
});

dropZone.addEventListener("drop", event => {
    setFile(event.dataTransfer.files[0]);
});

detectBtn.addEventListener(
    "click",
    detect
);

async function detect() {
    if (!selectedFile) {
        return;
    }

    loader.classList.add("show");
    detectBtn.disabled = true;
    $("#modelState").textContent = "Running";

    try {
        const formData = new FormData();

        formData.append(
            "image",
            selectedFile
        );

        formData.append(
            "confidence",
            $("#confidence").value
        );

        formData.append(
            "iou",
            $("#iou").value
        );

        const response = await fetch(
            "/api/detect",
            {
                method: "POST",
                body: formData,
            }
        );

        const data = await response.json();

        if (!response.ok || !data.ok) {
            throw new Error(
                data.detail || "Detection failed"
            );
        }

        lastData = data;
        currentView = "detected";
        activeCat = "All";
        activeId = null;

        renderAll();
        toast("Detection complete");
    } catch (error) {
        toast(
            error.message ||
            "Something went wrong"
        );
    } finally {
        loader.classList.remove("show");
        detectBtn.disabled = false;
        $("#modelState").textContent = "Ready";
    }
}

function renderAll() {
    if (!lastData) {
        return;
    }

    $("#filePill").textContent =
        lastData.filename;

    $("#sizePill").textContent =
        `${lastData.image_width} × ` +
        `${lastData.image_height}`;

    $("#speedPill").textContent =
        lastData.process_ms
            ? `${lastData.process_ms} ms • ` +
              `${lastData.model || "model"}`
            : "—";

    $("#uniqueStat").textContent =
        lastData.summary.unique_items;

    $("#totalStat").textContent =
        lastData.summary.total_detections;

    $("#confStat").textContent =
        lastData.summary.avg_confidence + "%";

    $("#catStat").textContent =
        lastData.summary.top_category;

    renderImage();
    renderChips();
    renderItems();
}

function setView(view) {
    currentView = view;

    $$(".viewbtn").forEach(button => {
        button.classList.toggle(
            "active",
            button.dataset.view === view
        );
    });

    renderImage();
}

$$(".viewbtn").forEach(button => {
    button.addEventListener(
        "click",
        () => setView(button.dataset.view)
    );
});

function clearHotspots() {
    imageBox.querySelectorAll(
        ".hotspot," +
        ".hotspot-label," +
        ".leader-line," +
        ".compare-line," +
        ".compare-range," +
        ".compare-img"
    ).forEach(element => element.remove());

    imageBox.classList.remove("compare");
}

function renderImage() {
    clearHotspots();

    if (!lastData) {
        return;
    }

    empty.classList.add("hidden");
    stage.classList.remove("hidden");

    if (currentView === "original") {
        mainImage.src = lastData.original_image;
        return;
    }

    if (currentView === "compare") {
        imageBox.classList.add("compare");
        mainImage.src = lastData.original_image;

        const after = document.createElement("img");
        after.src = lastData.image;
        after.className =
            "main-img after compare-img";

        const line = document.createElement("div");
        line.className = "compare-line";

        const range = document.createElement("input");
        range.className = "compare-range";
        range.type = "range";
        range.min = 0;
        range.max = 100;
        range.value = 50;

        range.oninput = () => {
            after.style.clipPath =
                `inset(0 0 0 ${range.value}%)`;

            line.style.left =
                range.value + "%";
        };

        imageBox.append(
            after,
            line,
            range
        );

        return;
    }

    mainImage.src = lastData.original_image;
    mainImage.onload = renderHotspots;

    if (mainImage.complete) {
        renderHotspots();
    }
}

function clamp(number, minimum, maximum) {
    return Math.max(
        minimum,
        Math.min(maximum, number)
    );
}

function itemArea(item) {
    const [x1, y1, x2, y2] = item.bbox;

    return (
        Math.max(1, x2 - x1) *
        Math.max(1, y2 - y1)
    );
}

function smartHotspotBbox(item) {
    const imageWidth = lastData.image_width;
    const imageHeight = lastData.image_height;

    let [x1, y1, x2, y2] =
        item.bbox.map(Number);

    x1 = clamp(
        x1,
        0,
        imageWidth - 1
    );

    y1 = clamp(
        y1,
        0,
        imageHeight - 1
    );

    x2 = clamp(
        x2,
        x1 + 1,
        imageWidth
    );

    y2 = clamp(
        y2,
        y1 + 1,
        imageHeight
    );

    const name = String(
        item.name || ""
    ).toLowerCase();

    const display = String(
        item.display_name || ""
    ).toLowerCase();

    const category = String(
        item.category || ""
    ).toLowerCase();

    const boxWidth = x2 - x1;
    const boxHeight = y2 - y1;

    const imageArea =
        imageWidth * imageHeight;

    const area =
        boxWidth * boxHeight;

    const isBig =
        area / imageArea > 0.035 ||
        boxWidth / imageWidth > 0.34 ||
        boxHeight / imageHeight > 0.34;

    const mediumPatchNames = [
        "window blind",
        "blind",
        "roller blind",
        "vertical blind",
        "whiteboard",
        "conference table",
        "table",
        "floor",
        "carpet",
        "rug",
        "floor rug",
        "floor mat",
        "glass partition",
        "glass window",
        "glass door",
        "door",
        "window",
        "wall",
        "ceiling",
    ];

    const needsMediumPatch =
        isBig &&
        (
            mediumPatchNames.some(keyword =>
                name.includes(keyword) ||
                display.includes(keyword)
            ) ||
            category.includes("structure") ||
            category.includes("flooring") ||
            category.includes("doors")
        );

    if (!needsMediumPatch) {
        return [x1, y1, x2, y2];
    }

    let centerX = (x1 + x2) / 2;
    let centerY = (y1 + y2) / 2;

    let patchWidth = Math.round(
        clamp(
            boxWidth * 0.42,
            90,
            330
        )
    );

    let patchHeight = Math.round(
        clamp(
            boxHeight * 0.38,
            65,
            230
        )
    );

    if (
        name.includes("whiteboard") ||
        display.includes("whiteboard")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.42,
                110,
                300
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.42,
                80,
                230
            )
        );

        centerX = x1 + boxWidth * 0.58;
        centerY = y1 + boxHeight * 0.52;

    } else if (
        name.includes("conference table") ||
        display.includes("conference table") ||
        name === "table"
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.42,
                130,
                360
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.34,
                90,
                240
            )
        );

        centerX = x1 + boxWidth * 0.50;
        centerY = y1 + boxHeight * 0.55;

    } else if (
        name.includes("window blind") ||
        name.includes("blind") ||
        display.includes("window blind")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.48,
                120,
                340
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.36,
                80,
                230
            )
        );

        centerX = x1 + boxWidth * 0.50;
        centerY = y1 + boxHeight * 0.42;

    } else if (
        name.includes("floor") ||
        name.includes("carpet") ||
        name.includes("rug") ||
        display.includes("floor")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.34,
                120,
                330
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.28,
                75,
                200
            )
        );

        centerX = x1 + boxWidth * 0.62;
        centerY = y1 + boxHeight * 0.70;

    } else if (
        name.includes("ceiling")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.30,
                100,
                300
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.26,
                65,
                170
            )
        );

        centerX = x1 + boxWidth * 0.50;
        centerY = y1 + boxHeight * 0.28;

    } else if (
        name.includes("glass") ||
        name.includes("door") ||
        name.includes("window") ||
        category.includes("doors")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.36,
                100,
                310
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.42,
                95,
                280
            )
        );

        centerX = x1 + boxWidth * 0.48;
        centerY = y1 + boxHeight * 0.50;

    } else if (
        name.includes("wall")
    ) {
        patchWidth = Math.round(
            clamp(
                boxWidth * 0.32,
                100,
                290
            )
        );

        patchHeight = Math.round(
            clamp(
                boxHeight * 0.32,
                80,
                220
            )
        );

        centerX = x1 + boxWidth * 0.52;
        centerY = y1 + boxHeight * 0.48;
    }

    let newX1 = Math.round(
        centerX - patchWidth / 2
    );

    let newY1 = Math.round(
        centerY - patchHeight / 2
    );

    let newX2 =
        newX1 + patchWidth;

    let newY2 =
        newY1 + patchHeight;

    newX1 = clamp(
        newX1,
        x1,
        x2 - 1
    );

    newY1 = clamp(
        newY1,
        y1,
        y2 - 1
    );

    newX2 = clamp(
        newX2,
        newX1 + 1,
        x2
    );

    newY2 = clamp(
        newY2,
        newY1 + 1,
        y2
    );

    return [
        newX1,
        newY1,
        newX2,
        newY2,
    ];
}

function labelOrder(first, second) {
    const firstPriority =
        labelPriority(first);

    const secondPriority =
        labelPriority(second);

    if (firstPriority !== secondPriority) {
        return firstPriority - secondPriority;
    }

    return itemArea(first) - itemArea(second);
}

function labelPriority(item) {
    const display = String(
        item.display_name || ""
    ).toLowerCase();

    const name = String(
        item.name || ""
    ).toLowerCase();

    if (
        display.includes("sideboard") ||
        display.includes("landline") ||
        display.includes("whiteboard") ||
        display.includes("conference table")
    ) {
        return 0;
    }

    if (
        display.includes("conference chair") ||
        display.includes("chair")
    ) {
        return 1;
    }

    if (
        display.includes("television") ||
        display.includes("screen") ||
        display.includes("lamp")
    ) {
        return 2;
    }

    if (
        name.includes("ceiling") ||
        name.includes("floor") ||
        name.includes("glass") ||
        name.includes("window") ||
        name.includes("partition")
    ) {
        return 4;
    }

    return 3;
}

function renderHotspots() {
    if (
        !lastData ||
        currentView !== "detected"
    ) {
        return;
    }

    clearHotspots();

    const scaleX =
        mainImage.clientWidth /
        lastData.image_width;

    const scaleY =
        mainImage.clientHeight /
        lastData.image_height;

    const items = filteredItems()
        .slice()
        .sort(labelOrder);

    const placed = [];

    const boxWidth =
        mainImage.clientWidth;

    const boxHeight =
        mainImage.clientHeight;

    function overlaps(first, second, padding = 6) {
        return !(
            first.x + first.w + padding < second.x ||
            second.x + second.w + padding < first.x ||
            first.y + first.h + padding < second.y ||
            second.y + second.h + padding < first.y
        );
    }

    function overlapScore(rectangle) {
        let score = 0;

        placed.forEach(previous => {
            if (!overlaps(rectangle, previous, 4)) {
                return;
            }

            const intersectionWidth = Math.max(
                0,
                Math.min(
                    rectangle.x + rectangle.w,
                    previous.x + previous.w
                ) - Math.max(
                    rectangle.x,
                    previous.x
                )
            );

            const intersectionHeight = Math.max(
                0,
                Math.min(
                    rectangle.y + rectangle.h,
                    previous.y + previous.h
                ) - Math.max(
                    rectangle.y,
                    previous.y
                )
            );

            score +=
                intersectionWidth *
                intersectionHeight +
                5000;
        });

        return score;
    }

    function clampRect(rectangle) {
        rectangle.x = clamp(
            rectangle.x,
            4,
            Math.max(
                4,
                boxWidth - rectangle.w - 4
            )
        );

        rectangle.y = clamp(
            rectangle.y,
            4,
            Math.max(
                4,
                boxHeight - rectangle.h - 4
            )
        );

        return rectangle;
    }

    function bestLabelRect(
        centerX,
        centerY,
        x1,
        y1,
        x2,
        y2,
        text
    ) {
        const width = Math.ceil(
            clamp(
                text.length * 6.2 + 26,
                58,
                210
            )
        );

        const height = 24;
        const candidates = [];

        const offsets = [
            8,
            18,
            30,
            44,
            60,
            78,
            98,
            122,
        ];

        offsets.forEach(offset => {
            candidates.push(
                {
                    x: centerX - width / 2,
                    y: y1 - height - offset,
                    w: width,
                    h: height,
                },
                {
                    x: centerX - width / 2,
                    y: y2 + offset,
                    w: width,
                    h: height,
                },
                {
                    x: x2 + offset,
                    y: centerY - height / 2,
                    w: width,
                    h: height,
                },
                {
                    x: x1 - width - offset,
                    y: centerY - height / 2,
                    w: width,
                    h: height,
                }
            );
        });

        candidates.push(
            {
                x: centerX - width / 2,
                y: 8,
                w: width,
                h: height,
            },
            {
                x: centerX - width / 2,
                y: boxHeight - height - 8,
                w: width,
                h: height,
            },
            {
                x: 8,
                y: centerY - height / 2,
                w: width,
                h: height,
            },
            {
                x: boxWidth - width - 8,
                y: centerY - height / 2,
                w: width,
                h: height,
            }
        );

        let best = null;
        let bestScore = Infinity;

        candidates.forEach(raw => {
            const rectangle = clampRect({
                ...raw,
            });

            const distance = Math.hypot(
                rectangle.x +
                    rectangle.w / 2 -
                    centerX,
                rectangle.y +
                    rectangle.h / 2 -
                    centerY
            );

            const edgePenalty =
                rectangle.x < 8 ||
                rectangle.y < 8 ||
                rectangle.x + rectangle.w >
                    boxWidth - 8 ||
                rectangle.y + rectangle.h >
                    boxHeight - 8
                    ? 60
                    : 0;

            const score =
                overlapScore(rectangle) +
                distance +
                edgePenalty;

            if (score < bestScore) {
                bestScore = score;
                best = rectangle;
            }
        });

        placed.push(best);

        return best;
    }

    function addLeader(
        id,
        x1,
        y1,
        x2,
        y2,
        labelX,
        labelY,
        labelWidth,
        labelHeight
    ) {
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;

        const targetX =
            labelX + labelWidth / 2;

        const targetY =
            labelY + labelHeight / 2;

        const differenceX =
            targetX - centerX;

        const differenceY =
            targetY - centerY;

        const length = Math.max(
            10,
            Math.hypot(
                differenceX,
                differenceY
            )
        );

        const line =
            document.createElement("div");

        line.className = "leader-line";
        line.dataset.id = id;
        line.style.left = centerX + "px";
        line.style.top = centerY + "px";
        line.style.width = length + "px";

        line.style.transform =
            `rotate(${
                Math.atan2(
                    differenceY,
                    differenceX
                )
            }rad)`;

        imageBox.appendChild(line);
    }

    items.forEach((item, index) => {
        const [
            boxX1,
            boxY1,
            boxX2,
            boxY2,
        ] = smartHotspotBbox(item);

        const x1 = boxX1 * scaleX;
        const y1 = boxY1 * scaleY;
        const x2 = boxX2 * scaleX;
        const y2 = boxY2 * scaleY;

        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;

        const labelText =
            `${item.display_name}` +
            `${
                item.count > 1
                    ? " ×" + item.count
                    : ""
            }`;

        const hotspot =
            document.createElement("a");

        hotspot.className = "hotspot";
        hotspot.dataset.id = item.id;

        hotspot.setAttribute(
            "aria-label",
            item.display_name
        );

        hotspot.href = item.link;
        hotspot.target = "_blank";
        hotspot.rel = "noopener";

        hotspot.style.left = x1 + "px";
        hotspot.style.top = y1 + "px";

        hotspot.style.width =
            Math.max(12, x2 - x1) + "px";

        hotspot.style.height =
            Math.max(12, y2 - y1) + "px";

        hotspot.style.zIndex =
            String(10 + index);

        hotspot.addEventListener(
            "mouseenter",
            () => activate(item.id)
        );

        hotspot.addEventListener(
            "mouseleave",
            () => activate(null)
        );

        imageBox.appendChild(hotspot);

        const rectangle = bestLabelRect(
            centerX,
            centerY,
            x1,
            y1,
            x2,
            y2,
            labelText
        );

        addLeader(
            item.id,
            x1,
            y1,
            x2,
            y2,
            rectangle.x,
            rectangle.y,
            rectangle.w,
            rectangle.h
        );

        const label =
            document.createElement("a");

        label.className = "hotspot-label";
        label.dataset.id = item.id;
        label.href = item.link;
        label.target = "_blank";
        label.rel = "noopener";

        label.setAttribute(
            "aria-label",
            `Open Amazon search for ` +
            `${item.display_name}`
        );

        label.textContent = labelText;
        label.style.left =
            rectangle.x + "px";

        label.style.top =
            rectangle.y + "px";

        label.style.width =
            rectangle.w + "px";

        label.addEventListener(
            "mouseenter",
            () => activate(item.id)
        );

        label.addEventListener(
            "mouseleave",
            () => activate(null)
        );

        imageBox.appendChild(label);
    });
}

function displayItems() {
    const raw = lastData?.items || [];

    const hasConferenceChair = raw.some(
        item =>
            String(
                item.display_name || ""
            )
                .toLowerCase()
                .includes("conference chair")
    );

    const whiteboards = raw.filter(
        item =>
            String(
                item.display_name || ""
            )
                .toLowerCase()
                .includes("whiteboard")
    );

    const cleaned = [];

    function bboxCenter(item) {
        const [x1, y1, x2, y2] =
            item.bbox.map(Number);

        return [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
        ];
    }

    function inside(
        item,
        boxItem,
        padding = 18
    ) {
        const [centerX, centerY] =
            bboxCenter(item);

        const [x1, y1, x2, y2] =
            boxItem.bbox.map(Number);

        return (
            centerX >= x1 - padding &&
            centerX <= x2 + padding &&
            centerY >= y1 - padding &&
            centerY <= y2 + padding
        );
    }

    function overlapRatio(first, second) {
        const [
            firstX1,
            firstY1,
            firstX2,
            firstY2,
        ] = first.bbox.map(Number);

        const [
            secondX1,
            secondY1,
            secondX2,
            secondY2,
        ] = second.bbox.map(Number);

        const intersectionWidth = Math.max(
            0,
            Math.min(firstX2, secondX2) -
            Math.max(firstX1, secondX1)
        );

        const intersectionHeight = Math.max(
            0,
            Math.min(firstY2, secondY2) -
            Math.max(firstY1, secondY1)
        );

        const intersection =
            intersectionWidth *
            intersectionHeight;

        const firstArea = Math.max(
            1,
            (firstX2 - firstX1) *
            (firstY2 - firstY1)
        );

        return intersection / firstArea;
    }

    raw.forEach(item => {
        const name = String(
            item.name || ""
        ).toLowerCase();

        const display = String(
            item.display_name || ""
        ).toLowerCase();

        if (
            name.includes(
                "linear slot diffuser"
            ) ||
            display.includes(
                "linear slot diffuser"
            ) ||
            name.includes(
                "ceiling linear diffuser"
            )
        ) {
            return;
        }

        if (
            hasConferenceChair &&
            (
                display === "chair" ||
                name === "chair"
            )
        ) {
            return;
        }

        const isMonitor =
            display.includes("monitor") ||
            (
                display.includes("screen") &&
                !display.includes("television")
            ) ||
            name === "monitor" ||
            name === "screen" ||
            name === "computer monitor";

        if (
            isMonitor &&
            whiteboards.some(
                whiteboard =>
                    inside(
                        item,
                        whiteboard,
                        28
                    ) ||
                    overlapRatio(
                        item,
                        whiteboard
                    ) > 0.10
            )
        ) {
            return;
        }

        cleaned.push({
            ...item,
        });
    });

    const merged = new Map();

    cleaned.forEach(item => {
        const key =
            String(
                item.display_name ||
                item.name
            ).toLowerCase() +
            "|" +
            String(
                item.category || ""
            ).toLowerCase();

        if (!merged.has(key)) {
            merged.set(
                key,
                {
                    ...item,
                    count: Number(
                        item.count || 1
                    ),
                }
            );

            return;
        }

        const previous = merged.get(key);

        previous.count =
            Number(previous.count || 1) +
            Number(item.count || 1);

        if (
            Number(item.confidence || 0) >
            Number(previous.confidence || 0)
        ) {
            const keepCount =
                previous.count;

            merged.set(
                key,
                {
                    ...item,
                    count: keepCount,
                }
            );
        }
    });

    return Array.from(
        merged.values()
    );
}

function categories() {
    const categorySet = new Set(
        displayItems().map(
            item => item.category
        )
    );

    return [
        "All",
        ...Array.from(categorySet).sort(),
    ];
}

function renderChips() {
    const wrapper = $("#chips");

    wrapper.innerHTML = "";

    categories().forEach(category => {
        const button =
            document.createElement("button");

        button.className =
            "chip" +
            (
                category === activeCat
                    ? " active"
                    : ""
            );

        button.dataset.cat = category;
        button.textContent = category;

        button.onclick = () => {
            activeCat = category;
            renderChips();
            renderItems();
            renderImage();
        };

        wrapper.appendChild(button);
    });
}

function filteredItems() {
    const query = $("#search")
        .value
        .trim()
        .toLowerCase();

    return displayItems().filter(item => {
        const categoryMatches =
            activeCat === "All" ||
            item.category === activeCat;

        const queryMatches =
            !query ||
            item.display_name
                .toLowerCase()
                .includes(query) ||
            item.category
                .toLowerCase()
                .includes(query) ||
            item.name
                .toLowerCase()
                .includes(query);

        return (
            categoryMatches &&
            queryMatches
        );
    });
}

$("#search").addEventListener(
    "input",
    () => {
        renderItems();
        renderImage();
    }
);

function renderItems() {
    const list = $("#items");
    const items = filteredItems();

    if (!lastData) {
        return;
    }

    if (!items.length) {
        list.innerHTML =
            '<p class="mini">' +
            "No items match this filter." +
            "</p>";

        return;
    }

    list.innerHTML = "";

    items.forEach(item => {
        const card =
            document.createElement("div");

        card.className = "card";
        card.dataset.id = item.id;

        card.innerHTML =
            `<div>` +
                `<h3>` +
                    `${escapeHtml(
                        item.display_name
                    )} ` +
                    `${
                        item.count > 1
                            ? `×${item.count}`
                            : ""
                    }` +
                `</h3>` +
                `<div class="meta">` +
                    `<span class="tag">` +
                        `${escapeHtml(
                            item.category
                        )}` +
                    `</span>` +
                    `<span class="tag">` +
                        `Detected` +
                    `</span>` +
                    `<span class="tag">` +
                        `${item.bbox.join(", ")}` +
                    `</span>` +
                `</div>` +
            `</div>` +
            `<a ` +
                `class="buy" ` +
                `href="${item.link}" ` +
                `target="_blank" ` +
                `rel="noopener"` +
            `>` +
                `Buy` +
            `</a>`;

        card.addEventListener(
            "mouseenter",
            () => activate(item.id)
        );

        card.addEventListener(
            "mouseleave",
            () => activate(null)
        );

        card.addEventListener(
            "click",
            event => {
                if (
                    event.target.tagName !== "A"
                ) {
                    activeId = item.id;

                    activate(
                        item.id,
                        true
                    );
                }
            }
        );

        list.appendChild(card);
    });
}

function activate(id, persist = false) {
    if (!persist && activeId) {
        id = activeId;
    }

    $$(
        ".card," +
        ".hotspot," +
        ".hotspot-label," +
        ".leader-line"
    ).forEach(element => {
        element.classList.toggle(
            "active",
            Boolean(id) &&
            element.dataset.id === id
        );
    });
}

function escapeHtml(value) {
    return String(value).replace(
        /[&<>'"]/g,
        character => ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            "'": "&#39;",
            '"': "&quot;",
        }[character])
    );
}

function download(
    filename,
    content,
    type = "application/octet-stream"
) {
    const blob =
        content instanceof Blob
            ? content
            : new Blob(
                [content],
                {type}
            );

    const anchor =
        document.createElement("a");

    anchor.href =
        URL.createObjectURL(blob);

    anchor.download = filename;
    anchor.click();

    setTimeout(
        () => URL.revokeObjectURL(
            anchor.href
        ),
        1000
    );
}

$("#exportJson").onclick = () => {
    if (!lastData) {
        toast("Detect first");
        return;
    }

    download(
        "furniture-detection-report.json",
        JSON.stringify(
            lastData,
            null,
            2
        ),
        "application/json"
    );
};

$("#exportCsv").onclick = () => {
    if (!lastData) {
        toast("Detect first");
        return;
    }

    const rows = [
        [
            "Name",
            "Category",
            "Confidence",
            "Count",
            "BBox",
            "Amazon Link",
        ],
        ...lastData.items.map(item => [
            item.display_name,
            item.category,
            item.confidence_percent,
            item.count,
            item.bbox.join(" "),
            item.link,
        ]),
    ];

    const csv = rows.map(row =>
        row.map(value =>
            '"' +
            String(value).replaceAll(
                '"',
                '""'
            ) +
            '"'
        ).join(",")
    ).join("\n");

    download(
        "furniture-detection-report.csv",
        csv,
        "text/csv"
    );
};

$("#copyLinks").onclick = async () => {
    if (!lastData) {
        toast("Detect first");
        return;
    }

    await navigator.clipboard.writeText(
        lastData.items.map(item =>
            `${item.display_name}: ` +
            `${item.link}`
        ).join("\n")
    );

    toast("Links copied");
};

$("#downloadImg").onclick = () => {
    if (!lastData) {
        toast("Detect first");
        return;
    }

    fetch(lastData.image)
        .then(response => response.blob())
        .then(blob => download(
            "detected-furniture.jpg",
            blob,
            "image/jpeg"
        ));
};

function setZoom(value) {
    zoom = Math.max(
        50,
        Math.min(200, value)
    );

    $("#zoomRange").value = zoom;
    $("#zoomLabel").textContent =
        zoom + "%";

    stage.style.transform =
        `scale(${zoom / 100})`;
}

$("#zoomRange").oninput = event => {
    setZoom(
        Number(event.target.value)
    );
};

$("#zoomIn").onclick = () => {
    setZoom(zoom + 10);
};

$("#zoomOut").onclick = () => {
    setZoom(zoom - 10);
};

$("#fitBtn").onclick = () => {
    setZoom(100);
};

$("#resetBtn").onclick = () => {
    selectedFile = null;
    lastData = null;
    activeId = null;
    activeCat = "All";
    currentView = "detected";

    fileInput.value = "";
    detectBtn.disabled = true;

    mainImage.removeAttribute("src");

    clearHotspots();

    stage.classList.add("hidden");
    empty.classList.remove("hidden");

    $("#filePill").textContent =
        "No image selected";

    $("#sizePill").textContent = "—";
    $("#speedPill").textContent = "—";
    $("#uniqueStat").textContent = "0";
    $("#totalStat").textContent = "0";
    $("#confStat").textContent = "0%";
    $("#catStat").textContent = "—";

    $("#chips").innerHTML =
        '<button ' +
            'class="chip active" ' +
            'data-cat="All"' +
        '>' +
            'All' +
        '</button>';

    $("#items").innerHTML =
        '<p class="mini">' +
            "Detected objects will appear here. " +
            "Hover a card to highlight the object; " +
            "click Buy to open product search." +
        "</p>";

    toast("Reset done");
};

$("#demoTips").onclick = () => {
    toast(
        "Use clear room photos. " +
        "Try confidence 10–18% for more tags, " +
        "25%+ for cleaner tags."
    );
};

window.addEventListener(
    "resize",
    () => {
        if (lastData) {
            renderImage();
        }
    }
);
</script>
</body>
</html>
'''


def find_free_port(
    start_port: int = 7860,
    max_tries: int = 50,
) -> int:
    for port in range(
        start_port,
        start_port + max_tries,
    ):
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as sock:
            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )

            try:
                sock.bind(
                    ("127.0.0.1", port)
                )

                return port
            except OSError:
                continue

    return start_port


if __name__ == "__main__":
    import uvicorn

    configured_port = os.environ.get("PORT")

    port = (
        int(configured_port)
        if configured_port
        else find_free_port(7860)
    )

    print(
        "\nFurniVision AI running at: "
        f"http://127.0.0.1:{port}\n"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )