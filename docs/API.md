# REST API Reference

## Overview

The AI Furniture Detector API provides RESTful endpoints for furniture detection and product link generation.

**Base URL**: `http://127.0.0.1:8000`

**API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`

## Endpoints

### GET /health
Health check endpoint.

**Response**: `{"status": "ok"}`

### GET /
API information and available endpoints.

### POST /api/detect
Upload image and receive furniture detections.

**Parameters:**
- `image` (file): Room/interior image
- `confidence` (float): Detection threshold, default 0.25
- `iou` (float): NMS IoU threshold, default 0.70
- `include_image` (bool): Include base64 annotated image, default true

**Example:**
```bash
curl -X POST http://127.0.0.1:8000/api/detect \
  -F "image=@room.jpg" \
  -F "confidence=0.25"
```

**Response:**
```json
{
  "filename": "room.jpg",
  "total_boxes": 24,
  "unique_count": 8,
  "unique_items": [
    {
      "name": "chair",
      "confidence": 0.94,
      "coords": [100, 150, 250, 400],
      "category": "Furniture",
      "link": "https://www.amazon.in/s?k=chair"
    }
  ],
  "all_items": []
}
```

### GET /api/product-link
Generate Amazon product search link.

**Parameters:**
- `name` (string): Product name

**Example:**
```bash
curl "http://127.0.0.1:8000/api/product-link?name=office+chair"
```

**Response:**
```json
{
  "name": "office chair",
  "link": "https://www.amazon.in/s?k=office+chair"
}
```

## Categories

Items are categorized as:
- Furniture, Lighting, Doors & Windows, Flooring & Textiles, Electronics, Decor, Accessories, Other

## Parameters

- **confidence** (0-1): Detection sensitivity. Default 0.25 for balanced results.
- **iou** (0-1): Overlap filtering. Default 0.70 for balanced deduplication.

See [ARCHITECTURE.md](../ARCHITECTURE.md) for more details.
