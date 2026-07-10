# Architecture

## Overview

AI Furniture Detector is a production-grade, modular furniture detection system built with:
- **YOLOv8 World** for zero-shot object detection
- **OpenCV** for image processing and visualization
- **FastAPI** for REST API
- **Gradio** for web UI

## Core Components

### 1. Detector Module (`detector.py`)

The heart of the application. Handles:

```
detector.py
├── Model Loading
│   ├── load_model(path) → YOLO
│   ├── configure_yolo_world_classes()
│   └── default_model_path()
├── Detection Pipeline
│   ├── run_detection() → DetectionResult
│   ├── detect() → (items, annotated_image)
│   └── _parse_results()
├── Post-Processing
│   ├── normalize_item_name()
│   ├── categorize_item()
│   └── unique_sidebar_items()
├── Visualization
│   ├── draw_annotations()
│   ├── color_for_item()
│   └── build_amazon_link()
└── I/O
    ├── load_image_bgr()
    ├── save_annotated_image()
    └── save_json_report()
```

**Key Data Structures:**
- `DetectedItem`: Frozen dataclass for immutable detection results
- `DetectionResult`: Aggregates all items and annotated image

### 2. Configuration Module (`config.py`)

Centralized settings management:
- Environment variable support via `.env`
- Pydantic validation
- Path normalization
- Auto-creation of required directories

### 3. Logging Module (`logging_config.py`)

Standardized logging:
- Configurable log levels
- Suppresses verbose third-party logs
- Structured logging throughout codebase

### 4. CLI (`cli.py`)

Command-line interface with:
- Argument parsing
- Interactive OpenCV window
- Clickable bounding boxes
- Real-time sidebar filtering

### 5. FastAPI Backend (`api.py`)

REST API with endpoints:
- `GET /health` - Health check
- `GET /api/product-link?name=...` - Generate product link
- `POST /api/detect` - Image upload and detection
- `GET /docs` - Auto-generated API documentation

### 6. Gradio Web UI (`web_app.py`)

Modern web interface featuring:
- Real-time image upload
- Adjustable confidence/IOU sliders
- Detected items sidebar with filtering
- JSON export
- Dark mode styling

## Data Flow

### Detection Pipeline

```
Input Image (BGR)
    ↓
Load YOLO Model
    ↓
Run Inference (YOLOv8 World)
    ↓
Parse Results
    ├─ Normalize item names
    ├─ Filter by area ratio
    └─ Exclude unwanted items
    ↓
Deduplicate Items
    └─ Keep highest confidence per name
    ↓
Generate Output
    ├─ Draw annotations
    ├─ Create product links
    └─ Serialize to JSON
    ↓
DetectionResult
```

### API Request Flow

```
HTTP POST /api/detect
    ↓
Decode Image (JPEG → BGR)
    ↓
Get Model (cached)
    ↓
run_detection()
    ↓
Serialize to JSON
    ↓
Optional: Encode image to base64
    ↓
HTTP 200 with JSON response
```

## Performance Considerations

1. **Model Caching**: API caches loaded models in-memory (3 slots)
2. **Deduplication**: Reduces drawing complexity by 80%+ on typical images
3. **JPEG Encoding**: Annotated images compressed to 92% quality
4. **Lazy Loading**: Models loaded only when first used

## Error Handling

All modules implement:
- Input validation with descriptive errors
- Try-catch blocks with logging
- Graceful degradation
- Non-zero exit codes on CLI failure

## Type Hints

The codebase uses Python 3.10+ type hints throughout:
- Enables static type checking with MyPy
- Improves IDE autocomplete
- Documents expected inputs/outputs

## Testing

Test structure:
```
tests/
├── test_detector.py      # Core detection logic
├── test_config.py        # Configuration parsing
├── test_api.py           # API endpoints
└── fixtures/             # Sample images
```

Run with: `pytest tests/ --cov=ai_furniture_detector`

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["ai-furniture-detector-api"]
```

### Environment Variables
```bash
MODEL_PATH=./yolov8x-worldv2.pt
DEFAULT_CONFIDENCE=0.25
LOG_LEVEL=INFO
DEBUG=false
```

### Model Files
- `yolov8x-worldv2.pt` (preferred, 130MB)
- `yolov8m.pt` (fallback, 50MB)
- `yolov8n.pt` (fallback, 10MB)

## Extensibility

### Adding New Categories

Update `CATEGORY_COLORS_BGR` in `detector.py`:
```python
CATEGORY_COLORS_BGR = {
    "Your New Category": (B, G, R),
    ...
}
```

### Adding New Object Types

Extend `TARGET_OBJECTS` in `detector.py`:
```python
TARGET_OBJECTS = [
    "existing_item",
    "new_item_type",
    ...
]
```

### Custom Output Formats

Extend `DetectedItem.to_dict()` or create subclass:
```python
class CustomDetectedItem(DetectedItem):
    def to_custom_format(self) -> dict:
        ...
```

## Security

- Input validation on all API endpoints
- CORS configured (can be restricted)
- No arbitrary code execution
- Safe file operations with parent directory checks
- Dependency scanning in CI

## Dependencies

See `pyproject.toml` for:
- Production dependencies (core functionality)
- Development dependencies (testing, linting, type checking)
- Optional dependencies (extensions)

## Version Strategy

- Semantic versioning: MAJOR.MINOR.PATCH
- Breaking changes: Major version bump
- New features: Minor version bump
- Bug fixes: Patch version bump
