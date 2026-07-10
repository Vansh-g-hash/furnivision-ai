# Configuration Guide

## Environment Variables

Create `.env` file in project root (copy from `.env.example`):

```bash
cp .env.example .env
```

### Model Configuration

```bash
# Path to YOLO model weights
MODEL_PATH=./yolov8x-worldv2.pt

# Default detection confidence threshold (0-1)
DEFAULT_CONFIDENCE=0.25

# Default NMS IoU threshold (0-1)
DEFAULT_IOU=0.70

# Minimum box area ratio (filter very small detections)
MIN_BOX_AREA_RATIO=0.00008
```

### Server Configuration

```bash
# API server host
API_HOST=127.0.0.1

# API server port
API_PORT=8000

# Web UI server host
WEB_HOST=127.0.0.1

# Web UI server port
WEB_PORT=7860

# Number of API workers (for production)
WORKERS=1
```

### Logging

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log format (Python logging format)
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### Paths

```bash
# Directory for exports (results, JSON, annotated images)
EXPORTS_DIR=./ai-furniture-detector/exports

# Debug mode (verbose logging, development features)
DEBUG=false
```

## Programmatic Configuration

### Python

```python
from ai_furniture_detector.config import settings

# Access settings
print(settings.model_path)
print(settings.default_confidence)
print(settings.log_level)

# Convert to dict
config_dict = settings.to_dict()
```

### Using with Scripts

```python
from ai_furniture_detector.config import settings
from ai_furniture_detector.detector import load_model, load_image_bgr, run_detection

model = load_model(settings.model_path)
image = load_image_bgr(Path("room.jpg"))
result = run_detection(model, image, confidence=settings.default_confidence)
```

## Model Selection

The detector auto-selects the first available model in this order:

1. `yolov8x-worldv2.pt` (recommended, 130MB)
2. `yolov8m.pt` (medium, 50MB)
3. `yolov8n.pt` (nano, 10MB)

Or specify explicitly:

```bash
# Via environment variable
MODEL_PATH=/path/to/custom/yolov8.pt

# Via CLI
ai-furniture-detector --model /path/to/yolov8.pt

# Via API
curl -X POST http://localhost:8000/api/detect \
  -F "model_path=/path/to/yolov8.pt"
```

## Detection Thresholds

### Confidence Threshold (0-1)

Controls detection sensitivity. Lower = more detections (more false positives).

| Value | Behavior |
|-------|----------|
| 0.10 | Very permissive, many false positives |
| 0.25 | Default, good balance |
| 0.50 | Strict, only confident detections |
| 0.80 | Very strict, only extremely confident detections |

### IoU Threshold (0-1)

Non-Maximum Suppression. Controls overlap filtering between boxes.

| Value | Behavior |
|-------|----------|
| 0.45 | Aggressive, removes most overlaps |
| 0.70 | Default, balanced |
| 0.90 | Permissive, keeps more overlapping boxes |

## Deployment Configs

### Development

```bash
DEBUG=true
LOG_LEVEL=DEBUG
WORKERS=1
```

### Production

```bash
DEBUG=false
LOG_LEVEL=INFO
WORKERS=4
API_HOST=0.0.0.0
```

### Docker

```dockerfile
ENV MODEL_PATH=/models/yolov8x-worldv2.pt
ENV DEFAULT_CONFIDENCE=0.25
ENV LOG_LEVEL=INFO
ENV WORKERS=2
```

### Kubernetes

```yaml
env:
  - name: MODEL_PATH
    value: /models/yolov8x-worldv2.pt
  - name: DEFAULT_CONFIDENCE
    value: "0.25"
  - name: LOG_LEVEL
    value: INFO
```

## Advanced Configuration

### Custom Logging

```python
from ai_furniture_detector.logging_config import configure_logging

# Custom log level
configure_logging(level="DEBUG")

# Custom format
configure_logging(
    format_string="%(name)s: %(message)s"
)
```

### Custom Model Path

```python
from ai_furniture_detector.detector import load_model

model = load_model("/absolute/path/to/custom/yolov8.pt")
```

### Override Default Values

```python
from ai_furniture_detector.detector import run_detection

# Use custom thresholds (overrides config)
result = run_detection(
    model,
    image,
    confidence=0.35,  # Override default
    iou=0.50          # Override default
)
```

## Validation

Settings are validated on load:

```python
from ai_furniture_detector.config import Settings

try:
    settings = Settings(
        default_confidence=1.5  # Invalid: must be 0-1
    )
except ValueError as e:
    print(f"Config error: {e}")
```

## Troubleshooting

### Port already in use

App auto-finds next available port:
```bash
API_PORT=8001 ai-furniture-detector-api
WEB_PORT=7861 ai-furniture-detector-web
```

### Model not found

Model auto-downloads on first use. To manually download:
```bash
# Will download from Hugging Face
python -c "from ultralytics import YOLO; YOLO('yolov8x-worldv2.pt')"
```

### Export directory permissions

Ensure writable:
```bash
mkdir -p ./ai-furniture-detector/exports
chmod 755 ./ai-furniture-detector/exports
```

### GPU not detected

Force CPU:
```bash
export CUDA_VISIBLE_DEVICES=""
ai-furniture-detector-api
```

## See Also

- [Architecture](../ARCHITECTURE.md)
- [Contributing](../CONTRIBUTING.md)
- [API Reference](./API.md)
