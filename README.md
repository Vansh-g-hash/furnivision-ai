# AI Interior Furniture Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

A production-grade, AI-powered furniture detection system using YOLOv8 World and OpenCV. Detects and categorizes furniture and interior elements in room images with clickable shopping links.

## ✨ Features

- **Zero-shot Detection**: YOLOv8 World model detects 35+ furniture and interior items
- **Interactive UIs**: 
  - Modern Gradio web interface with dark mode
  - Native OpenCV window with clickable boxes
  - REST API with Swagger documentation
- **Smart Deduplication**: Shows one best detection per item type
- **Shopping Integration**: One-click links to Amazon India product search
- **Export Options**: Save annotated images and JSON reports
- **Production Ready**:
  - Comprehensive logging and error handling
  - Type hints throughout
  - Extensive tests and CI/CD
  - Docker-ready
  - Configuration management

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- macOS, Linux, or Windows

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-furniture-detector.git
cd ai-furniture-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with all dependencies
make install-dev

# (Optional) Install pre-commit hooks
make pre-commit-install
```

### Run Web UI

```bash
make run-web
```

Open http://127.0.0.1:7860 in your browser.

### Run CLI

```bash
make run-cli
# or with custom image:
ai-furniture-detector --source path/to/image.jpg --confidence 0.25
```

### Run API

```bash
make run-api
```

API docs: http://127.0.0.1:8000/docs

## 📚 Documentation

- [Architecture](./ARCHITECTURE.md) - System design and data flow
- [Contributing](./CONTRIBUTING.md) - Development guidelines
- [API Guide](./docs/API.md) - REST API reference
- [Configuration](./docs/CONFIG.md) - Environment variables and settings

## 💻 Usage

### Web UI

1. Upload an interior image
2. Adjust confidence/IoU thresholds if needed
3. Click "Run AI Detection"
4. Browse detected items in sidebar
5. Click items to open Amazon search
6. Export JSON or save annotated image

### CLI

```bash
ai-furniture-detector \
  --source room.jpg \
  --confidence 0.25 \
  --output results.jpg \
  --output-json results.json
```

**Options:**
- `--model`: Path to YOLO weights (default: auto-detect)
- `--confidence`: Detection threshold 0-1 (default: 0.25)
- `--iou`: NMS threshold 0-1 (default: 0.70)
- `--no-show`: Skip OpenCV preview
- `-v, --verbose`: Enable debug logging

Interactive controls:
- Click bounding boxes → Open Amazon search
- Type letters → Filter sidebar
- Backspace → Delete filter character
- R → Reset filter
- Q or ESC → Quit

### REST API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Detect furniture
curl -X POST http://127.0.0.1:8000/api/detect \
  -F "image=@room.jpg" \
  -F "confidence=0.25" \
  -F "iou=0.45" \
  -F "include_image=true"

# Get product link
curl "http://127.0.0.1:8000/api/product-link?name=office+chair"
```

See full [API documentation](./docs/API.md).

## 🏗️ Project Structure

```
ai-furniture-detector/
├── ai_furniture_detector/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Configuration management
│   ├── detector.py           # Core detection pipeline
│   ├── logging_config.py     # Logging setup
│   ├── cli.py                # Command-line interface
│   ├── api.py                # FastAPI backend
│   └── web_app.py            # Gradio web UI
├── tests/                    # Unit tests
├── .github/workflows/        # CI/CD pipelines
├── docs/                     # Documentation
├── Makefile                  # Development tasks
├── pyproject.toml            # Dependencies and config
├── ARCHITECTURE.md           # System design
├── CONTRIBUTING.md           # Dev guidelines
└── README.md                 # This file
```

## 🛠️ Development

### Code Quality

```bash
# Lint code
make lint

# Format code
make format

# Run tests with coverage
make test-cov
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_detector.py::TestDetectedItem -v

# Generate coverage report
pytest tests/ --cov=ai_furniture_detector --cov-report=html
```

### Pre-commit Hooks

```bash
# Install hooks
make pre-commit-install

# Run manually
pre-commit run --all-files
```

## 📦 Installation Options

### With Development Tools

```bash
pip install -e ".[dev]"
```

### Docker

```bash
docker build -t furniture-detector .
docker run -p 8000:8000 furniture-detector ai-furniture-detector-api
```

### From Source

```bash
git clone https://github.com/yourusername/ai-furniture-detector.git
cd ai-furniture-detector
pip install -e .
```

## 🔧 Configuration

Create `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

**Key settings:**
- `MODEL_PATH`: Path to YOLO weights (auto-detected if not set)
- `DEFAULT_CONFIDENCE`: Detection threshold (0.25)
- `DEFAULT_IOU`: NMS threshold (0.70)
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, etc.)
- `API_PORT`: API server port (8000)
- `WEB_PORT`: Web UI port (7860)

See [Configuration Guide](./docs/CONFIG.md) for all options.

## 📊 Detected Objects

**Furniture**: Chair, Office Chair, Visitor Chair, Sofa, Desk, Table, Center Table, Coffee Table, Cabinet, Shelf, Wall Shelf, Pillow

**Flooring**: Floor Mat, Carpet, Rug

**Lighting**: Ceiling Light, Wall Light, Lamp

**Electronics**: Monitor, Screen, TV

**Doors & Windows**: Door, Window, Glass Door, Sliding Glass Door, Wooden Door

**Decor**: Plant, Potted Plant, Plant Stand, Painting, Wall Painting, Hanging Painting, Clock, Vase

**Other**: Pen Holder, and more via YOLOv8 World's zero-shot capabilities

## 🚨 Performance

- **Detection Time**: ~500-1500ms per image (depends on size and model)
- **Memory**: ~2-4GB for inference
- **Model Sizes**:
  - YOLOv8x-worldv2: 130MB (recommended)
  - YOLOv8m: 50MB
  - YOLOv8n: 10MB

### Optimization Tips

1. Use smaller image inputs (< 1920px)
2. Reduce confidence threshold for faster processing
3. Run API with multiple workers: `WORKERS=4`
4. Use GPU if available

## 🐛 Troubleshooting

### Model not found

```bash
# Download will be attempted automatically
# Or manually download from Hugging Face and set MODEL_PATH
```

### Port already in use

```bash
# App auto-finds next available port, or specify manually:
API_PORT=8001 ai-furniture-detector-api
```

### GPU support

```bash
# PyTorch will auto-detect CUDA if available
# To force CPU:
export CUDA_VISIBLE_DEVICES=""
```

### Memory issues

Use smaller model:
```bash
ai-furniture-detector --model yolov8n.pt
```

## 📝 API Examples

### Python

```python
import requests
from pathlib import Path

# Upload image
with open("room.jpg", "rb") as f:
    files = {"image": f}
    data = {"confidence": 0.25, "include_image": False}
    response = requests.post(
        "http://127.0.0.1:8000/api/detect",
        files=files,
        data=data
    )
    result = response.json()
    print(f"Found {result['unique_count']} unique items")
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('confidence', 0.25);

const response = await fetch('http://127.0.0.1:8000/api/detect', {
    method: 'POST',
    body: formData
});
const result = await response.json();
console.log(`Found ${result.unique_count} items`);
```

## 📋 Requirements

- Python 3.10+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- FastAPI + Uvicorn
- Gradio

See `pyproject.toml` for complete dependency list.

## 📄 License

MIT License - see [LICENSE](./LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 🔗 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Gradio Docs](https://gradio.app/)
- [OpenCV Tutorials](https://docs.opencv.org/)

## ⭐ Star History

If you find this useful, please consider starring! ⭐

## 📞 Support

- 📖 [Documentation](./docs/)
- 🐛 [Report Issues](https://github.com/yourusername/ai-furniture-detector/issues)
- 💬 [Discussions](https://github.com/yourusername/ai-furniture-detector/discussions)

---

**Made with ❤️ for furniture enthusiasts and developers.**

- Click any bounding box to open the Amazon product search.
- Click any sidebar card to open the same product search.
- Type letters to filter the sidebar.
- Backspace deletes filter text.
- `R` resets the filter.
- `Q` or `Esc` quits.

## Save Outputs Without Preview

```bash
ai-furniture-detector \
  --source ai-furniture-detector/room2.jpg \
  --output ai-furniture-detector/exports/annotated_latest.jpg \
  --output-json ai-furniture-detector/exports/detections_latest.json \
  --no-show
```

## Detection Tuning

Small interior objects such as lights, clocks, floor mats, and pen holders often need a lower threshold:

```bash
ai-furniture-detector --confidence 0.025 --iou 0.45
```

Use a slightly higher threshold, such as `0.08`, if the model returns too many weak detections.

## Architecture for Expansion

- `detector.py` is framework-independent and can be reused by FastAPI, Flask, Streamlit, or a mobile backend.
- `DetectedItem.to_dict()` already includes name, confidence, category, coordinates, and product link for database storage.
- `run_detection()` returns all bounding boxes plus unique sidebar/product entries, which keeps UI and inventory workflows separate.
- Product links currently target Amazon India, but `build_amazon_link()` can be swapped for a marketplace API, inventory database, or recommendation engine.
- Category labels are centralized for future analytics, product ranking, and AI design suggestions.

## Troubleshooting

- If the model file is missing, confirm `yolov8x-worldv2.pt` exists in the project root or pass `--model /path/to/model.pt`.
- If OpenCV windows do not appear on macOS, run the command from Terminal with the virtual environment activated.
- If the web UI package is missing, run `pip install -r requirements.txt` again inside the virtual environment.
- If detection is slow, try `--model yolov8n.pt` for faster but less interior-specific detection.
