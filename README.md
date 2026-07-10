# FurniVision AI

### AI-powered furniture and interior-element detection for room images

FurniVision AI analyzes interior photographs using **YOLO-World** and **OpenCV**, identifies furniture and room elements, draws annotated detections, groups results by category, and generates useful product-search links.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/guptavansh/Furnvision-ai)
[![Source Code](https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/Vansh-g-hash/furnivision-ai)

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Model-YOLO--World-00FFFF)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316)
![Deployment](https://img.shields.io/badge/Deployment-Hugging%20Face%20Spaces-FFD21E)

---

## Live Project

### Public demo

**Try FurniVision AI here:**

https://huggingface.co/spaces/guptavansh/Furnvision-ai

The hosted demo allows users to:

- Upload room and interior images
- Adjust confidence and IoU thresholds
- Run furniture detection
- View an annotated output image
- Review detected items and categories
- Open relevant Amazon India search links

### Source code

https://github.com/Vansh-g-hash/furnivision-ai

---

## Project Overview

FurniVision AI is an end-to-end computer-vision project designed to detect furniture, fixtures, decor, electronics, flooring, lighting, windows, doors, and other room elements from interior images.

The project combines a zero-shot YOLO-World model with a custom post-processing pipeline built using OpenCV.

Instead of relying only on raw model predictions, the detector also applies:

- Custom furniture and interior prompts
- Per-class confidence thresholds
- Bounding-box geometry checks
- Duplicate removal
- Context-aware false-positive filtering
- Category-based grouping
- Fallback detection for selected room elements
- Annotated labels and shopping links

The result is a more practical interior-analysis workflow than plain object detection alone.

---

## Key Features

### AI-powered detection

- Uses YOLO-World for zero-shot object detection
- Supports more than 170 custom furniture and interior prompts
- Automatically downloads a deployment-friendly model when local weights are unavailable
- Supports larger local YOLO-World weights when present

### Interior-specific processing

The project includes custom logic for detecting and filtering:

- Chairs and conference chairs
- Tables, desks, cabinets, shelves, and sideboards
- Sofas, beds, stools, benches, and storage units
- Televisions, monitors, laptops, speakers, and projectors
- Lamps, ceiling lights, and recessed lights
- Plants, paintings, mirrors, clocks, and decorative objects
- Doors, windows, glass partitions, and blinds
- Flooring, carpets, rugs, and wall panels
- HVAC vents and selected room utilities

### Smart filtering

- Removes duplicate and overlapping detections
- Uses different confidence thresholds for different classes
- Rejects detections that are too small or geometrically unlikely
- Reduces common false positives using room context
- Groups repeated items such as chairs and ceiling lights
- Separates raw detections from unique display results

### Interactive applications

- FastAPI-powered local web application
- Public Gradio demo on Hugging Face Spaces
- REST detection endpoint
- Adjustable confidence and IoU controls
- Annotated detection output
- Category and item summaries
- JSON and CSV export in the local application
- Amazon India product-search integration

---

## Technology Stack

| Area | Technology |
|---|---|
| Programming language | Python |
| Object detection | YOLO-World through Ultralytics |
| Computer vision | OpenCV |
| Numerical processing | NumPy |
| Local web backend | FastAPI |
| Local server | Uvicorn |
| Hosted interface | Gradio |
| Cloud deployment | Hugging Face Spaces |
| Model execution | PyTorch |
| Validation and configuration | Pydantic |
| Version control | Git and GitHub |

---

## How It Works

```text
User uploads an interior image
              │
              ▼
Image is validated and converted to RGB/BGR
              │
              ▼
YOLO-World runs zero-shot inference
              │
              ▼
Raw bounding boxes and confidence scores are extracted
              │
              ▼
Interior-specific filters remove weak or unlikely results
              │
              ▼
Overlapping detections are deduplicated and grouped
              │
              ▼
Fallback logic adds selected missed room elements
              │
              ▼
Labels and bounding boxes are drawn with OpenCV
              │
              ▼
Annotated image, summary, categories, and product links are returned
```

---

## Detection Pipeline

The main detection pipeline is implemented in:

```text
ai-furniture-detector/ai_furniture_detector/detector.py
```

The pipeline performs the following operations:

1. Loads a local YOLO model when available.
2. Downloads `yolov8s-worldv2.pt` as a cloud fallback.
3. Loads the custom furniture and room-element prompt list.
4. Runs object detection using the selected confidence and IoU values.
5. Normalizes model labels into consistent display names.
6. Applies class-specific confidence and minimum-area filters.
7. Removes duplicate and overlapping bounding boxes.
8. Applies context-aware false-positive cleanup.
9. Adds selected fallback detections.
10. Draws labels and returns structured detection results.

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vansh-g-hash/furnivision-ai.git
cd furnivision-ai
```

### 2. Create a virtual environment

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the local application

```bash
python3 ai-furniture-detector/ai_furniture_detector/web_app.py
```

Open:

```text
http://127.0.0.1:7860
```

The application automatically checks for an available port when running locally.

---

## Using the Local Web Application

1. Open the application in your browser.
2. Upload a JPG, PNG, JPEG, or WEBP image.
3. Adjust the confidence threshold.
4. Adjust the IoU threshold.
5. Click **Detect Furniture**.
6. Review the detected items and annotated image.
7. Filter results by category or search term.
8. Open available product-search links.
9. Export the results as JSON or CSV.
10. Download the annotated image.

---

## REST API

The local FastAPI application exposes a detection endpoint.

### Health check

```bash
curl http://127.0.0.1:7860/health
```

Example response:

```json
{
  "status": "ok",
  "app": "FurniVision AI — Furniture Detection Studio"
}
```

### Detect furniture

```bash
curl -X POST http://127.0.0.1:7860/api/detect \
  -F "image=@room.jpg" \
  -F "confidence=0.12" \
  -F "iou=0.50"
```

The response contains:

- Uploaded filename
- Original image
- Annotated image
- Image dimensions
- Unique detected items
- Total detections
- Bounding boxes
- Categories
- Confidence values
- Product-search links
- Processing time
- Model name

---

## Example API Response

```json
{
  "ok": true,
  "filename": "room.jpg",
  "unique_count": 7,
  "all_count": 11,
  "items": [
    {
      "name": "conference chair",
      "display_name": "Conference Chair",
      "category": "Furniture",
      "confidence": 0.81,
      "confidence_percent": 81.0,
      "count": 1,
      "bbox": [412, 280, 538, 510],
      "link": "https://www.amazon.in/s?k=Conference+Chair+furniture"
    }
  ],
  "summary": {
    "unique_items": 7,
    "total_detections": 11,
    "top_category": "Furniture",
    "avg_confidence": 67.4
  }
}
```

---

## Project Structure

```text
furnivision-ai/
│
├── ai-furniture-detector/
│   ├── ai_furniture_detector/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── api.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── detector.py
│   │   ├── logging_config.py
│   │   └── web_app.py
│   │
│   ├── detect.py
│   ├── pyproject.toml
│   └── TODO.md
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   ├── API.md
│   └── CONFIG.md
│
├── tests/
│   ├── __init__.py
│   └── test_detector.py
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── ARCHITECTURE.md
├── CLEANUP_SUMMARY.md
├── CONTRIBUTING.md
├── Makefile
├── README.md
├── requirements.txt
└── verify_cleanup.sh
```

---

## Core Components

### `detector.py`

Contains the main computer-vision pipeline:

- Model loading
- Custom object prompts
- Label normalization
- Confidence filtering
- Context-aware filtering
- Duplicate removal
- Bounding-box annotation
- Product-link generation
- Detection result creation

### `web_app.py`

Provides the local FastAPI web application:

- Image uploads
- Detection settings
- Browser-based interface
- REST detection endpoint
- JSON and CSV export
- Annotated image download
- Search and category filters

### `api.py`

Contains reusable API functionality for programmatic integration.

### `cli.py`

Contains command-line interface functionality.

### `config.py`

Handles application settings and environment-based configuration.

---

## Model Handling

FurniVision AI searches for available model weights in the project directory.

Supported local filenames include:

```text
yolov8x-worldv2.pt
yolov8s-worldv2.pt
yolov8m-worldv2.pt
yolov8m.pt
yolov8n.pt
```

When no local model is found, the application automatically downloads:

```text
yolov8s-worldv2.pt
```

This provides a smaller and more deployment-friendly fallback for the hosted demo.

Large model files are intentionally excluded from Git because they can be downloaded automatically.

---

## Hosted Deployment

The public demo is deployed using:

- Hugging Face Spaces
- Gradio
- ZeroGPU
- Ultralytics YOLO-World
- OpenCV headless
- PyTorch

The hosted application downloads the latest detector source from the GitHub repository when the Space starts.

This keeps the deployed demonstration connected to the main project code.

### Live deployment

https://huggingface.co/spaces/guptavansh/Furnvision-ai

---

## Current Limitations

- Zero-shot detection can still produce false positives in complex scenes.
- Detection quality depends on lighting, image resolution, camera angle, and object visibility.
- Small or partially hidden objects may not be detected.
- The first hosted inference may take longer because the Space and model may need to start.
- Amazon links currently open search pages rather than matching exact products.
- The cloud deployment uses a smaller model for better startup time and memory usage.
- Context filters are optimized mainly for office and indoor-room photographs.
- Some structural elements may require additional scene-specific tuning.

---

## Future Improvements

- Train or fine-tune a dedicated furniture-detection model
- Add segmentation masks for more accurate object boundaries
- Add room-style and interior-design classification
- Add furniture dimension estimation
- Add object tracking for video input
- Add database storage for previous detections
- Add exact product recommendation using visual similarity
- Add user accounts and saved projects
- Add batch image analysis
- Add PDF inventory reports
- Add mobile camera support
- Add 3D room understanding
- Add furniture placement and design recommendations

---

## Resume Highlights

This project demonstrates experience with:

- Computer vision
- Zero-shot object detection
- YOLO-World and Ultralytics
- OpenCV image processing
- FastAPI backend development
- Gradio application development
- REST API design
- Cloud deployment
- Model fallback handling
- Custom filtering algorithms
- Data export
- Git and GitHub workflows
- Debugging platform-specific inference issues

---

## Author

**Vansh Gupta**

- GitHub: [Vansh-g-hash](https://github.com/Vansh-g-hash)
- Live project: [FurniVision AI](https://huggingface.co/spaces/guptavansh/Furnvision-ai)

---

## Support

For issues or suggestions, open an issue in the GitHub repository:

https://github.com/Vansh-g-hash/furnivision-ai/issues

---

Built as a practical AI and computer-vision portfolio project.
