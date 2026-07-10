# AI Interior Furniture Detection System - Implementation TODO

## Plan confirmation
Gradio-based modern interactive UI will be implemented, reusing/improving the existing YOLOv8+OpenCV detection logic.

## Steps
1. Create a reusable detection module
   - Add `ai_furniture_detector/detector.py` with: model loading, detection parsing with confidence, deduplication, and annotation drawing.
2. Add category-to-color mapping + stable palette for labels/boxes.
3. Implement Gradio web app UI
   - File upload
   - Confidence slider / threshold controls
   - Run detection and show annotated image
   - Right sidebar cards (unique detections), confidence display
   - Sidebar search/filter
   - Clicking a card opens `https://www.amazon.in/s?k=<object_name>`
   - JSON export + “Save annotated image”
   - Dark mode toggle (UI toggle)
4. Wire CLI + packaging

   - Update `pyproject.toml` to include gradio and add a console script for running the web app.
   - Ensure detector CLI still works and uses shared detector module.
5. Update documentation
   - Update root `furniture-project/README.md` (or existing README) with setup + run commands.
6. Smoke test on provided sample image (`room2.jpg`)
   - Verify bounding boxes, labels, sidebar dedupe, clicks, JSON download, and saving annotated image.

