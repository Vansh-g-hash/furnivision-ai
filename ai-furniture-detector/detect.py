from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QEvent
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QCursor,
    QPainter,
    QKeySequence,
    QShortcut,
    QPen,
    QBrush,
    QColor,
)
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QScrollArea,
    QFrame,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsTextItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QSlider,
    QFileDialog,
    QMessageBox,
)

ROOT = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT / "ai_furniture_detector"

if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Use explicit package import so static analyzers (Pylance) can resolve it.
from ai_furniture_detector.detector import (
    FurnitureDetector,
    build_amazon_link,
    category_for_item as core_category_for_item,
    display_name as core_display_name,
)

DISPLAY_NAME_MAP = {
    "chair": "Conference Chair",
    "conference chair": "Conference Chair",
    "office chair": "Office Chair",
    "dining chair": "Chair",
    "armchair": "Armchair",
    "stool": "Stool",
    "bench": "Bench",
    "sofa": "Sofa",
    "couch": "Sofa",
    "table": "Conference Table",
    "conference table": "Conference Table",
    "center table": "Conference Table",
    "coffee table": "Conference Table",
    "linear table": "Conference Table",
    "side table": "Conference Table",
    "desk": "Conference Table",
    "work desk": "Work Desk",
    "television": "Television / Screen",
    "television screen": "Television / Screen",
    "tv": "Television / Screen",
    "monitor": "Monitor / Screen",
    "computer monitor": "Monitor / Screen",
    "screen": "Screen",
    "laptop": "Laptop",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "speaker": "Soundbar / Speaker",
    "soundbar": "Soundbar / Speaker",
    "soundbar speaker": "Soundbar / Speaker",
    "plant": "Plant",
    "potted plant": "Plant",
    "indoor plant": "Plant",
    "door": "Door",
    "a door": "Door",
    "glass door": "Glass Door",
    "ac vent": "AC Vent",
    "ac": "Air Conditioner",
    "glass door": "Glass Door",
    "window": "Window",
    "glass window": "Glass Window",
    "glass partition": "Glass Partition",
    "lamp": "Lamp",
    "table lamp": "Table Lamp",
    "floor lamp": "Floor Lamp",
    "ceiling light": "Ceiling Light",
    "recessed light": "Ceiling Light",
    "downlight": "Ceiling Light",
    "ceiling spotlight": "Ceiling Light",
    "wall light": "Wall Light",
    "blind": "Window Blind",
    "window blind": "Window Blind",
    "roller blind": "Window Blind",
    "vertical blind": "Window Blind",    "ceiling": "Ceiling",
    "ceiling pattern": "Ceiling Pattern",
    "ceiling panel": "Ceiling Panel",
    "ceiling design": "Ceiling Design",
    "wooden ceiling": "Wooden Ceiling",
    "wall": "Wall",
    "floor": "Floor",
    "control panel": "Control Panel",
    "ac vent": "AC Vent",
    "table cable port": "Table Cable Port",
    "phone": "Landline Phone",
    "telephone": "Landline Phone",
    "landline phone": "Landline Phone",
    "landline telephone": "Landline Phone",
    "desk phone": "Landline Phone",
    "office phone": "Landline Phone",
    "wall art": "Wall Art",
    "picture frame": "Picture Frame",
    "painting": "Painting",
    "mirror": "Mirror",
    "clock": "Clock",
    "vase": "Vase",
    "bottle": "Bottle",
    "mug": "Mug",
    "cup": "Cup",
    "book": "Book",
    "notebook": "Notebook",
    "whiteboard": "Whiteboard",
    "trash bin": "Trash Bin",
    "dustbin": "Trash Bin",
    "rug": "Rug Floor",
    "carpet": "Carpet Floor",
    "floor mat": "Floor Mat",
    "wooden wall panel": "Wooden Wall Panel",
    "wood slat wall": "Wood Slat Wall",
    "tv wall panel": "Wooden TV Wall Panel",
    "wooden backdrop": "Wooden Backdrop",
    "wall paneling": "Wall Paneling",
    "decorative wall panel": "Decorative Wall Panel",
}

CATEGORY_FIX = {
    "chair": "Furniture",
    "conference chair": "Furniture",
    "office chair": "Furniture",
    "dining chair": "Furniture",
    "armchair": "Furniture",
    "stool": "Furniture",
    "table": "Furniture",
    "conference table": "Furniture",
    "desk": "Furniture",
    "bench": "Furniture",
    "sofa": "Furniture",
    "couch": "Furniture",
    "television": "Electronics",
    "television screen": "Electronics",
    "tv": "Electronics",
    "monitor": "Electronics",
    "laptop": "Electronics",
    "landline phone": "Electronics / Utility",
    "phone": "Electronics / Utility",
    "telephone": "Electronics / Utility",
    "keyboard": "Electronics",
    "mouse": "Electronics",
    "speaker": "Electronics",
    "soundbar": "Electronics",
    "plant": "Decor",
    "potted plant": "Decor",
    "indoor plant": "Decor",
    "wall art": "Decor",
    "picture frame": "Decor",
    "painting": "Decor",
    "mirror": "Decor",
    "clock": "Decor",
    "vase": "Decor",
    "lamp": "Lighting",
    "table lamp": "Lighting",
    "floor lamp": "Lighting",
    "ceiling light": "Lighting",
    "recessed light": "Lighting",
    "downlight": "Lighting",
    "ceiling spotlight": "Lighting",
    "recessed light": "Lighting",
    "wall light": "Lighting",
    "blind": "Doors & Windows",
    "window blind": "Doors & Windows",
    "roller blind": "Doors & Windows",
    "vertical blind": "Doors & Windows",
    "door": "Doors & Windows",
    "glass door": "Doors & Windows",
    "window": "Doors & Windows",
    "glass window": "Doors & Windows",
    "glass partition": "Doors & Windows",
    "floor": "Flooring & Textiles",
    "rug": "Flooring & Textiles",
    "carpet": "Flooring & Textiles",
    "floor mat": "Flooring & Textiles",
    "wall": "Structure",
    "ceiling": "Structure",
    "ceiling pattern": "Structure",
    "ceiling panel": "Structure",
    "ceiling design": "Structure",
    "wooden ceiling": "Structure",
    "wooden wall panel": "Structure",
    "wood slat wall": "Structure",
    "tv wall panel": "Structure",
    "wooden backdrop": "Structure",
    "wall paneling": "Structure",
    "decorative wall panel": "Structure",
    "ac vent": "HVAC",    "control panel": "Electronics / Utility",
    "cup": "Accessories",
    "mug": "Accessories",
    "bottle": "Accessories",
    "book": "Accessories",
    "notebook": "Accessories",
    "whiteboard": "Accessories",
    "trash bin": "Accessories",
}


SURFACE_SAMPLE_HIGHLIGHT_CLASSES = {
    "floor",
    "carpet",
    "rug",
    "floor mat",
    "ceiling",
    "ceiling pattern",
    "ceiling panel",
    "ceiling design",
    "wooden ceiling",
    "recessed light",
    "downlight",
    "ceiling spotlight",
    "ac vent",    "wall",
    "wooden wall panel",
    "wood slat wall",
    "tv wall panel",
    "wooden backdrop",
    "wall paneling",
    "decorative wall panel",
}


RECT_FULL_HIGHLIGHT_CLASSES = set()

# Doors/windows/glass panels are often detected as a very large wall-sized region.
# We show a small accurate sample patch instead of covering the whole scene/table/chairs.
PORTAL_SAMPLE_HIGHLIGHT_CLASSES = {
    "window",
    "glass window",
    "door",
    "glass door",
    "glass partition",
    "blind",
    "window blind",    "roller blind",
    "vertical blind",
}

# The model sometimes adds a second false table/desk tag on the same conference table.
# We display the real table detection as Conference Table and hide these duplicate desk detections.
HIDE_FROM_UI_LABELS = set()  


def get_value(item, key, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def get_name(item) -> str:
    return str(get_value(item, "name", "Unknown")).lower().strip()


def get_display_name(item) -> str:
    name = get_name(item)
    mapped = core_display_name(name)
    if mapped != name.title():
        return mapped
    return DISPLAY_NAME_MAP.get(name, mapped)


def get_bbox(item) -> list[int]:
    bbox = get_value(item, "bbox", [0, 0, 0, 0])
    return [int(v) for v in bbox]


def get_category(item) -> str:
    name = get_name(item)
    category = core_category_for_item(name)
    if category != "Other":
        return category
    return CATEGORY_FIX.get(name, str(get_value(item, "category", "Other")))


def get_confidence(item) -> float:
    return float(get_value(item, "confidence", 0))


def get_link(item) -> str:
    link = get_value(item, "link", "")
    if link:
        return str(link)
    return build_amazon_link(get_name(item))


def representative_score(det):
    """
    Pick the best object to represent a grouped UI item.

    For furniture, prefer the lowest visible detection so hover/click behavior
    points to the actual chair/table area instead of a false high detection.
    """
    x1, y1, x2, y2 = get_bbox(det)
    area = max(1, x2 - x1) * max(1, y2 - y1)

    if get_category(det) == "Furniture":
        return float(y2), float(area), float(get_confidence(det))

    return float(get_confidence(det)), float(area), float(y2)


def unique_items(detections):
    grouped = {}

    for det in detections:
        name = get_name(det)

        if not name or name in HIDE_FROM_UI_LABELS:
            continue

        grouped.setdefault(name, []).append(det)

    final_items = []

    for name, group in grouped.items():
        representative = max(group, key=representative_score)

        try:
            representative.count = len(group)
        except Exception:
            pass

        final_items.append(representative)

    return sorted(final_items, key=get_confidence, reverse=True)


def clamp_bbox(bbox, image_w, image_h):
    x1, y1, x2, y2 = [int(v) for v in bbox]

    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    if x2 <= x1:
        x2 = min(image_w - 1, x1 + 1)

    if y2 <= y1:
        y2 = min(image_h - 1, y1 + 1)

    return [x1, y1, x2, y2]


def tightened_bbox(item, image_w: int, image_h: int) -> list[int]:
    x1, y1, x2, y2 = clamp_bbox(get_bbox(item), image_w, image_h)

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    name = get_name(item)

    if name in SURFACE_SAMPLE_HIGHLIGHT_CLASSES or name in RECT_FULL_HIGHLIGHT_CLASSES:
        pad_x = int(w * 0.008)
        pad_y = int(h * 0.008)
    elif name in {"television", "tv", "monitor", "laptop", "speaker", "soundbar"}:
        pad_x = int(w * 0.018)
        pad_y = int(h * 0.025)
    elif name in {"bottle", "cup", "mug", "vase", "clock", "book", "notebook"}:
        pad_x = int(w * 0.045)
        pad_y = int(h * 0.045)
    else:
        pad_x = int(w * 0.030)
        pad_y = int(h * 0.030)

    nx1 = x1 + pad_x
    ny1 = y1 + pad_y
    nx2 = x2 - pad_x
    ny2 = y2 - pad_y

    if nx2 <= nx1:
        nx1, nx2 = x1, x2

    if ny2 <= ny1:
        ny1, ny2 = y1, y2

    return clamp_bbox([nx1, ny1, nx2, ny2], image_w, image_h)


def surface_sample_bbox(item, image_w: int, image_h: int) -> list[int]:
    """
    For very large things like floor/carpet/ceiling/wall, we do NOT highlight
    the whole huge detection area. We show only one clean sample patch.
    """
    x1, y1, x2, y2 = tightened_bbox(item, image_w, image_h)
    name = get_name(item)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Size of sample patch.
    patch_w = int(min(max(bw * 0.26, 90), 260))
    patch_h = int(min(max(bh * 0.22, 55), 160))

    if name in {"floor", "carpet", "rug", "floor mat"}:
        # Bottom-right-ish patch, but kept inside bbox.
        px2 = x2 - int(bw * 0.05)
        py2 = y2 - int(bh * 0.05)
        px1 = px2 - patch_w
        py1 = py2 - patch_h

    elif name in {"ceiling", "ceiling pattern", "ceiling panel", "ceiling design", "wooden ceiling"}:
        # Small top-center patch for ceiling.
        cx = (x1 + x2) // 2
        px1 = cx - patch_w // 2
        py1 = y1 + int(bh * 0.10)
        px2 = px1 + patch_w
        py2 = py1 + patch_h

    else:
        # Walls / panels: show a central sample patch.
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        px1 = cx - patch_w // 2
        py1 = cy - patch_h // 2
        px2 = px1 + patch_w
        py2 = py1 + patch_h

    px1 = max(x1, min(x2 - 1, px1))
    py1 = max(y1, min(y2 - 1, py1))
    px2 = max(px1 + 1, min(x2, px2))
    py2 = max(py1 + 1, min(y2, py2))

    return clamp_bbox([px1, py1, px2, py2], image_w, image_h)


def portal_sample_bbox(item, image_w: int, image_h: int) -> list[int]:
    """
    For glass doors/windows, avoid a massive rectangle over the whole glass wall.
    Use a clean vertical sample patch that stays inside the detected region.
    """
    x1, y1, x2, y2 = tightened_bbox(item, image_w, image_h)
    name = get_name(item)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # If YOLO detects a full glass wall as a door, highlight only the likely door strip.
    if name in {"door", "glass door"} and bw > bh * 0.55:
        patch_w = int(min(max(bw * 0.20, 90), 210))
        px1 = x1 + int(bw * 0.025)
        px2 = px1 + patch_w
        py1 = y1 + int(bh * 0.06)
        py2 = y2 - int(bh * 0.03)
    else:
        patch_w = int(min(max(bw * 0.35, 90), 260))
        patch_h = int(min(max(bh * 0.55, 120), 360))
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        px1 = cx - patch_w // 2
        px2 = px1 + patch_w
        py1 = cy - patch_h // 2
        py2 = py1 + patch_h

    px1 = max(x1, min(x2 - 1, px1))
    py1 = max(y1, min(y2 - 1, py1))
    px2 = max(px1 + 1, min(x2, px2))
    py2 = max(py1 + 1, min(y2, py2))

    return clamp_bbox([px1, py1, px2, py2], image_w, image_h)


def rgba_to_pixmap(rgba: np.ndarray) -> QPixmap:
    h, w, channels = rgba.shape
    bytes_per_line = channels * w

    qimage = QImage(
        rgba.data,
        w,
        h,
        bytes_per_line,
        QImage.Format.Format_RGBA8888,
    )

    return QPixmap.fromImage(qimage.copy())

def upscale_small_image_for_detection(image_bgr: np.ndarray, min_width: int = 1280) -> np.ndarray:
    """
    Small uploaded images look bad in the PyQt viewer because the app must zoom
    them a lot, and the fixed OpenCV labels become huge on screen.

    This safely upscales only tiny images before detection + drawing. Big images
    are untouched, so your existing working outputs do not change.
    """
    h, w = image_bgr.shape[:2]

    if w >= min_width:
        return image_bgr

    scale = min_width / max(1, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


class SAM2Highlighter:
    def __init__(self, image_rgb: np.ndarray, model_name: str = "sam2_b.pt"):
        self.image_rgb = image_rgb
        self.image_h, self.image_w = image_rgb.shape[:2]
        self.model_name = model_name
        self.model = None
        self.ready = False
        self.cache: dict[tuple[str, tuple[int, int, int, int]], tuple[QPixmap, int, int] | None] = {}
        self.load_model()

    def load_model(self):
        try:
            from ultralytics import SAM

            print(f"Loading SAM2 model: {self.model_name}")
            self.model = SAM(self.model_name)
            self.ready = True
            print("SAM2 loaded successfully.")

        except Exception as exc:
            print(f"SAM2 failed to load. Using clean fallback. Error: {exc}")
            self.model = None
            self.ready = False

    def mask_for_item(self, item) -> tuple[QPixmap, int, int] | None:
        name = get_name(item)

        if (
            name in SURFACE_SAMPLE_HIGHLIGHT_CLASSES
            or name in PORTAL_SAMPLE_HIGHLIGHT_CLASSES
            or name in RECT_FULL_HIGHLIGHT_CLASSES
        ):
            return None

        bbox = clamp_bbox(get_bbox(item), self.image_w, self.image_h)
        key = (name, tuple(bbox))

        if key in self.cache:
            return self.cache[key]

        result = self.create_mask_pixmap(bbox)
        self.cache[key] = result
        return result

    def create_mask_pixmap(self, bbox: list[int]) -> tuple[QPixmap, int, int] | None:
        if not self.ready or self.model is None:
            return None

        x1, y1, x2, y2 = clamp_bbox(bbox, self.image_w, self.image_h)

        try:
            results = self.model.predict(
                source=self.image_rgb,
                bboxes=[[x1, y1, x2, y2]],
                retina_masks=True,
                verbose=False,
            )

            if not results:
                return None

            result = results[0]

            if result.masks is None:
                return None

            raw_masks = result.masks.data

            try:
                raw_masks = raw_masks.cpu().numpy()
            except Exception:
                raw_masks = np.asarray(raw_masks)

            if raw_masks is None or len(raw_masks) == 0:
                return None

            best_mask = self.pick_best_mask(raw_masks, bbox)

            if best_mask is None:
                return None

            return self.mask_to_pixmap(best_mask)

        except Exception as exc:
            print(f"SAM2 mask failed for bbox {bbox}. Fallback used. Error: {exc}")
            return None

    def pick_best_mask(self, raw_masks, bbox: list[int]) -> np.ndarray | None:
        x1, y1, x2, y2 = clamp_bbox(bbox, self.image_w, self.image_h)

        box_mask = np.zeros((self.image_h, self.image_w), dtype=bool)
        box_mask[y1:y2, x1:x2] = True

        box_area = max(1, int(box_mask.sum()))

        best_score = 0.0
        best_mask = None

        for mask in raw_masks:
            mask = mask.astype(np.float32)

            if mask.shape[:2] != (self.image_h, self.image_w):
                mask = cv2.resize(
                    mask,
                    (self.image_w, self.image_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            binary = mask > 0.5

            if binary.sum() < 30:
                continue

            mask_area = max(1, int(binary.sum()))
            overlap = int((binary & box_mask).sum())

            if overlap <= 0:
                continue

            overlap_in_box = overlap / box_area
            overlap_in_mask = overlap / mask_area
            size_ratio = mask_area / box_area

            if size_ratio > 1.60:
                continue

            if overlap_in_mask < 0.52:
                continue

            score = overlap_in_box * 0.55 + overlap_in_mask * 0.45

            if score > best_score:
                best_score = score
                best_mask = binary

        if best_mask is None or best_score < 0.28:
            return None

        expand_x = int((x2 - x1) * 0.025)
        expand_y = int((y2 - y1) * 0.025)

        ex1 = max(0, x1 - expand_x)
        ey1 = max(0, y1 - expand_y)
        ex2 = min(self.image_w, x2 + expand_x)
        ey2 = min(self.image_h, y2 + expand_y)

        clip = np.zeros_like(best_mask)
        clip[ey1:ey2, ex1:ex2] = True

        final_mask = best_mask & clip

        if final_mask.sum() < 30:
            return None

        final_mask_uint8 = final_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            final_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        cleaned = np.zeros_like(final_mask_uint8)
        min_area = max(40, box_area * 0.012)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(cleaned, [contour], -1, 255, -1)

        if cleaned.sum() < 30:
            return None

        return cleaned > 0

    def mask_to_pixmap(self, mask: np.ndarray) -> tuple[QPixmap, int, int] | None:
        ys, xs = np.where(mask)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = max(0, int(xs.min()) - 14)
        y1 = max(0, int(ys.min()) - 14)
        x2 = min(self.image_w, int(xs.max()) + 15)
        y2 = min(self.image_h, int(ys.max()) + 15)

        cropped_mask = mask[y1:y2, x1:x2].astype(np.uint8) * 255

        if cropped_mask.size == 0:
            return None

        kernel_small = np.ones((3, 3), dtype=np.uint8)
        kernel_medium = np.ones((5, 5), dtype=np.uint8)

        cropped_mask = cv2.morphologyEx(cropped_mask, cv2.MORPH_CLOSE, kernel_medium)
        cropped_mask = cv2.morphologyEx(cropped_mask, cv2.MORPH_OPEN, kernel_small)

        contours, _ = cv2.findContours(
            cropped_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None

        cleaned = np.zeros_like(cropped_mask)
        min_area = max(80, cropped_mask.shape[0] * cropped_mask.shape[1] * 0.002)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            epsilon = 0.010 * cv2.arcLength(contour, True)
            smooth = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(cleaned, [smooth], -1, 255, -1)

        if cleaned.sum() < 40:
            return None

        h, w = cleaned.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        fill = cv2.GaussianBlur(cleaned, (7, 7), 0)
        outer = cv2.dilate(cleaned, np.ones((11, 11), dtype=np.uint8), iterations=1)
        outer = cv2.GaussianBlur(outer, (21, 21), 0)
        edge = cv2.morphologyEx(cleaned, cv2.MORPH_GRADIENT, np.ones((5, 5), dtype=np.uint8))
        edge = cv2.GaussianBlur(edge, (5, 5), 0)

        cyan = (56, 189, 248)
        bright_cyan = (125, 211, 252)

        glow_alpha = np.clip(outer * 0.34, 0, 85).astype(np.uint8)

        rgba[:, :, 0] = cyan[0]
        rgba[:, :, 1] = cyan[1]
        rgba[:, :, 2] = cyan[2]
        rgba[:, :, 3] = glow_alpha

        fill_alpha = np.clip(fill * 0.18, 0, 42).astype(np.uint8)
        inside = cleaned > 0

        rgba[inside, 0] = cyan[0]
        rgba[inside, 1] = cyan[1]
        rgba[inside, 2] = cyan[2]
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], fill_alpha)

        edge_alpha = np.clip(edge * 0.65, 0, 135).astype(np.uint8)
        edge_pixels = edge > 0

        rgba[edge_pixels, 0] = bright_cyan[0]
        rgba[edge_pixels, 1] = bright_cyan[1]
        rgba[edge_pixels, 2] = bright_cyan[2]
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], edge_alpha)

        return rgba_to_pixmap(rgba), x1, y1


class ProTrackpadImageViewer(QGraphicsView):
    zoomChanged = pyqtSignal(float)

    def __init__(self, display_rgb, original_rgb, items, sam_model_name: str):
        super().__init__()

        self.display_rgb = display_rgb
        self.original_rgb = original_rgb
        self.items = items
        self.image_h, self.image_w = display_rgb.shape[:2]

        self.min_zoom = 0.10
        self.max_zoom = 8.00
        self.zoom_level = 1.0
        self.fit_zoom_level = 1.0

        self.hovered_item = None
        self.highlight_item = None
        self.highlight_label_item: QGraphicsTextItem | None = None

        # Click areas for the permanent labels already drawn on the image.
        # Those labels are pixels inside output_bgr, not real Qt widgets, so we
        # recreate their rectangles here and map each rectangle back to its item.
        self.static_label_click_boxes = self.build_static_label_click_boxes(items)

        self.segmenter = SAM2Highlighter(original_rgb, sam_model_name)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        image_h, image_w, channels = display_rgb.shape
        bytes_per_line = channels * image_w

        qimage = QImage(
            display_rgb.data,
            image_w,
            image_h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )

        self.pixmap = QPixmap.fromImage(qimage.copy())
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)

        self.scene_obj.addItem(self.pixmap_item)
        self.setSceneRect(QRectF(self.pixmap.rect()))

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setInteractive(True)

        self.viewport().setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.setStyleSheet(
            """
            QGraphicsView {
                background-color: #05070d;
                border: none;
            }

            QScrollBar:vertical {
                background: #080b12;
                width: 12px;
                margin: 0;
            }

            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 6px;
                min-height: 34px;
            }

            QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }

            QScrollBar:horizontal {
                background: #080b12;
                height: 12px;
                margin: 0;
            }

            QScrollBar::handle:horizontal {
                background: #334155;
                border-radius: 6px;
                min-width: 34px;
            }

            QScrollBar::handle:horizontal:hover {
                background: #38bdf8;
            }
            """
        )

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_to_screen()

    def current_zoom_percent(self) -> int:
        return int(round(self.zoom_level * 100))

    def set_zoom(self, zoom_value: float, anchor_under_mouse: bool = True):
        zoom_value = max(self.min_zoom, min(self.max_zoom, float(zoom_value)))
        old_anchor = self.transformationAnchor()

        if anchor_under_mouse:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        else:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self.resetTransform()
        self.scale(zoom_value, zoom_value)
        self.zoom_level = zoom_value
        self.zoomChanged.emit(self.zoom_level)
        self.setTransformationAnchor(old_anchor)

    def zoom_by_factor(self, factor: float):
        self.set_zoom(self.zoom_level * factor, anchor_under_mouse=True)

    def zoom_in(self):
        self.zoom_by_factor(1.18)

    def zoom_out(self):
        self.zoom_by_factor(1 / 1.18)

    def actual_size(self):
        self.set_zoom(1.0, anchor_under_mouse=False)
        self.centerOn(self.pixmap_item)

    def fit_to_screen(self):
        if self.pixmap.isNull():
            return

        self.resetTransform()

        view_rect = self.viewport().rect()
        scene_rect = self.pixmap_item.boundingRect()

        if view_rect.width() <= 0 or view_rect.height() <= 0:
            return

        x_ratio = view_rect.width() / scene_rect.width()
        y_ratio = view_rect.height() / scene_rect.height()

        fit_scale = min(x_ratio, y_ratio) * 0.96
        fit_scale = max(self.min_zoom, min(self.max_zoom, fit_scale))

        self.fit_zoom_level = fit_scale
        self.set_zoom(fit_scale, anchor_under_mouse=False)
        self.centerOn(self.pixmap_item)

    def event(self, event):
        if event.type() == QEvent.Type.NativeGesture:
            try:
                gesture_type = event.gestureType()

                if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                    value = float(event.value())
                    factor = 1.0 + value
                    factor = max(0.85, min(1.15, factor))
                    self.zoom_by_factor(factor)
                    event.accept()
                    return True

                if gesture_type in {
                    Qt.NativeGestureType.BeginNativeGesture,
                    Qt.NativeGestureType.EndNativeGesture,
                }:
                    event.accept()
                    return True
            except Exception:
                pass

        return super().event(event)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()

            if delta > 0:
                self.zoom_by_factor(1.12)
            elif delta < 0:
                self.zoom_by_factor(1 / 1.12)

            event.accept()
            return

        super().wheelEvent(event)


    def label_text_for_static_tag(self, item) -> str:
        text = get_display_name(item)
        count = int(get_value(item, "count", 1))

        if count > 1:
            text = f"{text} x{count}"

        return text

    def clamp_label_position_for_static_tag(
        self,
        label_x: int,
        label_y: int,
        label_w: int,
        label_h: int,
    ) -> tuple[int, int]:
        label_x = max(8, min(self.image_w - label_w - 8, int(label_x)))
        label_y = max(8, min(self.image_h - label_h - 8, int(label_y)))
        return label_x, label_y

    def label_position_for_static_tag(
        self,
        item,
        label_w: int,
        label_h: int,
    ) -> tuple[int, int]:
        # This mirrors detector.py::_label_position_attached_to_object().
        x1, y1, x2, y2 = clamp_bbox(get_bbox(item), self.image_w, self.image_h)
        name = get_name(item)
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        cx = (x1 + x2) // 2

        flooring_keys = {"floor", "wooden floor", "marble floor", "tile floor", "carpet"}

        if name in {"glass door", "door", "window", "glass window", "glass partition"}:
            return self.clamp_label_position_for_static_tag(x1 + 8, y1 + 8, label_w, label_h)

        if name in flooring_keys:
            label_x = min(x2 - label_w - 8, max(x1 + 8, cx - label_w // 2))
            label_y = max(y1 + 8, y2 - label_h - 10)
            return self.clamp_label_position_for_static_tag(label_x, label_y, label_w, label_h)

        if name in {"ceiling", "ceiling panel", "ceiling panels", "false ceiling"}:
            label_x = min(x2 - label_w - 8, max(x1 + 8, cx - label_w // 2))
            label_y = y1 + 8
            return self.clamp_label_position_for_static_tag(label_x, label_y, label_w, label_h)

        if box_w >= label_w + 18 and box_h >= label_h + 18:
            return self.clamp_label_position_for_static_tag(x1 + 8, y1 + 8, label_w, label_h)

        candidates = [
            (x2 + 8, y1),
            (x1 - label_w - 8, y1),
            (cx - label_w // 2, y2 + 8),
            (cx - label_w // 2, y1 - label_h - 8),
            (x1 + 4, y1 + 4),
        ]

        for lx, ly in candidates:
            if 8 <= lx <= self.image_w - label_w - 8 and 8 <= ly <= self.image_h - label_h - 8:
                return int(lx), int(ly)

        return self.clamp_label_position_for_static_tag(candidates[-1][0], candidates[-1][1], label_w, label_h)

    def build_static_label_click_boxes(self, items) -> list[tuple[tuple[int, int, int, int], object]]:
        # These settings match detector.py::draw_detection_labels().
        font_scale = 0.42
        thickness = 1
        pad_x = 9
        pad_y = 7

        priority = {
            "ac vent": 110,
            "air conditioner": 108,
            "glass door": 106,
            "door": 104,
            "glass window": 102,
            "window": 101,
            "ceiling": 100,
            "floor": 99,
            "carpet": 98,
            "wooden wall panel": 96,
            "television": 94,
            "monitor": 92,
            "landline phone": 91,
            "conference table": 90,
            "table": 88,
            "desk": 86,
            "conference chair": 84,
            "chair": 82,
            "plant": 80,
            "laptop": 78,
            "keyboard": 70,
            "lamp": 68,
            "painting": 66,
            "wall art": 66,
        }

        drawable_items = sorted(
            list(items),
            key=lambda item: (
                priority.get(get_name(item), 0),
                get_confidence(item),
                int(get_value(item, "count", 1)),
            ),
            reverse=True,
        )[:30]

        boxes: list[tuple[tuple[int, int, int, int], object]] = []

        for item in drawable_items:
            label = self.label_text_for_static_tag(item)
            (tw, th), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            label_w = tw + pad_x * 2
            label_h = th + pad_y * 2
            label_x, label_y = self.label_position_for_static_tag(item, label_w, label_h)

            # Slightly expand the hitbox so clicking the rounded label edges still works.
            x1 = max(0, label_x - 5)
            y1 = max(0, label_y - 5)
            x2 = min(self.image_w - 1, label_x + label_w + 7)
            y2 = min(self.image_h - 1, label_y + label_h + 7)

            boxes.append(((x1, y1, x2, y2), item))

        return boxes

    def find_static_label_at_scene_position(self, x: float, y: float):
        matched = []

        for bbox, item in self.static_label_click_boxes:
            x1, y1, x2, y2 = bbox

            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(1, (x2 - x1) * (y2 - y1))
                matched.append((area, item))

        if not matched:
            return None

        matched.sort(key=lambda pair: pair[0])
        return matched[0][1]

    def point_hits_item_for_hover(self, item, x: float, y: float) -> bool:
        """
        Return True only when the cursor is really over the visible object area.

        Important fix: large conference-table / desk detections often have a very
        loose rectangular bbox. If we use that full bbox for hover, the desk gets
        highlighted even when the cursor is beside the table. For table-like
        objects we use a perspective trapezoid inside the bbox, which matches the
        visible conference table much better and keeps the rest of the project
        unchanged.
        """
        x1, y1, x2, y2 = clamp_bbox(get_bbox(item), self.image_w, self.image_h)

        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return False

        name = get_name(item)
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Main fix for your screenshot: do not use the full rectangular desk box.
        # A conference table in perspective is narrow near the top and wider near
        # the bottom, so this trapezoid prevents false hover from nearby glass/wall
        # areas while still allowing hover on the actual table surface.
        if name in {
            "conference table",
            "conference desk",
            "table",
            "desk",
            "work desk",
            "center table",
            "coffee table",
            "side table",
        }:
            # Ignore the very top part, because YOLO sometimes includes wall/TV/
            # glass area above the actual tabletop.
            top_cut = y1 + h * 0.16
            if y < top_cut:
                return False

            progress = (y - top_cut) / max(1.0, (y2 - top_cut))
            progress = max(0.0, min(1.0, progress))

            # At the top of the table: allow only the middle area.
            # Near the bottom: allow a wider area.
            side_margin_ratio = 0.36 - (0.26 * progress)
            left_limit = x1 + w * side_margin_ratio
            right_limit = x2 - w * side_margin_ratio

            return left_limit <= x <= right_limit

        # Large furniture bboxes can also be slightly loose, so shrink them a bit.
        if get_category(item) == "Furniture" and w * h > (self.image_w * self.image_h * 0.05):
            shrink_x = w * 0.08
            shrink_y = h * 0.06
            return (x1 + shrink_x) <= x <= (x2 - shrink_x) and (y1 + shrink_y) <= y <= (y2 - shrink_y)

        return True

    def find_item_at_scene_position(self, x: float, y: float):
        matched = []

        for item in self.items:
            if self.point_hits_item_for_hover(item, x, y):
                x1, y1, x2, y2 = clamp_bbox(get_bbox(item), self.image_w, self.image_h)
                area = max(1, (x2 - x1) * (y2 - y1))
                matched.append((area, item))

        if not matched:
            return None

        matched.sort(key=lambda pair: pair[0])
        return matched[0][1]

    def clear_hover_highlight(self):
        self.hovered_item = None

        if self.highlight_item is not None:
            self.scene_obj.removeItem(self.highlight_item)
            self.highlight_item = None

        if self.highlight_label_item is not None:
            self.scene_obj.removeItem(self.highlight_label_item)
            self.highlight_label_item = None

    def set_hovered_item(self, item):
        if item is self.hovered_item:
            return

        self.clear_hover_highlight()

        if item is None:
            return

        self.hovered_item = item
        self.draw_hover_highlight()

    def create_rect_highlight_item(
        self,
        bbox: list[int],
        border: QColor,
        fill: QColor,
        width: int = 3,
    ):
        x1, y1, x2, y2 = clamp_bbox(bbox, self.image_w, self.image_h)

        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        rect_item = QGraphicsRectItem(rect)

        pen = QPen(border)
        pen.setWidth(width)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        rect_item.setPen(pen)
        rect_item.setBrush(QBrush(fill))
        rect_item.setZValue(10_000)

        return rect_item, x1, y1, x2, y2

    def create_surface_sample_highlight(self, item):
        name = get_name(item)
        bbox = surface_sample_bbox(item, self.image_w, self.image_h)

        if name in {"floor", "carpet", "rug", "floor mat"}:
            border = QColor(56, 189, 248, 235)
            fill = QColor(56, 189, 248, 34)
        elif name in {"ceiling", "ceiling pattern", "ceiling panel", "ceiling design", "wooden ceiling"}:
            border = QColor(250, 204, 21, 235)
            fill = QColor(250, 204, 21, 34)
        else:
            border = QColor(45, 212, 191, 230)
            fill = QColor(45, 212, 191, 30)

        return self.create_rect_highlight_item(
            bbox=bbox,
            border=border,
            fill=fill,
            width=3,
        )

    def create_portal_sample_highlight(self, item):
        name = get_name(item)
        bbox = portal_sample_bbox(item, self.image_w, self.image_h)

        if name in {"window", "glass window"}:
            border = QColor(56, 189, 248, 230)
            fill = QColor(56, 189, 248, 28)
        else:
            border = QColor(45, 212, 191, 230)
            fill = QColor(45, 212, 191, 28)

        return self.create_rect_highlight_item(
            bbox=bbox,
            border=border,
            fill=fill,
            width=3,
        )

    def create_full_rect_highlight(self, item):
        name = get_name(item)
        bbox = tightened_bbox(item, self.image_w, self.image_h)

        if name in {"window", "glass window"}:
            border = QColor(56, 189, 248, 220)
            fill = QColor(56, 189, 248, 22)
        else:
            border = QColor(45, 212, 191, 220)
            fill = QColor(45, 212, 191, 22)

        return self.create_rect_highlight_item(
            bbox=bbox,
            border=border,
            fill=fill,
            width=3,
        )

    def create_corner_fallback_item(self, item):
        x1, y1, x2, y2 = tightened_bbox(item, self.image_w, self.image_h)

        group = QGraphicsRectItem(QRectF(0, 0, 0, 0))
        group.setPen(QPen(Qt.PenStyle.NoPen))
        group.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        group.setZValue(10_000)

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        corner_len = int(min(max(width, height) * 0.13, 42))
        corner_len = max(18, corner_len)

        color = QColor(56, 189, 248, 235)
        glow_color = QColor(56, 189, 248, 70)

        glow_pen = QPen(glow_color)
        glow_pen.setWidth(8)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        pen = QPen(color)
        pen.setWidth(3)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        lines = [
            (x1, y1, x1 + corner_len, y1),
            (x1, y1, x1, y1 + corner_len),
            (x2, y1, x2 - corner_len, y1),
            (x2, y1, x2, y1 + corner_len),
            (x1, y2, x1 + corner_len, y2),
            (x1, y2, x1, y2 - corner_len),
            (x2, y2, x2 - corner_len, y2),
            (x2, y2, x2, y2 - corner_len),
        ]

        for lx1, ly1, lx2, ly2 in lines:
            line = QGraphicsLineItem(lx1, ly1, lx2, ly2, group)
            line.setPen(glow_pen)
            line.setZValue(10_000)

        for lx1, ly1, lx2, ly2 in lines:
            line = QGraphicsLineItem(lx1, ly1, lx2, ly2, group)
            line.setPen(pen)
            line.setZValue(10_001)

        return group, x1, y1, x2, y2

    def draw_hover_highlight(self):
        if self.hovered_item is None:
            return

        name = get_name(self.hovered_item)

        if name in SURFACE_SAMPLE_HIGHLIGHT_CLASSES:
            self.highlight_item, object_x1, object_y1, object_x2, object_y2 = (
                self.create_surface_sample_highlight(self.hovered_item)
            )
        elif name in PORTAL_SAMPLE_HIGHLIGHT_CLASSES:
            self.highlight_item, object_x1, object_y1, object_x2, object_y2 = (
                self.create_portal_sample_highlight(self.hovered_item)
            )
        elif name in RECT_FULL_HIGHLIGHT_CLASSES:
            self.highlight_item, object_x1, object_y1, object_x2, object_y2 = (
                self.create_full_rect_highlight(self.hovered_item)
            )
        else:
            sam_data = self.segmenter.mask_for_item(self.hovered_item)

            if sam_data is not None:
                pixmap, x1, y1 = sam_data
                self.highlight_item = QGraphicsPixmapItem(pixmap)
                self.highlight_item.setPos(x1, y1)
                object_x1 = x1
                object_y1 = y1
                object_x2 = x1 + pixmap.width()
                object_y2 = y1 + pixmap.height()
            else:
                self.highlight_item, object_x1, object_y1, object_x2, object_y2 = (
                    self.create_corner_fallback_item(self.hovered_item)
                )

        self.highlight_item.setZValue(10_000)
        self.highlight_item.setOpacity(1.0)
        self.scene_obj.addItem(self.highlight_item)

        item_name = get_display_name(self.hovered_item)
        link = get_link(self.hovered_item)

        if link:
            label_text = f"{item_name} • Click to open"
        else:
            label_text = item_name

        label_width_guess = min(280, max(130, len(label_text) * 7))
        label_height = 34

        label_x = int(object_x1)
        label_y = int(object_y1 - label_height - 8)

        if name in PORTAL_SAMPLE_HIGHLIGHT_CLASSES:
            # Put the text directly on the door/glass area.
            label_x = int(object_x1 + 8)
            label_y = int(object_y1 + 8)

        # If there is no space above, keep the tag inside/near the highlighted object
        # instead of pushing it off-screen or hiding it behind the toolbar.
        if label_y < 8:
            label_y = int(object_y1 + 8)

        if label_x + label_width_guess > self.image_w:
            label_x = max(8, self.image_w - label_width_guess - 8)

        label_x = max(8, min(self.image_w - label_width_guess - 8, label_x))
        label_y = max(8, min(self.image_h - label_height - 8, label_y))

        self.highlight_label_item = QGraphicsTextItem(label_text)
        self.highlight_label_item.setDefaultTextColor(Qt.GlobalColor.white)
        self.highlight_label_item.setZValue(10_002)
        self.highlight_label_item.setOpacity(1.0)
        self.highlight_label_item.setPos(label_x, label_y)

        self.highlight_label_item.setHtml(
            f"""
            <div style="
                background-color: rgba(15, 23, 42, 228);
                color: white;
                border: 1px solid rgba(56, 189, 248, 210);
                border-radius: 8px;
                padding: 5px 9px;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.15px;
            ">
                {label_text}
            </div>
            """
        )

        self.scene_obj.addItem(self.highlight_label_item)

    def is_mouse_over_hover_label(self, scene_pos) -> bool:
        """Return True when the mouse/click is on the floating label tag."""
        if self.highlight_label_item is None:
            return False

        local_pos = self.highlight_label_item.mapFromScene(scene_pos)
        return self.highlight_label_item.boundingRect().contains(local_pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.position().toPoint())
            x = scene_pos.x()
            y = scene_pos.y()

            # Permanent image labels are drawn pixels, not Qt buttons.
            # Check those label rectangles first so clicking the visible tag works.
            label_item = self.find_static_label_at_scene_position(x, y)

            if label_item is not None:
                link = get_link(label_item)

                if link:
                    webbrowser.open(link)

                event.accept()
                return

            # Hover label tag itself is clickable too.
            if self.hovered_item is not None and self.is_mouse_over_hover_label(scene_pos):
                link = get_link(self.hovered_item)

                if link:
                    webbrowser.open(link)

                event.accept()
                return

            item = self.find_item_at_scene_position(x, y)

            if item is not None:
                link = get_link(item)

                if link:
                    webbrowser.open(link)

                event.accept()
                return

            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        x = scene_pos.x()
        y = scene_pos.y()

        # Keep hover active on floating label, and also activate hover when
        # cursor is over a permanent drawn label.
        if self.hovered_item is not None and self.is_mouse_over_hover_label(scene_pos):
            item = self.hovered_item
        else:
            item = self.find_static_label_at_scene_position(x, y)

            if item is None:
                item = self.find_item_at_scene_position(x, y)

        self.set_hovered_item(item)

        if item is not None:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        elif event.buttons() & Qt.MouseButton.LeftButton:
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.clear_hover_highlight()
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_to_screen()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)


class ProductCard(QFrame):
    hoverStarted = pyqtSignal(object)
    hoverEnded = pyqtSignal(object)

    def __init__(self, item):
        super().__init__()

        self.item = item
        self.link = get_link(item)

        self.setMouseTracking(True)

        if self.link:
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

        self.setStyleSheet(
            """
            QFrame {
                background-color: #111827;
                border: 1px solid #303848;
                border-radius: 13px;
            }

            QFrame:hover {
                border: 1px solid #38bdf8;
                background-color: #162235;
            }

            QLabel {
                border: none;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 11, 14, 11)
        layout.setSpacing(5)

        name = QLabel(get_display_name(item))
        name.setMouseTracking(True)
        name.setStyleSheet("color: white; font-size: 15px; font-weight: 900;")

        category = QLabel(f"{get_category(item)} | {get_confidence(item):.2f}")
        category.setMouseTracking(True)
        category.setStyleSheet("color: #a1a1aa; font-size: 12px;")

        count = int(get_value(item, "count", 1))

        count_label = QLabel(f"Count detected: {count}")
        count_label.setMouseTracking(True)
        count_label.setStyleSheet("color: #a1a1aa; font-size: 12px;")

        if self.link:
            amazon = QLabel("Open Amazon Products →")
            amazon.setMouseTracking(True)
            amazon.setStyleSheet(
                """
                color: #38bdf8;
                font-size: 12px;
                font-weight: 900;
                margin-top: 5px;
                """
            )
        else:
            amazon = QLabel("No product link")
            amazon.setMouseTracking(True)
            amazon.setStyleSheet(
                """
                color: #64748b;
                font-size: 12px;
                font-weight: 800;
                margin-top: 5px;
                """
            )

        layout.addWidget(name)
        layout.addWidget(category)
        layout.addWidget(count_label)
        layout.addWidget(amazon)

        self.setLayout(layout)

    def enterEvent(self, event):
        self.hoverStarted.emit(self.item)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hoverEnded.emit(self.item)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.link:
            webbrowser.open(self.link)
            event.accept()
            return

        super().mousePressEvent(event)


class DetectionWindow(QWidget):
    def __init__(
        self,
        original_bgr,
        output_bgr,
        detections,
        image_path,
        output_path,
        sam_model_name: str,
        detector: FurnitureDetector | None = None,
    ):
        super().__init__()

        self.setWindowTitle("AI Furniture Detection Output")
        self.setStyleSheet("background-color: #05070d; color: white;")
        self.detector = detector or FurnitureDetector()
        self.sam_model_name = sam_model_name
        self.current_image_path = Path(image_path)
        self.current_output_path = Path(output_path)
        self.items = unique_items(detections)

        display_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        self.viewer = ProTrackpadImageViewer(
            display_rgb=display_rgb,
            original_rgb=original_rgb,
            items=self.items,
            sam_model_name=sam_model_name,
        )

        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(
            """
            color: #e5e7eb;
            font-size: 13px;
            font-weight: 900;
            padding: 8px 13px;
            background-color: #111827;
            border: 1px solid #303848;
            border-radius: 10px;
            """
        )

        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(800)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(220)
        self.zoom_slider.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.zoom_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 7px;
                background: #1f2937;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #38bdf8;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }

            QSlider::handle:horizontal:hover {
                background: #7dd3fc;
            }

            QSlider::sub-page:horizontal {
                background: #38bdf8;
                border-radius: 4px;
            }
            """
        )

        self.viewer.zoomChanged.connect(self.sync_zoom_ui_from_viewer)

        toolbar = self.build_toolbar()

        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(toolbar)
        left_layout.addWidget(self.viewer)
        left_panel.setLayout(left_layout)

        sidebar = self.build_sidebar(image_path, output_path, sam_model_name)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(sidebar)

        self.setLayout(main_layout)

        screen = QApplication.primaryScreen().availableGeometry()
        final_w = min(screen.width(), int(screen.width() * 0.96))
        final_h = min(screen.height(), int(screen.height() * 0.92))

        self.resize(final_w, final_h)
        self.move(
            max(0, int((screen.width() - final_w) / 2)),
            max(0, int((screen.height() - final_h) / 2)),
        )

        self.setup_shortcuts()

    def setup_shortcuts(self):
        QShortcut(QKeySequence("+"), self, activated=self.handle_zoom_in)
        QShortcut(QKeySequence("="), self, activated=self.handle_zoom_in)
        QShortcut(QKeySequence("-"), self, activated=self.handle_zoom_out)
        QShortcut(QKeySequence("0"), self, activated=self.handle_actual_size)
        QShortcut(QKeySequence("F"), self, activated=self.handle_fit)
        QShortcut(QKeySequence("U"), self, activated=self.handle_upload_image)

    def make_toolbar_button(self, text, width=None):
        btn = QPushButton(text)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        if width:
            btn.setFixedWidth(width)

        btn.setStyleSheet(
            """
            QPushButton {
                background-color: #111827;
                color: white;
                border: 1px solid #303848;
                border-radius: 11px;
                padding: 9px 15px;
                font-weight: 900;
                font-size: 12px;
            }

            QPushButton:hover {
                border-color: #38bdf8;
                background-color: #162235;
            }

            QPushButton:pressed {
                background-color: #0f172a;
            }
            """
        )

        return btn

    def build_toolbar(self):
        toolbar = QWidget()
        toolbar.setFixedHeight(64)
        toolbar.setStyleSheet(
            """
            QWidget {
                background-color: #080b12;
                border-bottom: 1px solid #1f2937;
            }
            """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(10)

        upload_btn = self.make_toolbar_button("Upload Image")
        zoom_out_btn = self.make_toolbar_button("−", 46)
        zoom_in_btn = self.make_toolbar_button("+", 46)
        fit_btn = self.make_toolbar_button("Fit Screen")
        actual_btn = self.make_toolbar_button("100%")
        center_btn = self.make_toolbar_button("Center")

        upload_btn.clicked.connect(self.handle_upload_image)
        zoom_out_btn.clicked.connect(self.handle_zoom_out)
        zoom_in_btn.clicked.connect(self.handle_zoom_in)
        fit_btn.clicked.connect(self.handle_fit)
        actual_btn.clicked.connect(self.handle_actual_size)
        center_btn.clicked.connect(lambda: self.viewer.centerOn(self.viewer.pixmap_item))

        self.zoom_slider.valueChanged.connect(self.handle_slider_zoom)

        hint = QLabel(
            "Upload image • hover object/label/card to highlight • click label/object/card: Amazon"
        )
        hint.setStyleSheet(
            """
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            """
        )

        layout.addWidget(upload_btn)
        layout.addWidget(zoom_out_btn)
        layout.addWidget(self.zoom_slider)
        layout.addWidget(zoom_in_btn)
        layout.addWidget(self.zoom_label)
        layout.addWidget(fit_btn)
        layout.addWidget(actual_btn)
        layout.addWidget(center_btn)
        layout.addStretch(1)
        layout.addWidget(hint)

        toolbar.setLayout(layout)
        return toolbar

    def sync_zoom_ui_from_viewer(self, zoom_value: float):
        percent = int(round(zoom_value * 100))
        self.zoom_label.setText(f"{percent}%")
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(max(10, min(800, percent)))
        self.zoom_slider.blockSignals(False)

    def handle_zoom_in(self):
        self.viewer.zoom_in()

    def handle_zoom_out(self):
        self.viewer.zoom_out()

    def handle_fit(self):
        self.viewer.fit_to_screen()

    def handle_actual_size(self):
        self.viewer.actual_size()

    def handle_slider_zoom(self, value):
        self.viewer.set_zoom(value / 100, anchor_under_mouse=False)

    def handle_upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload room/interior image",
            str(ROOT),
            "Images (*.jpg *.jpeg *.png *.webp *.bmp);;All Files (*)",
        )

        if not file_path:
            return

        self.open_new_image(Path(file_path))

    def open_new_image(self, image_path: Path):
        image_path = image_path.expanduser().resolve()

        if not image_path.exists():
            QMessageBox.critical(self, "Image not found", f"Image not found:\n{image_path}")
            return

        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))

        try:
            image = cv2.imread(str(image_path))

            if image is None:
                QMessageBox.critical(self, "Invalid image", f"Could not read image:\n{image_path}")
                return

            image = upscale_small_image_for_detection(image)

            detections = self.detector.detect(image)
            output = self.detector.draw_detections(image, detections)

            export_dir = ROOT / "exports"
            export_dir.mkdir(exist_ok=True)

            safe_stem = image_path.stem.replace(" ", "_")
            output_path = export_dir / f"{safe_stem}_detected_output.jpg"
            cv2.imwrite(str(output_path), output)

            new_window = DetectionWindow(
                original_bgr=image,
                output_bgr=output,
                detections=detections,
                image_path=image_path,
                output_path=output_path,
                sam_model_name=self.sam_model_name,
                detector=self.detector,
            )

            new_window.show()
            self.close()

        except Exception as exc:
            QMessageBox.critical(self, "Detection failed", f"Detection failed:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def handle_card_hover_started(self, item):
        self.viewer.set_hovered_item(item)

    def handle_card_hover_ended(self, item):
        if self.viewer.hovered_item is item:
            self.viewer.clear_hover_highlight()
            self.viewer.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))

    def build_sidebar(self, image_path, output_path, sam_model_name):
        container = QWidget()
        container.setFixedWidth(390)
        container.setStyleSheet(
            """
            QWidget {
                background-color: #080b12;
                border-left: 1px solid #1f2937;
            }
            """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 18, 18, 12)
        layout.setSpacing(10)

        title = QLabel("Detected Products")
        title.setStyleSheet(
            """
            color: white;
            font-size: 25px;
            font-weight: 900;
            border: none;
            """
        )

        subtitle = QLabel(f"{len(self.items)} unique products")
        subtitle.setStyleSheet(
            """
            color: #a1a1aa;
            font-size: 13px;
            border: none;
            """
        )

        source = QLabel(f"Image: {Path(image_path).name}")
        source.setStyleSheet(
            """
            color: #64748b;
            font-size: 11px;
            border: none;
            """
        )

        sam_label = QLabel(f"SAM2 model: {sam_model_name}")
        sam_label.setStyleSheet(
            """
            color: #64748b;
            font-size: 11px;
            border: none;
            """
        )

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(source)
        layout.addWidget(sam_label)

        upload_sidebar_btn = self.make_toolbar_button("Upload Another Image")
        upload_sidebar_btn.clicked.connect(self.handle_upload_image)
        layout.addWidget(upload_sidebar_btn)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            """
            QScrollArea {
                border: none;
                background-color: #080b12;
            }

            QScrollBar:vertical {
                background: #080b12;
                width: 9px;
            }

            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 4px;
            }

            QScrollBar::handle:vertical:hover {
                background: #38bdf8;
            }
            """
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #080b12; border: none;")

        cards_layout = QVBoxLayout()
        cards_layout.setContentsMargins(0, 8, 0, 8)
        cards_layout.setSpacing(10)

        if not self.items:
            empty = QLabel("No detected products found.")
            empty.setStyleSheet(
                """
                color: #a1a1aa;
                font-size: 13px;
                padding: 20px;
                border: 1px dashed #303848;
                border-radius: 10px;
                """
            )
            cards_layout.addWidget(empty)

        for item in self.items:
            card = ProductCard(item)
            card.hoverStarted.connect(self.handle_card_hover_started)
            card.hoverEnded.connect(self.handle_card_hover_ended)
            cards_layout.addWidget(card)

        cards_layout.addStretch(1)
        scroll_content.setLayout(cards_layout)
        scroll_area.setWidget(scroll_content)

        layout.addWidget(scroll_area)

        hint = QLabel(
            "Hover normal objects for soft object glow. "
            "Large surfaces use small patches; normal objects keep tight boxes and clickable links."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(
            """
            color: #64748b;
            font-size: 11px;
            border: none;
            """
        )

        saved = QLabel(f"Saved output: {Path(output_path).name}")
        saved.setStyleSheet(
            """
            color: #64748b;
            font-size: 11px;
            border: none;
            """
        )

        layout.addWidget(hint)
        layout.addWidget(saved)

        container.setLayout(layout)
        return container


def run_detection_for_image(detector: FurnitureDetector, image_path: Path):
    image_path = image_path.expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = upscale_small_image_for_detection(image)

    detections = detector.detect(image)
    output = detector.draw_detections(image, detections)

    export_dir = ROOT / "exports"
    export_dir.mkdir(exist_ok=True)

    safe_stem = image_path.stem.replace(" ", "_")
    output_path = export_dir / f"{safe_stem}_detected_output.jpg"
    cv2.imwrite(str(output_path), output)

    return image, output, detections, output_path


def choose_image_file(parent=None) -> Path | None:
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Upload room/interior image",
        str(ROOT),
        "Images (*.jpg *.jpeg *.png *.webp *.bmp);;All Files (*)",
    )

    if not file_path:
        return None

    return Path(file_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default=None,
        help="Optional path to input room image. If not given, an upload dialog opens.",
    )

    parser.add_argument(
        "--sam-model",
        default="sam2_b.pt",
        help="SAM2 model name/path. Try sam2_s.pt if sam2_b.pt is slow.",
    )

    args = parser.parse_args()

    app = QApplication(sys.argv)

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
    else:
        image_path = choose_image_file()

        if image_path is None:
            return 0

    print(f"Using image: {image_path}")
    print(f"Using SAM2 model: {args.sam_model}")

    detector = FurnitureDetector()

    try:
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        image, output, detections, output_path = run_detection_for_image(detector, image_path)
    except Exception as exc:
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(None, "Detection failed", str(exc))
        return 1
    finally:
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

    print(f"Detected {len(detections)} objects")
    print(f"Output saved to: {output_path}")
    print("Opening detection window...")

    window = DetectionWindow(
        original_bgr=image,
        output_bgr=output,
        detections=detections,
        image_path=image_path,
        output_path=output_path,
        sam_model_name=args.sam_model,
        detector=detector,
    )

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
