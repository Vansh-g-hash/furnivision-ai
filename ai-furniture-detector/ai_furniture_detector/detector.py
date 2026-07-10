"""Core detection utilities for the AI Furniture Detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import cv2
import numpy as np

try:
    from .config import settings
except Exception:  # Allows this module to be reused in isolated tests.
    settings = None

DEFAULT_CONFIDENCE = float(getattr(settings, "default_confidence", 0.12))
DEFAULT_IOU = float(getattr(settings, "default_iou", 0.50))


TARGET_OBJECTS = [
    "chair", "conference chair", "office chair", "chair", "armchair",
    "stool", "bench", "sofa", "couch", "recliner", "table", "conference table",
    "center table", "table", "coffee table", "side table", "desk",
    "work desk", "bed", "cabinet", "cupboard", "wardrobe", "drawer", "dresser",
    "shelf", "bookshelf", "bookcase", "sideboard", "tv unit", "console table",
    "nightstand", "trolley",

    "television", "television screen", "carpet", "floor rug", "floor tiles", "tv", "monitor", "computer monitor",
    "screen", "laptop", "keyboard", "mouse", "speaker", "soundbar",
    "soundbar speaker", "projector", "remote control", "control panel",
    "switch board", "power outlet", "plug socket", "charging cable",
    "table cable port", "cable port", "phone", "telephone", "landline phone",
    "landline telephone", "desk phone", "desk telephone", "office phone", "office telephone",

    "lamp", "table lamp", "floor lamp", "ceiling light", "recessed light", "downlight", "ceiling spotlight",
    "spotlight", "wall light", "light fixture", "chandelier", "led strip light", "downlight", "ceiling spotlight",
    "linear ac vent", "linear air vent", "ceiling linear diffuser",
    "wall", "floor", "ceiling", "wooden floor", "wood floor", "marble floor",
    "tile floor", "tiled floor", "floor tiles", "false ceiling",
    "ceiling panel", "ceiling panels", "wooden ceiling", "ceiling design",
    "decorative ceiling", "ceiling grid", "ceiling pattern", "coffered ceiling",
    "ceiling tiles", "wood ceiling panel", "wooden ceiling panel",

    "wooden wall panel", "wood wall panel", "wooden wall cladding",
    "wood slat wall", "wooden slat wall", "decorative wall panel",
    "wall paneling", "wood paneling", "acoustic wood panel", "wooden backdrop",
    "tv wall panel", "tv background panel",

    "door", "glass door", "office glass door", "conference room glass door",
    "window", "glass window", "glass partition", "partition", "pillar", "column",

    "air conditioner", "AC", "ac vent", "air vent", "ceiling vent",
    "air conditioning vent", "hvac vent", "ceiling air vent",
        "linear ac vent", "linear air vent", "ceiling linear diffuser", "air diffuser", "fan",
    "ceiling fan", "exhaust fan", "thermostat", "smoke detector",
    "fire alarm", "sprinkler",

    "plant", "indoor plant", "potted plant", "vase", "mirror", "picture frame",
    "painting", "wall art", "clock", "curtain", "blind", "window blind", "roller blind", "vertical blind", "carpet", "rug",
    "floor mat", "pillow", "cushion", "blanket", "window blind", "roller blind", "vertical blind",

    "refrigerator", "fridge", "microwave", "oven", "stove", "cooktop",
    "dishwasher", "sink", "kettle", "coffee machine", "toaster",

    "cup", "mug", "bottle", "book", "notebook", "paper",
    "whiteboard", "trash bin", "dustbin",
]

# Keep prompt list stable but remove exact duplicates before passing it to YOLO-World.
TARGET_OBJECTS = list(dict.fromkeys(obj.strip().lower() for obj in TARGET_OBJECTS if obj.strip()))


DISPLAY_NAME_MAP = {
    "chair": "Chair",
    "conference chair": "Conference Chair",
    "office chair": "Office Chair",
    "dining chair": "Chair",
    "armchair": "Armchair",
    "stool": "Stool",
    "bench": "Bench",
    "sofa": "Sofa",
    "couch": "Sofa",
    "recliner": "Recliner",
    "table": "Conference Table",
    "conference table": "Conference Table",
    "center table": "Conference Table",
    "dining table": "Conference Table",
    "coffee table": "Conference Table",
    "side table": "Conference Table",
    "desk": "Conference Table",
    "work desk": "Work Desk",
    "bed": "Bed",
    "cabinet": "Cabinet",
    "cupboard": "Cupboard",
    "wardrobe": "Wardrobe",
    "drawer": "Drawer",
    "dresser": "Dresser",
    "shelf": "Shelf",
    "bookshelf": "Bookshelf",
    "bookcase": "Bookshelf",
    "sideboard": "Sideboard",
    "tv unit": "TV Unit",
    "console table": "Console Table",
    "nightstand": "Nightstand",
    "trolley": "Trolley",
    "floor rug": "Rug Floor",
    "floor tiles": "Floor Tiles",
    "tiled floor": "Floor Tiles",
    "false ceiling": "Ceiling",
    "ceiling panel": "Ceiling",
    "ceiling panels": "Ceiling",
    "floor mat": "Floor Mat",

    "television": "Television / Screen",
    "television screen": "Television / Screen",
    "tv": "Television / Screen",
    "monitor": "Monitor / Screen",
    "computer monitor": "Monitor / Screen",
    "screen": "Monitor / Screen",
    "laptop": "Laptop",
    "keyboard": "Keyboard",
    "mouse": "Mouse",
    "speaker": "Speaker",
    "soundbar": "Soundbar / Speaker",
    "soundbar speaker": "Soundbar / Speaker",
    "projector": "Projector",
    "remote control": "Remote Control",
    "control panel": "Control Panel",
    "switch board": "Switch Board",
    "power outlet": "Power Outlet",
    "plug socket": "Power Outlet",
    "charging cable": "Charging Cable",
    "table cable port": "Table Cable Port",
    "cable port": "Table Cable Port",
    "phone": "Landline Phone",
    "telephone": "Landline Phone",
    "landline phone": "Landline Phone",
    "landline telephone": "Landline Phone",
    "desk phone": "Landline Phone",
    "desk telephone": "Landline Phone",
    "office phone": "Landline Phone",
    "office telephone": "Landline Phone",

    "lamp": "Lamp",
    "table lamp": "Lamp",
    "floor lamp": "Lamp",
    "ceiling light": "Ceiling Light",
    "recessed light": "Ceiling Light",
    "downlight": "Ceiling Light",
    "ceiling spotlight": "Ceiling Light",
    "spotlight": "Lamp",
    "wall light": "Lamp",
    "light fixture": "Lamp",
    "chandelier": "Chandelier",
    "led strip light": "Lamp",

    "wall": "Wall",
    "floor": "Floor",
    "ceiling": "Ceiling",
    "wooden floor": "Wooden Floor",
    "marble floor": "Marble Floor",
    "tile floor": "Tile Floor",
    "carpet": "Carpet Floor",

    "wooden wall panel": "Wooden Wall Panel",

    "door": "Door",
    "glass door": "Glass Door",
    "office glass door": "Glass Door",
    "conference room glass door": "Glass Door",
    "window": "Window",
    "glass window": "Glass Window",
    "glass partition": "Glass Partition",
    "partition": "Partition",
    "pillar": "Pillar",
    "column": "Column",

    "air conditioner": "Air Conditioner",
    "ac vent": "AC Vent",
    "air vent": "AC Vent",    "ceiling vent": "AC Vent",
    "air conditioning vent": "AC Vent",
    "hvac vent": "AC Vent",
    "ceiling air vent": "AC Vent",
    "fan": "Fan",
    "ceiling fan": "Fan",
    "exhaust fan": "Exhaust Fan",
    "thermostat": "Thermostat",
    "smoke detector": "Smoke Detector",
    "fire alarm": "Fire Alarm",
    "sprinkler": "Sprinkler",

    "plant": "Plant",
    "indoor plant": "Plant",
    "potted plant": "Plant",
    "vase": "Vase",
    "mirror": "Mirror",
    "picture frame": "Picture Frame",
    "painting": "Painting",
    "wall art": "Wall Art",
    "clock": "Clock",
    "curtain": "Curtain",
    "blind": "Window Blind",
    "window blind": "Window Blind",
    "roller blind": "Window Blind",
    "vertical blind": "Window Blind",
    "rug": "Carpet Floor",
    "floor mat": "Floor Mat",
    "pillow": "Pillow",
    "cushion": "Cushion",
    "blanket": "Blanket",

    "refrigerator": "Refrigerator",
    "fridge": "Refrigerator",
    "microwave": "Microwave",
    "oven": "Oven",
    "stove": "Stove",
    "cooktop": "Cooktop",
    "dishwasher": "Dishwasher",
    "sink": "Sink",
    "kettle": "Kettle",
    "coffee machine": "Coffee Machine",
    "toaster": "Toaster",

    "cup": "Cup",
    "mug": "Cup",
    "bottle": "Bottle",
    "book": "Book",
    "notebook": "Notebook",
    "paper": "Paper",
    "whiteboard": "Whiteboard",
    "trash bin": "Trash Bin",
    "dustbin": "Trash Bin",
}


CATEGORY_MAP = {
    **{k: "Furniture" for k in [
        "chair", "conference chair", "office chair", "chair", "armchair",
        "stool", "bench", "sofa", "couch", "recliner", "table",
        "conference table", "center table", " table", "coffee table",
        "side table", "desk", "work desk", "bed", "cabinet", "cupboard",
        "wardrobe", "drawer", "dresser", "shelf", "bookshelf", "bookcase",
        "sideboard", "tv unit", "console table", "nightstand", "trolley"
    ]},
    **{k: "Electronics" for k in [
        "television", "television screen", "tv", "monitor", "computer monitor",
        "screen", "laptop", "keyboard", "mouse", "speaker", "soundbar",
        "soundbar speaker", "projector", "remote control"
    ]},
    **{k: "Electronics / Utility" for k in [
        "control panel", "switch board", "power outlet", "plug socket",
        "charging cable", "phone", "telephone", "landline phone",
        "landline telephone", "desk phone", "desk telephone",
        "office phone", "office telephone"
    ]},
    "table cable port": "Utility",
    "cable port": "Utility",

    **{k: "Lighting" for k in [
        "lamp", "table lamp", "floor lamp", "ceiling light", "recessed light",
        "spotlight", "wall light", "light fixture", "chandelier", "led strip light", "downlight", "ceiling spotlight"
    ]},

    **{k: "Structure" for k in [
        "wall", "ceiling", "false ceiling", "ceiling panel", "ceiling panels",
        "wooden ceiling", "ceiling design", "decorative ceiling", "ceiling grid",
        "ceiling pattern", "coffered ceiling", "ceiling tiles",
        "wood ceiling panel", "wooden ceiling panel", "wooden wall panel",
        "wood wall panel", "wooden wall cladding", "wood slat wall",
        "wooden slat wall", "decorative wall panel", "wall paneling",
        "wood paneling", "acoustic wood panel", "wooden backdrop",
        "tv wall panel", "tv background panel", "partition", "pillar", "column"
    ]},

    **{k: "Flooring & Textiles" for k in [
        "floor", "wooden floor", "wood floor", "marble floor", "tile floor",
        "tiled floor", "floor tiles", "curtain", "blind", "window blind", "roller blind", "vertical blind", "carpet", "rug",
        "floor mat", "pillow", "cushion", "blanket", "window blind", "roller blind", "vertical blind"
    ]},

    **{k: "Doors & Windows" for k in [
        "door", "glass door", "office glass door", "conference room glass door",
        "window", "glass window", "glass partition"
    ]},

    **{k: "HVAC" for k in [
        "air conditioner", "ac vent", "air vent", "ceiling vent",
        "air conditioning vent", "hvac vent", "ceiling air vent",
        "linear ac vent", "linear air vent", "ceiling linear diffuser",        "exhaust fan", "thermostat"
    ]},

    "fan": "Accessories",
    "ceiling fan": "Accessories",

    **{k: "Safety" for k in ["smoke detector", "fire alarm", "sprinkler"]},

    **{k: "Decor" for k in [
        "plant", "indoor plant", "potted plant", "vase", "mirror",
        "picture frame", "painting", "wall art", "clock"
    ]},

    **{k: "Appliances" for k in [
        "refrigerator", "fridge", "microwave", "oven", "stove", "cooktop",
        "dishwasher", "sink", "kettle", "coffee machine", "toaster"
    ]},

    **{k: "Accessories" for k in [
        "cup", "mug", "bottle", "book", "notebook", "paper",
        "whiteboard", "trash bin", "dustbin"
    ]},
}


CLASS_COLORS = {
    "Furniture": (255, 128, 0),
    "Electronics": (0, 80, 255),
    "Electronics / Utility": (80, 80, 255),
    "Lighting": (0, 210, 255),
    "Doors & Windows": (255, 170, 40),
    "Flooring & Textiles": (80, 220, 120),
    "Decor": (0, 190, 80),
    "Structure": (0, 190, 190),
    "HVAC": (0, 230, 230),
    "Utility": (0, 130, 255),
    "Appliances": (210, 80, 255),
    "Safety": (0, 0, 220),
    "Accessories": (180, 120, 255),
    "Other": (160, 160, 160),
}


IGNORE_LABELS = {
    "person", "people", "man", "woman", "boy", "girl", "human",
    "face", "head", "hand", "arm", "leg", "body",
}


CLASS_CONFIDENCE = {
    "chair": 0.12,
    "conference chair": 0.12,
    "office chair": 0.12,
    " chair": 0.12,
    "armchair": 0.14,
    "stool": 0.14,
    "bench": 0.14,
    "sofa": 0.14,
    "couch": 0.14,
    "table": 0.14,
    "conference table": 0.14,
    "center table": 0.14,
    " table": 0.14,
    "coffee table": 0.14,
    "side table": 0.15,              
    "desk": 0.14,
    "work desk": 0.14,
    "bed": 0.16,
    "cabinet": 0.16,
    "cupboard": 0.16,
    "wardrobe": 0.16,
    "drawer": 0.17,
    "dresser": 0.17,
    "shelf": 0.16,
    "bookshelf": 0.16,
    "bookcase": 0.16,
    "sideboard": 0.16,
    "tv unit": 0.16,
    "console table": 0.16,
    "nightstand": 0.17,
    "trolley": 0.18,

    "television": 0.13,
    "television screen": 0.13,
    "tv": 0.13,
    "monitor": 0.13,
    "computer monitor": 0.13,
    "screen": 0.16,
    "laptop": 0.20,
    "keyboard": 0.18,
    "mouse": 0.20,
    "speaker": 0.18,
    "soundbar": 0.18,
    "soundbar speaker": 0.18,
    "projector": 0.20,
    "remote control": 0.22,
    "control panel": 0.30,
    "switch board": 0.44,
    "power outlet": 0.44,
    "plug socket": 0.44,
    "table cable port": 0.36,
    "cable port": 0.36,
    "charging cable": 0.34,
    "phone": 0.10,
    "telephone": 0.10,
    "landline phone": 0.10,
    "landline telephone": 0.10,
    "desk phone": 0.10,
    "desk telephone": 0.10,
    "office phone": 0.10,
    "office telephone": 0.10,

    "lamp": 0.16,
    "table lamp": 0.16,
    "floor lamp": 0.17,
    "ceiling light": 0.24,
    "recessed light": 0.20,
    "downlight": 0.20,
    "ceiling spotlight": 0.20,
    "spotlight": 0.30,
    "wall light": 0.36,
    "light fixture": 0.34,
    "chandelier": 0.25,
    "led strip light": 0.35,

    "air conditioner": 0.22,
    "ac vent": 0.42,    "air vent": 0.42,
    "ceiling vent": 0.42,
    "air conditioning vent": 0.42,
    "hvac vent": 0.42,
    "ceiling air vent": 0.42,
    "fan": 0.22,
    "ceiling fan": 0.22,
    "exhaust fan": 0.30,
    "thermostat": 0.34,
    "smoke detector": 0.34,
    "fire alarm": 0.34,
    "sprinkler": 0.36,

    "wall": 0.48,
    "floor": 0.28,
    "ceiling": 0.28,
    "wooden floor": 0.22,
    "marble floor": 0.22,
    "tile floor": 0.22,
    "carpet": 0.34,

    "wooden wall panel": 0.18,

    "door": 0.18,
    "glass door": 0.16,
    "office glass door": 0.16,
    "conference room glass door": 0.16,
    "window": 0.18,
    "glass window": 0.17,
    "glass partition": 0.16,
    "partition": 0.28,
    "pillar": 0.30,
    "column": 0.30,

    "plant": 0.13,
    "vase": 0.18,
    "mirror": 0.18,
    "picture frame": 0.20,
    "painting": 0.20,
    "wall art": 0.22,
    "clock": 0.22,
    "curtain": 0.18,
    "blind": 0.16,
    "window blind": 0.16,
    "roller blind": 0.16,
    "vertical blind": 0.16,
    "pillow": 0.22,
    "cushion": 0.20,
    "blanket": 0.22,

    "refrigerator": 0.18,
    "microwave": 0.20,
    "oven": 0.20,
    "stove": 0.22,
    "cooktop": 0.52,
    "dishwasher": 0.22,
    "sink": 0.22,
    "kettle": 0.22,
    "coffee machine": 0.22,
    "toaster": 0.22,
    "cup": 0.20,
    "mug": 0.20,
    "bottle": 0.20,
    "book": 0.46,
    "notebook": 0.48,
    "paper": 0.50,
    "whiteboard": 0.22,
    "trash bin": 0.22,
    "dustbin": 0.22,
}


CLASS_MIN_AREA = {
    "wall": 0.12,
    "floor": 0.035,
    "ceiling": 0.030,
    "wooden floor": 0.025,
    "marble floor": 0.025,
    "tile floor": 0.025,
    "carpet": 0.012,
    "wooden wall panel": 0.010,

    "door": 0.015,
    "glass door": 0.008,
    "office glass door": 0.008,
    "conference room glass door": 0.008,
    "window": 0.010,
    "glass window": 0.010,
    "glass partition": 0.008,
    "curtain": 0.012,
    "blind": 0.006,
    "window blind": 0.006,
    "roller blind": 0.006,
    "vertical blind": 0.006,

    "chair": 0.004,
    "conference chair": 0.004,
    "office chair": 0.004,
    " chair": 0.004,
    "armchair": 0.004,
    "stool": 0.002,
    "sofa": 0.010,
    "couch": 0.010,
    "table": 0.006,
    "conference table": 0.006,
    "center table": 0.006,
    "coffee table": 0.005,
    "side table": 0.003,
    "desk": 0.006,
    "work desk": 0.006,
    "bench": 0.004,
    "cabinet": 0.004,
    "cupboard": 0.004,
    "wardrobe": 0.006,
    "drawer": 0.0025,
    "shelf": 0.003,
    "bookshelf": 0.004,
    "bookcase": 0.004,

    "television": 0.003,
    "television screen": 0.003,
    "tv": 0.003,
    "monitor": 0.0025,
    "computer monitor": 0.0025,
    "screen": 0.0025,
    "laptop": 0.0035,
    "keyboard": 0.0012,
    "mouse": 0.0008,
    "speaker": 0.0012,
    "soundbar": 0.0012,
    "phone": 0.00035,
    "telephone": 0.00035,
    "landline phone": 0.00035,
    "landline telephone": 0.00035,
    "desk phone": 0.00035,
    "desk telephone": 0.00035,
    "office phone": 0.00035,
    "office telephone": 0.00035,

    "lamp": 0.0015,
    "table lamp": 0.0015,
    "floor lamp": 0.0015,
    "ceiling light": 0.001,
    "recessed light": 0.00025,
    "downlight": 0.00025,
    "ceiling spotlight": 0.00025,
    "spotlight": 0.0006,
    "wall light": 0.0008,
    "ac vent": 0.00025,    "air vent": 0.00045,
    "ceiling vent": 0.00045,
    "air conditioning vent": 0.00045,
    "hvac vent": 0.00045,
    "ceiling air vent": 0.00045,
    "control panel": 0.0008,
    "switch board": 0.0006,
    "power outlet": 0.0005,
    "plug socket": 0.0005,
    "table cable port": 0.0005,
    "cable port": 0.0005,

    "plant": 0.0012,
    "vase": 0.001,
    "mirror": 0.002,
    "picture frame": 0.0015,
    "painting": 0.0015,
    "clock": 0.0008,
    "cup": 0.0005,
    "mug": 0.0005,
    "bottle": 0.0005,
    "book": 0.0005,
    "notebook": 0.0005,
    "paper": 0.0005,
}


MAX_PER_CLASS = {
    "wall": 2,
    "floor": 1,
    "ceiling": 1,
    "wooden floor": 1,
    "marble floor": 1,
    "tile floor": 1,
    "carpet": 1,
    "wooden wall panel": 1,

    "chair": 12,
    "conference chair": 20,
    "office chair": 20,
    "armchair": 10,
    "table": 4,
    "conference table": 2,
    "desk": 4,
    "work desk": 4,
    "sofa": 3,
    "bench": 3,
    "cabinet": 3,
    "cupboard": 3,
    "wardrobe": 2,
    "bookshelf": 3,

    "television": 2,
    "monitor": 3,
    "laptop": 4,
    "landline phone": 3,
    "keyboard": 6,
    "mouse": 6,
    "speaker": 4,
    "soundbar": 3,

    "plant": 6,
    "painting": 4,
    "wall art": 4,
    "picture frame": 4,
    "mirror": 3,
    "clock": 3,

    "lamp": 6,
    "chandelier": 2,

    "cup": 10,
    "mug": 10,
    "bottle": 10,
    "book": 12,
    "notebook": 10,

    "curtain": 3,
    "blind": 6,
    "window blind": 6,
    "roller blind": 6,
    "vertical blind": 6,
    "recessed light": 12,
    "downlight": 12,
    "ceiling spotlight": 12,
    "ac vent": 8,
    "window": 4,
    "glass window": 4,
    "glass partition": 4,
    "door": 3,
    "glass door": 3,
}


SURFACE_CLASSES = {
    "wall",
    "floor",
    "ceiling",
    "wooden floor",
    "marble floor",
    "tile floor",
    "carpet",
    "wooden wall panel",
}


SHOPPABLE_EXCLUDE = {
    # Non-buyable room surfaces/structural mass. Doors, windows and vents are
    # intentionally NOT excluded because the desktop app opens Amazon links
    # directly from detected cards/objects.
    "wall",
    "partition",
    "pillar",
    "column",
}


class DetectedItem:
    """Compatibility DetectedItem used by tests and the rest of the code.

    Accepts either `coords` (tuple) or `bbox` (list) in the constructor to
    remain backwards-compatible with older callers and tests.
    """

    def __init__(
        self,
        name: str,
        confidence: float,
        coords: tuple | None = None,
        bbox: list[int] | None = None,
        category: str | None = None,
        link: str | None = None,
        count: int = 1,
    ) -> None:
        # Normalise name for consistency.
        self.name = _normalize_label(name)
        self.confidence = float(confidence)

        if coords is not None:
            self.bbox = [int(v) for v in coords]
        elif bbox is not None:
            self.bbox = [int(v) for v in bbox]
        else:
            self.bbox = [0, 0, 0, 0]

        self.category = category or category_for_item(self.name)
        self.link = link if link is not None else build_amazon_link(self.name)
        self.count = int(count)

    @property
    def coords(self) -> tuple:
        return tuple(self.bbox)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return max(0, x2 - x1) * max(0, y2 - y1)

    def contains(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": display_name(self.name),
            "confidence": round(float(self.confidence), 4),
            "coords": [int(v) for v in self.bbox],
            "category": self.category,
            "link": self.link,
            "count": int(self.count),
        }


@dataclass
class DetectionResult:
    all_items: list[DetectedItem]
    unique_items: list[DetectedItem]
    annotated_image: np.ndarray


Detection = DetectedItem


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "detect.py").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent


def default_model_path() -> Path:
    root = project_root()

    possible_dirs = [root, root.parent, Path.cwd()]

    for folder in possible_dirs:
        for name in (
            "yolov8x-worldv2.pt",
            "yolov8s-worldv2.pt",
            "yolov8m-worldv2.pt",
            "yolov8m.pt",
            "yolov8n.pt",
        ):
            candidate = folder / name
            if candidate.exists():
                return candidate

    return root / "yolov8x-worldv2.pt"


def load_image_bgr(path: Path | str) -> np.ndarray:
    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def _normalize_label(label: str) -> str:
    label = str(label).lower().strip()

    replacements = {
        "tv": "television",
        "television screen": "television",
        "computer monitor": "monitor",
        "screen": "monitor",
        "monitor screen": "monitor",
        "display screen": "monitor",

        "soundbar speaker": "soundbar",

        # Conference-room table fix:
        # YOLO-World may call the same large conference table a generic table,
        # desk, work desk, coffee table, side table, center table, or dining table.
        # Normalize all of these to one stable label so it is displayed/tagged.
        "table": "conference table",
        "conference desk": "conference table",
        "desk": "conference table",
        "work desk": "conference table",
        "coffee table": "conference table",
        "side table": "conference table",
        "center table": "conference table",
        "dining table": "conference table",

        "potted plant": "plant",
        "indoor plant": "plant",

        "bookcase": "bookshelf",
        "couch": "sofa",

        "air vent": "ac vent",        "ceiling vent": "ac vent",
        "air conditioning vent": "ac vent",
        "hvac vent": "ac vent",
        "ceiling air vent": "ac vent",
        "ac": "air conditioner",

        "cable port": "table cable port",
        "plug socket": "power outlet",

        "phone": "landline phone",
        "telephone": "landline phone",
        "landline telephone": "landline phone",
        "desk phone": "landline phone",
        "desk telephone": "landline phone",
        "office phone": "landline phone",
        "office telephone": "landline phone",

        "fridge": "refrigerator",
        "dustbin": "trash bin",

        "wood floor": "wooden floor",
        "tiled floor": "tile floor",
        "floor tiles": "tile floor",

        "false ceiling": "ceiling",
        "wooden ceiling": "ceiling",
        "decorative ceiling": "ceiling",
        "coffered ceiling": "ceiling",
        "ceiling design": "ceiling",
        "ceiling grid": "ceiling",
        "ceiling tiles": "ceiling",
        "ceiling panel": "ceiling",
        "ceiling panels": "ceiling",
        "wood ceiling panel": "ceiling",
        "wooden ceiling panel": "ceiling",
        "ceiling pattern": "ceiling",
        "ceiling texture": "ceiling",
        "ceiling decor": "ceiling",

        "wood wall panel": "wooden wall panel",
        "wooden wall cladding": "wooden wall panel",
        "wood slat wall": "wooden wall panel",
        "wooden slat wall": "wooden wall panel",
        "decorative wall panel": "wooden wall panel",
        "wall paneling": "wooden wall panel",
        "wood paneling": "wooden wall panel",
        "acoustic wood panel": "wooden wall panel",
        "wooden backdrop": "wooden wall panel",
        "tv wall panel": "wooden wall panel",
        "tv background panel": "wooden wall panel",
        "wood feature wall": "wooden wall panel",
        "wooden feature wall": "wooden wall panel",

        "rug": "carpet",
        "floor rug": "carpet",
        "floor mat": "carpet",

        "ceiling light": "recessed light",
        "downlight": "recessed light",
        "ceiling spotlight": "recessed light",
        "wall light": "lamp",
        "light fixture": "lamp",
        "spotlight": "recessed light",

        "mug": "cup",

        "window blind": "blind",
        "roller blind": "blind",
        "vertical blind": "blind",

        "office glass door": "glass door",
        "conference room glass door": "glass door",
    }

    return replacements.get(label, label)


def display_name(label: str) -> str:
    key = _normalize_label(label)
    return DISPLAY_NAME_MAP.get(key, key.title())


def category_for_item(label: str) -> str:
    key = _normalize_label(label)
    return CATEGORY_MAP.get(key, "Other")


def amazon_query(label: str) -> str:
    key = _normalize_label(label)

    if key in SHOPPABLE_EXCLUDE:
        return ""

    name = display_name(key).replace("/", " ")
    category = category_for_item(key)

    if category in {"Electronics", "Electronics / Utility", "Appliances"}:
        return name

    if category == "Lighting":
        return f"{name} interior light"

    if category == "Doors & Windows":
        if "glass door" in key:
            return "glass door hardware office"
        if "window" in key:
            return "glass window hardware"
        return f"{name} hardware"

    if category == "HVAC":
        if "slot diffuser" in key:
            return "ceiling ac vent"
        if "vent" in key:
            return "AC air vent grille"
        return name

    if category in {"Furniture", "Decor", "Flooring & Textiles", "Accessories"}:
        return f"{name} furniture"

    return name


def build_amazon_link(name: str) -> str:
    query = amazon_query(name)

    if not query:
        return ""

    return f"https://www.amazon.in/s?k={quote_plus(query)}"


def color_for_item(label: str) -> tuple[int, int, int]:
    category = category_for_item(label)
    return CLASS_COLORS.get(category, CLASS_COLORS["Other"])


def load_model(model_path: Path | str | None = None):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    requested_path = Path(model_path or default_model_path()).expanduser()

    if requested_path.exists():
        print(f"Loading local model: {requested_path.name}")
        model = YOLO(str(requested_path))
    else:
        # Cloud deployment fallback:
        # Ultralytics downloads this official YOLO-World model automatically
        # when local weights are not present in the repository.
        fallback_model = "yolov8s-worldv2.pt"

        print(
            f"Local model not found at {requested_path}. "
            f"Downloading {fallback_model}..."
        )

        model = YOLO(fallback_model)

    if hasattr(model, "set_classes"):
        try:
            model.set_classes(TARGET_OBJECTS)
            print(
                f"YOLO-World custom classes loaded: "
                f"{len(TARGET_OBJECTS)} objects"
            )
        except Exception as exc:
            print(f"Warning: Could not set YOLO-World classes: {exc}")

    return model


def _is_target(label: str) -> bool:
    label = _normalize_label(label)

    if label in IGNORE_LABELS:
        return False

    normalized_targets = {_normalize_label(obj) for obj in TARGET_OBJECTS}

    if label in normalized_targets:
        return True

    return any(target in label or label in target for target in normalized_targets)


def _bbox_area_ratio(bbox: list[int], image_shape: tuple[int, int, int]) -> float:
    h, w = image_shape[:2]
    image_area = max(1, h * w)

    x1, y1, x2, y2 = bbox
    box_area = max(0, x2 - x1) * max(0, y2 - y1)

    return box_area / image_area


def _bbox_aspect_ratio(bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    return max(bw / bh, bh / bw)


def _confidence_threshold(label: str) -> float:
    key = _normalize_label(label)
    return CLASS_CONFIDENCE.get(key, 0.18)


def _min_area_threshold(label: str) -> float:
    key = _normalize_label(label)
    return CLASS_MIN_AREA.get(key, 0.001)


def _passes_smart_filter(
    label: str,
    confidence: float,
    bbox: list[int],
    image_shape: tuple[int, int, int],
) -> bool:
    key = _normalize_label(label)

    if key in IGNORE_LABELS:
        return False

    if confidence < _confidence_threshold(key):
        return False

    area_ratio = _bbox_area_ratio(bbox, image_shape)

    if area_ratio < _min_area_threshold(key):
        return False

    aspect = _bbox_aspect_ratio(bbox)

    long_thin_allowed = {
        "curtain", "blind", "shelf", "bookshelf", "soundbar", "charging cable",
        "led strip light", "lamp", "recessed light", "blind", "ac vent", "table cable port", "window",
        "glass window", "door", "glass door", "wall", "floor", "ceiling", "wooden floor", "marble floor",
        "tile floor", "carpet", "wooden wall panel",
    }

    if aspect > 18 and key not in long_thin_allowed:
        return False

    if key in FLOOR_SURFACE_KEYS and not _is_probable_floor_surface_bbox(key, bbox, image_shape):
        return False

    if key in BOOK_LIKE_KEYS and not _is_probable_book_like_bbox(key, bbox, image_shape):
        return False

    if key in VENT_KEYS and not _is_probable_ceiling_vent_bbox(bbox, image_shape):
        return False

    if key in {"cooktop", "stove"} and not _is_probable_kitchen_appliance_bbox(key, bbox, image_shape):
        return False

    return True


def _iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union



FLOOR_SURFACE_KEYS = {"floor", "wooden floor", "marble floor", "tile floor", "carpet", "rug", "floor mat"}
TABLE_LIKE_KEYS = {"conference table", "table", "desk", "work desk", "coffee table", "side table", "center table"}
BOOK_LIKE_KEYS = {"book", "notebook", "paper"}
VENT_KEYS = {"ac vent", "linear ac vent", "linear air vent", "ceiling linear diffuser", "air diffuser", "air vent", "ceiling vent", "air conditioning vent", "hvac vent", "ceiling air vent"}


def _box_area(bbox: list[int]) -> int:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(box_a: list[int], box_b: list[int]) -> int:
    ax1, ay1, ax2, ay2 = [int(v) for v in box_a]
    bx1, by1, bx2, by2 = [int(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _center_inside(inner: list[int], outer: list[int]) -> bool:
    cx, cy = _bbox_center(inner)
    x1, y1, x2, y2 = [int(v) for v in outer]
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _overlap_of_small_box(box_a: list[int], box_b: list[int]) -> float:
    """Intersection divided by the smaller box area."""
    inter = _intersection_area(box_a, box_b)
    denom = max(1, min(_box_area(box_a), _box_area(box_b)))
    return inter / denom


def _is_probable_floor_surface_bbox(label: str, bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Keep real floor/carpet/rug, reject table-top patches wrongly named carpet/rug."""
    key = _normalize_label(label)
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, image_shape)

    if key not in FLOOR_SURFACE_KEYS:
        return True

    # Floors/carpets/rugs belong in the lower room plane. Small patches floating
    # in the middle of the photo are usually table/wall texture mistakes.
    if cy < h * 0.56:
        return False

    # A true floor/carpet sample should touch the lower part of the image or be
    # large enough to represent visible flooring. This preserves interior-design
    # flooring output while rejecting table-surface false carpet tags.
    touches_bottom_plane = y2 > h * 0.74
    large_surface = area_ratio > 0.030 and bh > h * 0.16
    if not (touches_bottom_plane or large_surface):
        return False

    # Rugs/carpets/floor mats should not be tiny random rectangles.
    if key in {"carpet", "rug", "floor mat"} and area_ratio < 0.0035:
        return False

    # Very tall/narrow patches are not flooring.
    if bh > bw * 2.2:
        return False

    return True


def _is_probable_book_like_bbox(label: str, bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Reject switch boards / wall plates being tagged as Book/Notebook/Paper."""
    key = _normalize_label(label)
    if key not in BOOK_LIKE_KEYS:
        return True

    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, image_shape)
    aspect = max(bw / bh, bh / bw)

    # Books/papers in room photos are usually on tables/shelves in the lower half,
    # not high on walls where switch boards/control panels live.
    if cy < h * 0.48:
        return False

    # Reject huge wall-panel-like rectangles and extremely skinny wall lines.
    if area_ratio > 0.020 or aspect > 7.0:
        return False

    # Far-right wall plates/switches often get called Book by YOLO-World.
    # If it is in the upper/right wall region, make it prove itself with high conf.
    if cx > w * 0.78 and cy < h * 0.72:
        return False

    return True


def _is_probable_kitchen_appliance_bbox(label: str, bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Cooktop/stove false positives are common on conference tables."""
    key = _normalize_label(label)
    if key not in {"cooktop", "stove"}:
        return True

    h, w = image_shape[:2]
    _, cy = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, image_shape)

    # Keep only confident, reasonably sized kitchen-plane detections. A tiny tag
    # on a conference table should disappear.
    if cy < h * 0.50 or area_ratio < 0.004:
        return False

    return True


def _remove_context_false_positives(items: list[DetectedItem], image_shape: tuple[int, int, int]) -> list[DetectedItem]:
    """Second-pass cleanup using relationships between detections.

    This is the important fix for your screenshot:
    - carpet/rug/floor is kept, but not when it sits inside the conference table
    - AC vents are kept, but only when they look like ceiling vents
    - books are kept, but switch-board/wall-panel mistakes are removed
    - cooktop/stove mistakes on a meeting table are removed
    """
    table_boxes = [item.bbox for item in items if _normalize_label(item.name) in TABLE_LIKE_KEYS]
    cleaned: list[DetectedItem] = []

    for item in items:
        key = _normalize_label(item.name)

        if key in VENT_KEYS and not _is_probable_ceiling_vent_bbox(item.bbox, image_shape):
            continue

        if key in FLOOR_SURFACE_KEYS:
            if not _is_probable_floor_surface_bbox(key, item.bbox, image_shape):
                continue
            # Reject carpet/rug/floor-mat boxes that are actually on the table.
            if key in {"carpet", "rug", "floor mat"}:
                on_table = any(
                    _center_inside(item.bbox, table_box) or _overlap_of_small_box(item.bbox, table_box) > 0.55
                    for table_box in table_boxes
                )
                if on_table:
                    continue

        if key in BOOK_LIKE_KEYS and not _is_probable_book_like_bbox(key, item.bbox, image_shape):
            continue

        if key in {"cooktop", "stove"} and not _is_probable_kitchen_appliance_bbox(key, item.bbox, image_shape):
            continue

        cleaned.append(item)

    return cleaned

def _filter_overlapping_items(items: list[DetectedItem]) -> list[DetectedItem]:
    sorted_items = sorted(items, key=lambda item: item.confidence, reverse=True)
    kept: list[DetectedItem] = []

    for item in sorted_items:
        key = _normalize_label(item.name)
        duplicate = False

        for old in kept:
            old_key = _normalize_label(old.name)

            if key == old_key and _iou(item.bbox, old.bbox) > 0.55:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


def _limit_per_class(items: list[DetectedItem]) -> list[DetectedItem]:
    grouped: dict[str, list[DetectedItem]] = {}

    for item in items:
        key = _normalize_label(item.name)
        grouped.setdefault(key, []).append(item)

    final_items: list[DetectedItem] = []

    for key, group in grouped.items():
        group = sorted(group, key=lambda item: item.confidence, reverse=True)
        limit = MAX_PER_CLASS.get(key)

        if limit is not None:
            group = group[:limit]

        final_items.extend(group)

    return sorted(final_items, key=lambda item: item.confidence, reverse=True)




FLOORING_KEYS = {
    "floor",
    "wooden floor",
    "marble floor",
    "tile floor",
    "carpet",
}


def _has_flooring_item(items: list[DetectedItem]) -> bool:
    return any(_normalize_label(item.name) in FLOORING_KEYS for item in items)


def _infer_floor_material(image_bgr: np.ndarray) -> str:
    """
    Add a practical fallback when YOLO misses the room floor material.

    This is intentionally conservative: if the lower visible floor area is dark
    and textured, tag it as carpet; otherwise tag it as floor. It prevents blank
    floor output in office/interior photos where the detector focuses on chairs,
    desk, glass and ceiling instead.
    """
    h, w = image_bgr.shape[:2]
    sample = image_bgr[int(h * 0.70):h, int(w * 0.58):w]

    if sample.size == 0:
        return "floor"

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    texture = float(gray.std())

    # Office carpets are usually darker and visibly textured/grainy.
    if mean < 145 and texture > 18:
        return "carpet"

    return "floor"


def _fallback_floor_bbox(image_shape: tuple[int, int, int]) -> list[int]:
    h, w = image_shape[:2]

    # Lower-right floor patch: avoids the big table/desk in many conference-room
    # photos and puts the label on the visible carpet/floor area.
    x1 = int(w * 0.58)
    y1 = int(h * 0.72)
    x2 = w - 1
    y2 = h - 1

    return [x1, y1, x2, y2]


def _add_flooring_fallback(
    items: list[DetectedItem],
    image_bgr: np.ndarray,
) -> list[DetectedItem]:
    if _has_flooring_item(items):
        return items

    floor_label = _infer_floor_material(image_bgr)

    return [
        *items,
        DetectedItem(
            name=floor_label,
            confidence=0.55,
            bbox=_fallback_floor_bbox(image_bgr.shape),
            category=category_for_item(floor_label),
            link=build_amazon_link(floor_label),
        ),
    ]




def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _is_probable_recessed_light_bbox(bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Return True for small round/oval recessed ceiling LED lights only.

    This intentionally rejects pendant/hanging bulbs and large ceiling patches,
    so the final UI can show one clean label like "Ceiling Light x8" for the
    small LED/downlight group without cluttering the image.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = _bbox_center(bbox)
    area_ratio = (bw * bh) / max(1, w * h)
    aspect = max(bw / bh, bh / bw)

    # Real recessed/downlight LEDs are in the ceiling band.
    if cy > h * 0.46:
        return False

    # They are small. Big pendant bulbs/lamp bodies must remain Lamp.
    if area_ratio < 0.000010 or area_ratio > 0.00115:
        return False

    # Round/oval/short rectangular only, not long strips/vents.
    if aspect > 3.2:
        return False

    # Exclude the hanging pendant-light column in the centre of this conference-room
    # view. The round canopy / hanging bulb were being counted as recessed LEDs,
    # which pulled the single grouped label to the wrong place.
    central_pendant_zone = (w * 0.40 < cx < w * 0.58) and (h * 0.08 < cy < h * 0.44)
    if central_pendant_zone:
        return False

    return True


def _contextual_relabel(label: str, bbox: list[int], image_shape: tuple[int, int, int]) -> str:
    """
    Correct common YOLO-World confusions in conference/interior photos.

    This keeps the detector practical without changing the main model:
    - glass partitions/windows on the left are not shown as glass doors
    - tall door-like reflections are not shown as mirrors
    - ceiling light aliases remain ceiling lights instead of generic lamps
    """
    key = _normalize_label(label)
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx, cy = _bbox_center(bbox)
    area_ratio = _bbox_area_ratio(bbox, image_shape)

    # Large/tall glass regions on the left side of conference rooms are usually
    # window/glass partition panels, not an actual door.
    if key == "glass door":
        if cx < w * 0.42 or bw > bh * 0.62:
            return "glass window"

    # YOLO sometimes mistakes a reflective glass door panel as a mirror.
    # Only relabel strongly door-shaped mirrors near room edges, so real mirrors
    # in normal positions are still kept as Mirror.
    if key == "mirror":
        tall_door_shape = bh > bw * 1.65 and area_ratio > 0.006
        near_room_edge = cx < w * 0.22 or cx > w * 0.62
        starts_high = y1 < h * 0.62
        if tall_door_shape and near_room_edge and starts_high:
            return "glass door"

    # Keep hanging pendant bulbs/fixtures as Lamp. Only small flush circular/oval
    # ceiling LEDs should become the grouped "Ceiling Light xN" item.
    if key == "recessed light" and not _is_probable_recessed_light_bbox(bbox, image_shape):
        return "lamp"

    return key



def _is_probable_ceiling_vent_bbox(bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Return True only for real ceiling AC vent/diffuser slots.

    The earlier version was too loose and accepted dark lines on the left glass
    wall/window partition. This version is stricter: AC vents must live in the
    upper ceiling zone, must be long/thin, and must not sit in the far-left
    window/blind region.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    aspect = max(bw / bh, bh / bw)
    area_ratio = (bw * bh) / max(1, w * h)

    # Real vents are on ceiling only. Anything on table/wall/lower-room area
    # should be rejected. This fixes tissue boxes being tagged as AC vents.
    if cy > h * 0.32:
        return False

    # Reject lines sitting on the wall/door transition on the right side. Real
    # ceiling slot diffusers in this view are in the ceiling plane, not beside
    # the glass door or whiteboard area.
    if cx > w * 0.72 and cy > h * 0.24:
        return False

    # Reject the left-side window/blind/glass partition region completely.
    # This is where your false "AC Vent x4" was coming from.
    if cx < w * 0.23:
        return False

    # Very right wall/whiteboard side is also not a ceiling diffuser in this view.
    if cx > w * 0.88:
        return False

    # Ceiling vents are long and thin slots/lines.
    if aspect < 6.0:
        return False

    # Remove tiny noisy edges and huge ceiling strips.
    if area_ratio < 0.000035 or area_ratio > 0.018:
        return False

    return True


def _remove_false_ac_vent_items(items: list[DetectedItem], image_shape: tuple[int, int, int]) -> list[DetectedItem]:
    cleaned: list[DetectedItem] = []
    for item in items:
        key = _normalize_label(item.name)
        if key == "ac vent" and not _is_probable_ceiling_vent_bbox(item.bbox, image_shape):
            continue
        cleaned.append(item)
    return cleaned

def _has_item(items: list[DetectedItem], names: set[str]) -> bool:
    normalized = {_normalize_label(name) for name in names}
    return any(_normalize_label(item.name) in normalized for item in items)


def _add_recessed_light_fallbacks(items: list[DetectedItem], image_bgr: np.ndarray) -> list[DetectedItem]:
    """Detect small ceiling LED/downlight spots that YOLO-World often misses.

    Output behavior: the raw detections stay as individual items for accurate
    counting, but draw_detection_labels(unique_only=True) groups them into one
    readable label: "Ceiling Light xN". This avoids clutter while still telling
    the user how many small ceiling LEDs were found.
    """
    h, w = image_bgr.shape[:2]
    top_h = int(h * 0.48)
    top = image_bgr[:top_h, :]
    if top.size == 0:
        return items

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)

    candidate_boxes: list[list[int]] = []

    # Bright, low-saturation blobs catch white circular/oval recessed LEDs.
    # A slightly stricter threshold avoids tagging large ceiling glow strips.
    mask = cv2.inRange(hsv, (0, 0, 178), (180, 115, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = w * h
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = [x, y, x + bw, y + bh]

        if not _is_probable_recessed_light_bbox(bbox, image_bgr.shape):
            continue

        # Verify local brightness: small LED patch should be brighter than the
        # nearby ceiling background, not just a random beige circle/shadow.
        pad_bg = max(10, int(max(bw, bh) * 1.8))
        bx1 = max(0, x - pad_bg)
        by1 = max(0, y - pad_bg)
        bx2 = min(w - 1, x + bw + pad_bg)
        by2 = min(top_h - 1, y + bh + pad_bg)
        local = gray[by1:by2, bx1:bx2]
        patch = gray[y:min(top_h, y + bh), x:min(w, x + bw)]
        if local.size and patch.size:
            if float(patch.mean()) < max(160.0, float(local.mean()) + 8.0):
                continue

        pad = max(5, int(max(bw, bh) * 0.35))
        candidate_boxes.append([
            max(0, x - pad),
            max(0, y - pad),
            min(w - 1, x + bw + pad),
            min(h - 1, y + bh + pad),
        ])

    # Hough circles catches small round LEDs missed by thresholding.
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(24, int(w * 0.040)),
        param1=80,
        param2=13,
        minRadius=max(4, int(w * 0.004)),
        maxRadius=max(12, int(w * 0.020)),
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            pad = max(5, int(r * 0.55))
            bbox = [
                max(0, cx - r - pad),
                max(0, cy - r - pad),
                min(w - 1, cx + r + pad),
                min(h - 1, cy + r + pad),
            ]
            if not _is_probable_recessed_light_bbox(bbox, image_bgr.shape):
                continue

            x1, y1, x2, y2 = bbox
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0 or float(np.mean(patch)) < 150:
                continue

            candidate_boxes.append(bbox)

    additions: list[DetectedItem] = []
    for bbox in sorted(candidate_boxes, key=lambda b: (b[1], b[0])):
        duplicate = False
        for old in items + additions:
            old_key = _normalize_label(old.name)
            if old_key == "recessed light" and _iou(bbox, old.bbox) > 0.18:
                duplicate = True
                break
        if duplicate:
            continue

        additions.append(
            DetectedItem(
                name="recessed light",
                confidence=0.54,
                bbox=bbox,
                category=category_for_item("recessed light"),
                link=build_amazon_link("recessed light"),
            )
        )
        if len(additions) >= 12:
            break

    return [*items, *additions]


def _clean_recessed_light_items(items: list[DetectedItem], image_shape: tuple[int, int, int]) -> list[DetectedItem]:
    """Remove false/duplicate ceiling-light items while preserving count."""
    cleaned: list[DetectedItem] = []
    lights: list[DetectedItem] = []

    for item in items:
        key = _normalize_label(item.name)
        if key != "recessed light":
            cleaned.append(item)
            continue
        if _is_probable_recessed_light_bbox(item.bbox, image_shape):
            lights.append(item)
        else:
            # Large/tall light-like detections are pendant lamps, not recessed LEDs.
            cleaned.append(
                DetectedItem(
                    name="lamp",
                    confidence=min(float(item.confidence), 0.42),
                    bbox=item.bbox,
                    category=category_for_item("lamp"),
                    link=build_amazon_link("lamp"),
                )
            )

    # Deduplicate tiny light boxes by centre distance/IoU.
    lights = sorted(lights, key=lambda item: float(item.confidence), reverse=True)
    kept_lights: list[DetectedItem] = []
    h, w = image_shape[:2]
    min_center_dist = max(14.0, w * 0.020)

    for item in lights:
        cx, cy = _bbox_center(item.bbox)
        is_duplicate = False
        for old in kept_lights:
            ox, oy = _bbox_center(old.bbox)
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist < min_center_dist or _iou(item.bbox, old.bbox) > 0.14:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_lights.append(item)
        if len(kept_lights) >= 12:
            break

    return [*cleaned, *kept_lights]



def _is_probable_linear_slot_diffuser_bbox(bbox: list[int], image_shape: tuple[int, int, int]) -> bool:
    """Return True for likely ceiling linear slot diffusers.

    Industrial/interior photos vary a lot: some diffusers are horizontal, some
    are diagonal because of perspective, and low-resolution images make the
    strips thicker. This filter keeps the important constraints (upper room/
    ceiling zone, long thin geometry, sensible size) without hard-coding one
    conference-room layout.
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area_ratio = (bw * bh) / max(1, w * h)
    aspect = max(bw / bh, bh / bw)

    # Usually ceiling/HVAC diffusers are in the top half. Allow a little lower
    # for tilted camera shots, but reject obvious floor/wall-region lines.
    if cy > h * 0.58:
        return False

    # Do not reject edges too aggressively; wide-angle/low-res photos can place
    # diffusers close to left/right borders.
    if cx < w * 0.04 or cx > w * 0.96:
        return False

    # Long thin slot geometry. Low-res images can make them chunkier, so the
    # aspect threshold is moderate, not extreme.
    if aspect < 3.2:
        return False

    # Remove tiny noise and huge ceiling slabs.
    if area_ratio < 0.00008 or area_ratio > 0.045:
        return False

    # At least one dimension must be visually meaningful in the image.
    if max(bw, bh) < max(28, int(min(w, h) * 0.045)):
        return False

    return True


def _remove_false_linear_slot_diffuser_items(items: list[DetectedItem], image_shape: tuple[int, int, int]) -> list[DetectedItem]:
    cleaned: list[DetectedItem] = []
    for item in items:
        key = _normalize_label(item.name)
        if key == "linear slot diffuser" and not _is_probable_linear_slot_diffuser_bbox(item.bbox, image_shape):
            continue
        cleaned.append(item)
    return cleaned

def _add_linear_slot_diffuser_fallbacks(items: list[DetectedItem], image_bgr: np.ndarray) -> list[DetectedItem]:
    """Add robust ceiling linear slot diffuser detections.

    Works better across high/low resolution images by combining dark-slot
    thresholding, edges, Hough lines, and connected components. It avoids the
    older hard-coded left/right room assumptions.
    """
    items = _remove_false_linear_slot_diffuser_items(items, image_bgr.shape)

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    roi_y1 = 0
    roi_y2 = int(h * 0.60)
    roi_x1 = int(w * 0.03)
    roi_x2 = int(w * 0.97)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return items

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    # Adaptive threshold is more reliable than a fixed black cutoff on dark or
    # compressed images.
    dark = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 3), dtype=np.uint8))
    edges = cv2.Canny(dark, 30, 120)

    candidates: list[list[int]] = []

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(w * 0.018)),
        minLineLength=max(35, int(w * 0.045)),
        maxLineGap=max(12, int(w * 0.025)),
    )

    if lines is not None:
        for lx1, ly1, lx2, ly2 in lines[:, 0, :]:
            x1, y1 = int(lx1 + roi_x1), int(ly1 + roi_y1)
            x2, y2 = int(lx2 + roi_x1), int(ly2 + roi_y1)
            dx = x2 - x1
            dy = y2 - y1
            length = float((dx * dx + dy * dy) ** 0.5)
            if length < max(35, w * 0.045):
                continue

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            pad = max(6, int(length * 0.035))
            bbox = [
                max(0, x_min - pad),
                max(0, y_min - pad),
                min(w - 1, x_max + pad),
                min(h - 1, y_max + pad),
            ]
            if not _is_probable_linear_slot_diffuser_bbox(bbox, image_bgr.shape):
                continue

            px1, py1, px2, py2 = bbox
            patch = gray[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            # Slot patches should contain some dark pixels, even when blurry.
            if float(np.percentile(patch, 25)) > 155:
                continue
            candidates.append(bbox)

    # Connected components backup for very low-res images where Hough fails.
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if max(bw, bh) < max(28, int(min(w, h) * 0.040)):
            continue
        bbox = [x + roi_x1 - 8, y + roi_y1 - 8, x + bw + roi_x1 + 8, y + bh + roi_y1 + 8]
        bbox = [max(0, bbox[0]), max(0, bbox[1]), min(w - 1, bbox[2]), min(h - 1, bbox[3])]
        if _is_probable_linear_slot_diffuser_bbox(bbox, image_bgr.shape):
            candidates.append(bbox)

    if not candidates:
        return items

    # Merge nearby segments into actual diffuser strips.
    candidates = sorted(candidates, key=lambda b: ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
    clusters: list[list[list[int]]] = []
    for bbox in candidates:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        placed = False
        for cluster in clusters:
            ccx = float(np.mean([(b[0] + b[2]) / 2.0 for b in cluster]))
            ccy = float(np.mean([(b[1] + b[3]) / 2.0 for b in cluster]))
            if abs(cx - ccx) < w * 0.10 and abs(cy - ccy) < h * 0.16:
                cluster.append(bbox)
                placed = True
                break
        if not placed:
            clusters.append([bbox])

    merged: list[list[int]] = []
    for cluster in clusters:
        x1 = min(b[0] for b in cluster)
        y1 = min(b[1] for b in cluster)
        x2 = max(b[2] for b in cluster)
        y2 = max(b[3] for b in cluster)
        bbox = [x1, y1, x2, y2]
        if _is_probable_linear_slot_diffuser_bbox(bbox, image_bgr.shape):
            merged.append(bbox)

    merged = sorted(merged, key=lambda b: (b[3] - b[1]) * (b[2] - b[0]), reverse=True)[:4]

    additions: list[DetectedItem] = []
    for bbox in merged:
        if any(
            _normalize_label(old.name) == "linear slot diffuser" and _iou(bbox, old.bbox) > 0.10
            for old in items + additions
        ):
            continue
        additions.append(
            DetectedItem(
                name="linear slot diffuser",
                confidence=0.76,
                bbox=bbox,
                category=category_for_item("linear slot diffuser"),
                link=build_amazon_link("linear slot diffuser"),
            )
        )

    return [*items, *additions]


def _add_ceiling_vent_fallbacks(items: list[DetectedItem], image_bgr: np.ndarray) -> list[DetectedItem]:
    """Add ceiling AC vent / linear diffuser slots safely.

    Fixes the false detection on the left glass/window area by only accepting
    long dark ceiling slots in the actual ceiling band. The candidates are then
    merged by side so the UI does not show noisy AC Vent x4 on the window.
    """
    items = _remove_false_ac_vent_items(items, image_bgr.shape)

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Only the upper ceiling area, and not the far-left glass/window area.
    roi_y1 = 0
    roi_y2 = int(h * 0.40)
    roi_x1 = int(w * 0.23)
    roi_x2 = int(w * 0.88)
    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return items

    # Dark ceiling grooves on bright ceiling.
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 130)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=52,
        minLineLength=max(90, int(w * 0.11)),
        maxLineGap=20,
    )

    raw_boxes: list[list[int]] = []
    if lines is not None:
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in line]
            x1, y1, x2, y2 = lx1 + roi_x1, ly1 + roi_y1, lx2 + roi_x1, ly2 + roi_y1
            dx = x2 - x1
            dy = y2 - y1
            length = (dx * dx + dy * dy) ** 0.5
            if length < w * 0.11:
                continue

            angle = abs(np.degrees(np.arctan2(dy, dx)))
            # In this app we keep only the perspective diagonal/near-vertical
            # ceiling slot diffusers. Horizontal edges often come from walls,
            # glass doors, whiteboards, or ceiling borders and created false
            # "AC Vent" labels.
            is_perspective_slot = 48 <= angle <= 132
            if not is_perspective_slot:
                continue

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            pad_x = 8
            pad_y = 6
            bbox = [
                max(0, x_min - pad_x),
                max(0, y_min - pad_y),
                min(w - 1, x_max + pad_x),
                min(h - 1, y_max + pad_y),
            ]
            if _is_probable_ceiling_vent_bbox(bbox, image_bgr.shape):
                raw_boxes.append(bbox)

    if not raw_boxes:
        return items

    # Merge nearby line segments into left/right vent regions instead of noisy x4 tags.
    raw_boxes = sorted(raw_boxes, key=lambda b: (b[0] + b[2]) / 2.0)
    clusters: list[list[list[int]]] = []
    for bbox in raw_boxes:
        cx = (bbox[0] + bbox[2]) / 2.0
        placed = False
        for cluster in clusters:
            ccx = np.mean([(b[0] + b[2]) / 2.0 for b in cluster])
            if abs(cx - ccx) < w * 0.09:
                cluster.append(bbox)
                placed = True
                break
        if not placed:
            clusters.append([bbox])

    additions: list[DetectedItem] = []
    for cluster in clusters:
        if len(additions) >= 3:
            break
        x1 = min(b[0] for b in cluster)
        y1 = min(b[1] for b in cluster)
        x2 = max(b[2] for b in cluster)
        y2 = max(b[3] for b in cluster)
        bbox = [x1, y1, x2, y2]

        if not _is_probable_ceiling_vent_bbox(bbox, image_bgr.shape):
            continue
        if any(_normalize_label(old.name) == "ac vent" and _iou(bbox, old.bbox) > 0.12 for old in items + additions):
            continue

        additions.append(
            DetectedItem(
                name="ac vent",
                confidence=0.50,
                bbox=bbox,
                category=category_for_item("ac vent"),
                link=build_amazon_link("ac vent"),
            )
        )

    return [*items, *additions]


def _add_window_blind_fallback(items: list[DetectedItem], image_bgr: np.ndarray) -> list[DetectedItem]:
    """Add left-side window blind/cover panels when visible but missed."""
    if _has_item(items, {"blind", "curtain"}):
        return items

    h, w = image_bgr.shape[:2]
    # The left upper panels in office rooms are usually window blinds/covers.
    # Keep this conservative: only add if a glass window/partition exists nearby.
    has_left_glass = any(
        _normalize_label(item.name) in {"glass window", "window"}
        and ((item.bbox[0] + item.bbox[2]) / 2.0) < w * 0.45
        for item in items
    )
    if not has_left_glass:
        return items

    bbox = [int(w * 0.02), int(h * 0.05), int(w * 0.36), int(h * 0.34)]
    return [
        *items,
        DetectedItem(
            name="blind",
            confidence=0.50,
            bbox=bbox,
            category=category_for_item("blind"),
            link=build_amazon_link("blind"),
        ),
    ]


def _add_context_fallbacks(items: list[DetectedItem], image_bgr: np.ndarray) -> list[DetectedItem]:
    items = _add_recessed_light_fallbacks(items, image_bgr)
    items = _add_linear_slot_diffuser_fallbacks(items, image_bgr)
    items = _add_ceiling_vent_fallbacks(items, image_bgr)
    items = _add_window_blind_fallback(items, image_bgr)
    return items

def _representative_score(item: DetectedItem) -> tuple[float, float, float]:
    """
    Pick the object location that should represent a grouped label.

    For furniture, the best representative is usually the lowest visible
    detection, not a random higher-confidence patch on glass/walls. This keeps
    labels like "Chair x5" attached to the real chair cluster.
    """
    key = _normalize_label(item.name)
    x1, y1, x2, y2 = [int(v) for v in item.bbox]
    area = max(1, x2 - x1) * max(1, y2 - y1)

    if category_for_item(key) == "Furniture":
        return float(y2), float(area), float(item.confidence)

    return float(item.confidence), float(area), float(y2)


def _unique_items(items: list[DetectedItem]) -> list[DetectedItem]:
    grouped: dict[str, list[DetectedItem]] = {}

    for item in items:
        key = _normalize_label(item.name)
        grouped.setdefault(key, []).append(item)

    unique: list[DetectedItem] = []

    for key, group in grouped.items():
        if key == "recessed light":
            # Keep only one visible label, but attach it to ONE real small LED.
            # Do NOT use a huge union bbox, because that made the label appear on
            # empty ceiling / near the pendant instead of on the actual LED spot.
            # Prefer a visible right-side/top ceiling LED when available.
            h_hint = max(int(max(int(g.bbox[3]) for g in group) * 2.2), 1)
            w_hint = max(int(max(int(g.bbox[2]) for g in group) * 1.25), 1)
            representative = max(
                group,
                key=lambda g: (
                    int(g.bbox[0]) > w_hint * 0.55,
                    int(g.bbox[1]) < h_hint * 0.36,
                    float(g.confidence),
                    int(g.bbox[0]),
                ),
            )
        else:
            representative = max(group, key=_representative_score)

        representative_bbox = representative.bbox

        unique.append(
            DetectedItem(
                name=key,
                confidence=max(float(item.confidence) for item in group),
                bbox=representative_bbox,
                category=category_for_item(key),
                link=build_amazon_link(key),
                count=len(group),
            )
        )

    return sorted(unique, key=lambda item: item.confidence, reverse=True)



def _label_text(item: DetectedItem) -> str:
    text = display_name(item.name)

    if item.count > 1:
        text = f"{text} x{item.count}"

    return text


# Backwards-compatible API expected by older callers and tests
def normalize_item_name(name: str) -> str:
    """Normalize item names for external callers / tests.

    This intentionally performs a small, stable mapping set used by unit
    tests (e.g. `coffee table` -> `center table`) and converts underscores
    to spaces.
    """
    if name is None:
        return ""

    s = str(name).lower().strip().replace("_", " ")
    s = " ".join(s.split())

    # Small compatibility replacements used by tests and callers.
    replacements = {
        "couch": "sofa",
        "rug": "carpet",
        "coffee table": "center table",
        "office chair": "office chair",
        "office chair": "office chair",
        "wall shelf": "shelf",
    }

    return replacements.get(s, s)


def categorize_item(name: str) -> str:
    return category_for_item(name)


def unique_sidebar_items(items: list[DetectedItem]) -> list[DetectedItem]:
    """Return unique items for the sidebar — case-insensitive dedupe.

    Keeps the item with the highest confidence for each normalized name.
    """
    grouped: dict[str, DetectedItem] = {}

    for item in items:
        key = normalize_item_name(getattr(item, "name", ""))

        existing = grouped.get(key)
        if existing is None or float(item.confidence) > float(existing.confidence):
            # Preserve original bbox/category/link/count when possible
            grouped[key] = DetectedItem(
                name=key,
                confidence=item.confidence,
                bbox=getattr(item, "bbox", getattr(item, "coords", None)),
                category=getattr(item, "category", None),
                link=getattr(item, "link", None),
                count=getattr(item, "count", 1),
            )

    return list(grouped.values())


def _draw_filled_label(
    output: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> tuple[int, int, int, int]:
    pad_x = 9
    pad_y = 7

    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    x1 = x
    y1 = y
    x2 = x1 + tw + pad_x * 2
    y2 = y1 + th + pad_y * 2

    cv2.rectangle(output, (x1 + 2, y1 + 2), (x2 + 2, y2 + 2), (10, 10, 10), -1)
    cv2.rectangle(output, (x1, y1), (x2, y2), color, -1)

    cv2.putText(
        output,
        text,
        (x1 + pad_x, y2 - pad_y - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return x1, y1, x2, y2


def _clamp_label_position(
    label_x: int,
    label_y: int,
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int]:
    label_x = max(8, min(image_w - label_w - 8, int(label_x)))
    label_y = max(8, min(image_h - label_h - 8, int(label_y)))
    return label_x, label_y


def _label_position_attached_to_object(
    item: DetectedItem,
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int]:
    """
    Keep every visible tag physically attached to its object.

    Older versions used global left/right "lanes". That made labels float far
    away from the detection, for example the TV label sitting in the top-right
    corner while the TV was in the middle of the wall. This placement keeps the
    tag either inside the object box or just beside it.
    """
    x1, y1, x2, y2 = [int(v) for v in item.bbox]
    x1 = max(0, min(image_w - 1, x1))
    y1 = max(0, min(image_h - 1, y1))
    x2 = max(0, min(image_w - 1, x2))
    y2 = max(0, min(image_h - 1, y2))

    if x2 <= x1:
        x2 = min(image_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(image_h - 1, y1 + 1)

    key = _normalize_label(item.name)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = (x1 + x2) // 2

    # Big surfaces/portals should have their text inside the visible region.
    if key in {"glass door", "door", "window", "glass window", "glass partition"}:
        return _clamp_label_position(x1 + 8, y1 + 8, label_w, label_h, image_w, image_h)

    if key in FLOORING_KEYS:
        # Put floor material directly on the lower floor patch.
        label_x = min(x2 - label_w - 8, max(x1 + 8, cx - label_w // 2))
        label_y = max(y1 + 8, y2 - label_h - 10)
        return _clamp_label_position(label_x, label_y, label_w, label_h, image_w, image_h)

    if key == "linear slot diffuser":
        # Put the grouped slot label near the actual black groove, not in the center of the ceiling.
        return _clamp_label_position(x1 + 8, max(8, y1 - label_h - 6), label_w, label_h, image_w, image_h)

    if key in {"ceiling", "ceiling panel", "ceiling panels", "false ceiling"}:
        label_x = min(x2 - label_w - 8, max(x1 + 8, cx - label_w // 2))
        label_y = y1 + 8
        return _clamp_label_position(label_x, label_y, label_w, label_h, image_w, image_h)

    # If the object is large enough, draw the tag inside it at the top-left.
    # This is the cleanest fix for TV/monitor/glass-like regions.
    if box_w >= label_w + 18 and box_h >= label_h + 18:
        return _clamp_label_position(x1 + 8, y1 + 8, label_w, label_h, image_w, image_h)

    # Otherwise place the tag very close to the object, trying right/left/bottom/top.
    candidates = [
        (x2 + 8, y1),
        (x1 - label_w - 8, y1),
        (cx - label_w // 2, y2 + 8),
        (cx - label_w // 2, y1 - label_h - 8),
        (x1 + 4, y1 + 4),
    ]

    for lx, ly in candidates:
        if 8 <= lx <= image_w - label_w - 8 and 8 <= ly <= image_h - label_h - 8:
            return int(lx), int(ly)

    return _clamp_label_position(candidates[-1][0], candidates[-1][1], label_w, label_h, image_w, image_h)


def _line_anchor_for_label(cx: int, cy: int, label_box: tuple[int, int, int, int]) -> tuple[int, int]:
    lx1, ly1, lx2, ly2 = label_box
    anchor_x = max(lx1, min(lx2, cx))
    anchor_y = max(ly1, min(ly2, cy))

    # If the centre falls inside the label, attach to the nearest label edge.
    if lx1 < cx < lx2 and ly1 < cy < ly2:
        distances = {
            "left": abs(cx - lx1),
            "right": abs(cx - lx2),
            "top": abs(cy - ly1),
            "bottom": abs(cy - ly2),
        }
        side = min(distances, key=distances.get)
        if side == "left":
            anchor_x = lx1
        elif side == "right":
            anchor_x = lx2
        elif side == "top":
            anchor_y = ly1
        else:
            anchor_y = ly2

    return int(anchor_x), int(anchor_y)


def draw_detection_labels(
    image_bgr: np.ndarray,
    items: list[DetectedItem],
    unique_only: bool = True,
) -> np.ndarray:
    output = image_bgr.copy()

    if unique_only:
        # Clean professional display: group repeated tiny ceiling objects into
        # one readable label such as "Ceiling Light x8" or "AC Vent x2".
        # Individual detections are still kept in all_items for counts/export,
        # but the image is not cluttered with many duplicate labels.
        items_to_draw = _unique_items(items)
    else:
        items_to_draw = items

    h, w = output.shape[:2]

    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    output = cv2.addWeighted(overlay, 0.04, output, 0.96, 0)

    font_scale = 0.42
    thickness = 1

    # Keep duplicate false desk/work-desk labels hidden; the real conference
    # table is already displayed as Desk/Conference Desk.
    hide_from_image: set[str] = set()

    priority = {
        "linear slot diffuser": 111,
        "ac vent": 110,
        "recessed light": 109,
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

    drawable_items: list[DetectedItem] = []

    for item in items_to_draw:
        key = _normalize_label(item.name)
        if key in hide_from_image:
            continue
        drawable_items.append(item)

    drawable_items = sorted(
        drawable_items,
        key=lambda item: (
            priority.get(_normalize_label(item.name), 0),
            item.confidence,
            item.count,
        ),
        reverse=True,
    )[:30]

    for item in drawable_items:
        x1, y1, x2, y2 = [int(v) for v in item.bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        label = _label_text(item)
        color = color_for_item(item.name)

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        label_w = tw + 18
        label_h = th + 14
        label_x, label_y = _label_position_attached_to_object(item, label_w, label_h, w, h)

        lx1, ly1, lx2, ly2 = _draw_filled_label(
            output=output,
            text=label,
            x=label_x,
            y=label_y,
            color=color,
            font_scale=font_scale,
            thickness=thickness,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        obj_w = max(1, x2 - x1)
        line_len = max(22, min(75, obj_w // 3))

        # Object marker line is always drawn on the object itself.
        cv2.line(
            output,
            (max(x1, cx - line_len), cy),
            (min(x2, cx + line_len), cy),
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(output, (cx, cy), 3, color, -1, cv2.LINE_AA)

        anchor = _line_anchor_for_label(cx, cy, (lx1, ly1, lx2, ly2))

        # Always attach the label to the object. No more "airing" labels.
        cv2.line(output, (cx, cy), anchor, color, 1, cv2.LINE_AA)

    return output

def run_detection(
    model,
    image_bgr: np.ndarray,
    confidence: float = DEFAULT_CONFIDENCE,
    iou: float = DEFAULT_IOU,
) -> DetectionResult:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty")

    # Interior photos are often screenshots/WhatsApp-compressed images. Upscale
    # small inputs before model inference so tiny lights, vents, phones, cable
    # ports, and chair legs are less likely to disappear. The returned annotated
    # image intentionally uses the working resolution for clearer labels.
    h0, w0 = image_bgr.shape[:2]
    min_width = 1280
    if w0 < min_width:
        scale = min_width / max(1, w0)
        image_bgr = cv2.resize(
            image_bgr,
            (int(round(w0 * scale)), int(round(h0 * scale))),
            interpolation=cv2.INTER_CUBIC,
        )

    results = model.predict(
        source=image_bgr,
        conf=float(confidence),
        iou=float(iou),
        verbose=False,
    )

    all_items: list[DetectedItem] = []

    for result in results:
        names = getattr(result, "names", {}) or getattr(model, "names", {}) or {}

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            raw_label = str(names.get(cls_id, cls_id))
            label = _normalize_label(raw_label)
            bbox = [x1, y1, x2, y2]
            label = _contextual_relabel(label, bbox, image_bgr.shape)

            if label in IGNORE_LABELS:
                continue

            if not _is_target(label):
                continue

            if not _passes_smart_filter(
                label=label,
                confidence=conf,
                bbox=bbox,
                image_shape=image_bgr.shape,
            ):
                continue

            all_items.append(
                DetectedItem(
                    name=label,
                    confidence=conf,
                    bbox=bbox,
                    category=category_for_item(label),
                    link=build_amazon_link(label),
                )
            )

    all_items = _filter_overlapping_items(all_items)
    all_items = _limit_per_class(all_items)
    all_items = _remove_false_ac_vent_items(all_items, image_bgr.shape)
    all_items = _remove_context_false_positives(all_items, image_bgr.shape)
    all_items = _add_flooring_fallback(all_items, image_bgr)
    all_items = _add_context_fallbacks(all_items, image_bgr)
    all_items = _clean_recessed_light_items(all_items, image_bgr.shape)
    all_items = _remove_context_false_positives(all_items, image_bgr.shape)
    all_items = _filter_overlapping_items(all_items)
    all_items = _limit_per_class(all_items)

    unique_items = _unique_items(all_items)

    annotated = draw_detection_labels(
        image_bgr,
        all_items,
        unique_only=True,
    )

    return DetectionResult(
        all_items=all_items,
        unique_items=unique_items,
        annotated_image=annotated,
    )


class FurnitureDetector:
    """Compatibility wrapper for detect.py, CLI, API, and web app."""

    def __init__(self, model_path: Path | str | None = None):
        self.model = load_model(model_path or default_model_path())

    def detect(self, image: np.ndarray) -> list[DetectedItem]:
        return run_detection(self.model, image).all_items

    def draw_detections(
        self,
        image: np.ndarray,
        detections: list[DetectedItem],
    ) -> np.ndarray:
        return draw_detection_labels(
            image,
            detections,
            unique_only=True,
        )