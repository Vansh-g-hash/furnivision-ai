"""Tests for detector module."""

import numpy as np
import pytest

from ai_furniture_detector.detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DetectedItem,
    categorize_item,
    normalize_item_name,
    unique_sidebar_items,
)


class TestItemNormalization:
    """Test item name normalization."""

    def test_normalize_simple_name(self):
        """Test normalization of simple names."""
        assert normalize_item_name("CHAIR") == "chair"
        assert normalize_item_name("   office chair   ") == "office chair"

    def test_normalize_aliases(self):
        """Test alias normalization."""
        assert normalize_item_name("couch") == "sofa"
        assert normalize_item_name("rug") == "carpet"
        assert normalize_item_name("coffee table") == "center table"

    def test_normalize_underscores(self):
        """Test underscore replacement."""
        assert normalize_item_name("office_chair") == "office chair"
        assert normalize_item_name("wall_shelf") == "shelf"


class TestItemCategorization:
    """Test item categorization."""

    def test_categorize_furniture(self):
        """Test furniture category."""
        assert categorize_item("chair") == "Furniture"
        assert categorize_item("table") == "Furniture"
        assert categorize_item("desk") == "Furniture"

    def test_categorize_lighting(self):
        """Test lighting category."""
        assert categorize_item("lamp") == "Lighting"
        assert categorize_item("ceiling light") == "Lighting"

    def test_categorize_doors_windows(self):
        """Test doors & windows category."""
        assert categorize_item("door") == "Doors & Windows"
        assert categorize_item("window") == "Doors & Windows"

    def test_categorize_decor(self):
        """Test decor category."""
        assert categorize_item("plant") == "Decor"
        assert categorize_item("painting") == "Decor"

    def test_categorize_other(self):
        """Test other category fallback."""
        assert categorize_item("unknown_item") == "Other"


class TestDetectedItem:
    """Test DetectedItem dataclass."""

    def test_detected_item_creation(self):
        """Test creating detected item."""
        item = DetectedItem(
            name="chair",
            confidence=0.95,
            coords=(10, 20, 100, 150),
            category="Furniture",
        )
        assert item.name == "chair"
        assert item.confidence == 0.95
        assert item.coords == (10, 20, 100, 150)

    def test_detected_item_area(self):
        """Test area calculation."""
        item = DetectedItem(
            name="table",
            confidence=0.85,
            coords=(0, 0, 100, 200),
            category="Furniture",
        )
        assert item.area == 20000  # 100 * 200

    def test_detected_item_contains(self):
        """Test point containment check."""
        item = DetectedItem(
            name="sofa",
            confidence=0.90,
            coords=(10, 20, 110, 120),
            category="Furniture",
        )
        assert item.contains(50, 70)
        assert not item.contains(0, 0)
        assert not item.contains(200, 200)

    def test_detected_item_to_dict(self):
        """Test dictionary conversion."""
        item = DetectedItem(
            name="desk",
            confidence=0.88,
            coords=(5, 15, 95, 105),
            category="Furniture",
        )
        d = item.to_dict()
        assert d["name"] == "desk"
        assert d["confidence"] == 0.88
        assert isinstance(d["coords"], list)
        assert "link" in d


class TestUniqueSidebarItems:
    """Test unique item deduplication."""

    def test_single_item(self):
        """Test with single item."""
        item = DetectedItem(
            name="chair",
            confidence=0.90,
            coords=(0, 0, 50, 50),
            category="Furniture",
        )
        result = unique_sidebar_items([item])
        assert len(result) == 1
        assert result[0].name == "chair"

    def test_duplicate_items_keeps_best(self):
        """Test that duplicate items keep highest confidence."""
        items = [
            DetectedItem("chair", 0.75, (0, 0, 50, 50), "Furniture"),
            DetectedItem("chair", 0.95, (60, 60, 100, 100), "Furniture"),
            DetectedItem("chair", 0.85, (120, 120, 150, 150), "Furniture"),
        ]
        result = unique_sidebar_items(items)
        assert len(result) == 1
        assert result[0].confidence == 0.95

    def test_multiple_different_items(self):
        """Test with multiple different items."""
        items = [
            DetectedItem("chair", 0.90, (0, 0, 50, 50), "Furniture"),
            DetectedItem("table", 0.85, (60, 60, 100, 100), "Furniture"),
            DetectedItem("lamp", 0.88, (120, 120, 150, 150), "Lighting"),
        ]
        result = unique_sidebar_items(items)
        assert len(result) == 3

    def test_case_insensitive_dedup(self):
        """Test that deduplication is case-insensitive."""
        items = [
            DetectedItem("CHAIR", 0.75, (0, 0, 50, 50), "Furniture"),
            DetectedItem("chair", 0.95, (60, 60, 100, 100), "Furniture"),
        ]
        result = unique_sidebar_items(items)
        assert len(result) == 1
