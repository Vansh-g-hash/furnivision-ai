import argparse
import json
from pathlib import Path

import cv2

from .detector import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    default_model_path,
    load_image_bgr,
    load_model,
    run_detection,
)

# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AI Furniture Detector on a local room image."
    )

    parser.add_argument(
        "--source",
        "--image",
        dest="source",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to YOLO weights file",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE})",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save annotated image",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Path to save detection JSON",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the annotated image window",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose console output",
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Loading image: {args.source}")

    image = load_image_bgr(args.source)
    model = load_model(args.model) if args.model else load_model(default_model_path())
    result = run_detection(
        model,
        image,
        confidence=args.confidence,
        iou=args.iou,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), result.annotated_image):
            raise RuntimeError(f"Failed to save annotated image to {output_path}")
        print(f"Saved annotated image to {output_path}")

    if args.output_json:
        payload = {
            "filename": Path(args.source).name,
            "total_boxes": len(result.all_items),
            "unique_count": len(result.unique_items),
            "unique_items": [item.to_dict() for item in result.unique_items],
            "all_items": [item.to_dict() for item in result.all_items],
        }
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved detection JSON to {output_json_path}")

    if not args.no_show:
        cv2.imshow("AI Furniture Detection", result.annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    raise SystemExit(main())
