"""
Detect leaves in the first frame of a video using YOLOv8.
"""

import argparse
import sys

import cv2
from ultralytics import YOLO

DEFAULT_DET_MODEL = "yolov8n.pt"
DEFAULT_SEG_MODEL = "yolov8n-seg.pt"
DEFAULT_CONF = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect leaves in the first frame of a video.")
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument(
        "--mode",
        choices=("bbox", "seg"),
        default="bbox",
        help="Detection mode: bbox (rectangles) or seg (instance segmentation).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="YOLO model to use. If omitted, picks yolov8n.pt for bbox or yolov8n-seg.pt for seg.",
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold.")
    parser.add_argument("--out", default="leaves_detected.png", help="Output image path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read first frame from {args.video}")
        sys.exit(1)

    model_path = args.model
    if model_path is None:
        model_path = DEFAULT_SEG_MODEL if args.mode == "seg" else DEFAULT_DET_MODEL

    model = YOLO(model_path)
    results = model(frame, conf=args.conf)

    annotated = results[0].plot()
    cv2.imwrite(args.out, annotated)
    print(f"Saved: {args.out}")
    print(f"Mode: {args.mode} | Model: {model_path}")

    # Print what was detected
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        print(f"  {cls_name}: {conf:.2f}")

    if args.mode == "seg":
        if results[0].masks is None:
            print(
                "No masks returned. Check if the selected model supports segmentation "
                "(e.g., yolov8n-seg.pt)."
            )
        else:
            print(f"  Masks detected: {len(results[0].masks)}")


if __name__ == "__main__":
    main()
