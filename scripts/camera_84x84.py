#!/usr/bin/env python3
"""Grab frames from a camera, resize to 84x84, and display them.

Usage:
    python scripts/camera_84x84.py                # uses /dev/video0
    python scripts/camera_84x84.py --device 2     # uses /dev/video2
    python scripts/camera_84x84.py --device /dev/video0

Press 'q' (with the window focused) to quit.
"""

import argparse

import cv2
import numpy as np


def parse_device(value: str) -> int:
    """Accept either an index ('0') or a path ('/dev/video0')."""
    if value.startswith("/dev/video"):
        return int(value.replace("/dev/video", ""))
    return int(value)


def get_frame(cap: cv2.VideoCapture, size: int = 84) -> np.ndarray:
    """Read one frame and return it resized to (size, size, 3) as uint8 RGB."""
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame from camera")
    # INTER_AREA is best for shrinking
    small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    # OpenCV gives BGR; convert to RGB so the returned data matches most pipelines
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="0",
        help="Camera index or path, e.g. 0 or /dev/video0 (default: 0)",
    )
    parser.add_argument("--size", type=int, default=84, help="Output size (default: 84)")
    args = parser.parse_args()

    device = parse_device(args.device)
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera device {args.device}")

    print(f"Streaming from device {args.device} at {args.size}x{args.size}. Press 'q' to quit.")
    try:
        while True:
            rgb = get_frame(cap, size=args.size)  # <-- the (size, size, 3) uint8 data

            # `rgb` is the data you can feed to a model / store / send elsewhere.
            # Display it (convert back to BGR because imshow expects BGR).
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # Scale up 6x just so the tiny 84x84 image is visible on screen.
            preview = cv2.resize(bgr, (args.size * 6, args.size * 6), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(f"{args.size}x{args.size}", preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
