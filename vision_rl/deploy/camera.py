"""OpenCV camera source for real deployment.

The camera side of sim-to-real: grab frames from a USB camera and return them as
uint8 RGB at the model's input resolution. Same logic as
scripts/camera_84x84.py (resize with INTER_AREA, BGR->RGB), so what the policy
sees on hardware matches that tool's output. Joints come from LeRobot; frames
come from here.
"""

from __future__ import annotations

import numpy as np


def parse_device(value) -> int:
    """Accept a camera index ('0') or a path ('/dev/video0')."""
    value = str(value)
    if value.startswith("/dev/video"):
        return int(value.replace("/dev/video", ""))
    return int(value)


class OpenCVCamera:
    """A USB camera that yields square uint8 RGB frames at `size`x`size`.

    Mirrors scripts/camera_84x84.get_frame: a 4:3 frame is resized straight to a
    square (aspect ratio is squashed the same way that tool does) -- keep the
    physical camera framed on the workspace so the squash is consistent with what
    you'd calibrate against.
    """

    def __init__(self, device="0", size: int = 84, warmup: int = 5):
        import cv2

        self._cv2 = cv2
        self.size = int(size)
        self.cap = cv2.VideoCapture(parse_device(device))
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera device {device!r}")
        # Drop the first few frames while auto-exposure/white-balance settle.
        for _ in range(max(0, warmup)):
            self.cap.read()

    def read(self) -> np.ndarray:
        """One frame as (size, size, 3) uint8 RGB."""
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("failed to read frame from camera")
        cv2 = self._cv2
        small = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass
