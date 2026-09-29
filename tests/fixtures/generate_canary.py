"""
Generate a synthetic canary_frame.jpg for testing.

Creates a 640x480 image with a person-shaped blob so YOLO has
something detectable. Run once:
    python tests/fixtures/generate_canary.py
"""
import cv2
import numpy as np
from pathlib import Path

def generate():
    out = Path(__file__).parent / "canary_frame.jpg"
    img = np.ones((480, 640, 3), dtype=np.uint8) * 40  # dark background

    # Draw rough person silhouette (head + body + legs)
    cx, cy = 320, 240
    # Head
    cv2.circle(img, (cx, cy - 80), 35, (200, 180, 160), -1)
    # Body
    cv2.rectangle(img, (cx - 30, cy - 45), (cx + 30, cy + 60), (100, 100, 180), -1)
    # Arms
    cv2.rectangle(img, (cx - 60, cy - 40), (cx - 30, cy + 40), (100, 100, 180), -1)
    cv2.rectangle(img, (cx + 30, cy - 40), (cx + 60, cy + 40), (100, 100, 180), -1)
    # Legs
    cv2.rectangle(img, (cx - 28, cy + 60), (cx, cy + 130), (80, 80, 160), -1)
    cv2.rectangle(img, (cx, cy + 60), (cx + 28, cy + 130), (80, 80, 160), -1)

    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Canary frame written: {out}")

if __name__ == "__main__":
    generate()
