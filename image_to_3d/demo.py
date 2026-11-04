"""A synthetic glasses image, so the pipeline can run without a product photo."""
import math
from pathlib import Path

import cv2
import numpy as np

CANVAS_SIZE = (320, 800)  # height, width
BACKGROUND = (255, 255, 255)
FRAME_COLOR = (35, 28, 27)  # BGR
RIM_THICKNESS = 18
LENS_CENTERS = ((230, 160), (570, 160))
LENS_HALF_SIZE = (150, 105)
LENS_SEGMENTS = 96


def _lens_outline(center):
    """Rounded-rectangle (superellipse) outline, as int32 pixel points."""
    angles = np.linspace(0, 2 * math.pi, LENS_SEGMENTS, endpoint=False)
    # Exponents below 1 square the corners off an ellipse.
    x = center[0] + LENS_HALF_SIZE[0] * np.sign(np.cos(angles)) * np.abs(np.cos(angles)) ** 0.5
    y = center[1] + LENS_HALF_SIZE[1] * np.sign(np.sin(angles)) * np.abs(np.sin(angles)) ** 0.5
    return np.round(np.stack((x, y), axis=1)).astype(np.int32)


def make_demo_image():
    """Dark rims, a bridge, and hinge stubs on white, as a BGR array."""
    image = np.full((*CANVAS_SIZE, 3), BACKGROUND, np.uint8)
    for center in LENS_CENTERS:
        cv2.polylines(image, [_lens_outline(center)], True, FRAME_COLOR, RIM_THICKNESS, cv2.LINE_AA)
    cv2.ellipse(image, (400, 130), (25, 18), 0, 180, 360, FRAME_COLOR, 14, cv2.LINE_AA)
    cv2.line(image, (85, 125), (45, 110), FRAME_COLOR, 14, cv2.LINE_AA)
    cv2.line(image, (715, 125), (755, 110), FRAME_COLOR, 14, cv2.LINE_AA)
    return image


def write_demo_image(path):
    """Save the demo image as a PNG; returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _, encoded = cv2.imencode(".png", make_demo_image())
    path.write_bytes(encoded.tobytes())
    return path
