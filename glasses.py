"""A simple 2D glasses frame, defined in local coordinates and drawn with OpenCV.

Local coordinate system:
  - (0, 0) is the centre of the glasses (the nose bridge).
  - x points right, y points down (same directions as the image).
  - 1.0 unit = the width of the frame front, so the shape is independent of
    face size. Scaling by the frame width in pixels gives the on-screen size.

To try a different style, change the shapes below; the transform stays the same.
"""
import cv2
import numpy as np

from geometry import transform_points

FRAME_COLOR = (40, 40, 40)  # BGR, dark grey


def _rectangle(left, top, right, bottom):
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


# Each part is a list of local points joined by straight lines.
LEFT_LENS = _rectangle(-0.50, -0.15, -0.05, 0.15)
RIGHT_LENS = _rectangle(0.05, -0.15, 0.50, 0.15)
BRIDGE = [(-0.05, -0.05), (0.0, -0.08), (0.05, -0.05)]
LEFT_TEMPLE = [(-0.50, -0.12), (-0.58, -0.12)]
RIGHT_TEMPLE = [(0.50, -0.12), (0.58, -0.12)]

CLOSED_PARTS = [LEFT_LENS, RIGHT_LENS]
OPEN_PARTS = [BRIDGE, LEFT_TEMPLE, RIGHT_TEMPLE]


def _to_pixels(points, scale, angle, center):
    """Local points -> int32 pixel array in the shape cv2.polylines expects."""
    screen_points = transform_points(points, scale, angle, center)
    return np.array(screen_points, dtype=np.float32).round().astype(np.int32)


def draw_glasses(frame, center, frame_width, angle):
    """Draw the glasses on `frame`.

    center      -- (x, y) pixel position of the glasses' centre
    frame_width -- width of the frame front in pixels (local 1.0 unit)
    angle       -- head tilt in radians
    """
    thickness = max(2, round(frame_width * 0.02))

    closed = [_to_pixels(part, frame_width, angle, center) for part in CLOSED_PARTS]
    opened = [_to_pixels(part, frame_width, angle, center) for part in OPEN_PARTS]

    cv2.polylines(frame, closed, True, FRAME_COLOR, thickness, cv2.LINE_AA)
    cv2.polylines(frame, opened, False, FRAME_COLOR, thickness, cv2.LINE_AA)
