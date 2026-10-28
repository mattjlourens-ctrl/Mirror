"""Pure 2D geometry helpers. Points are (x, y) tuples in image coordinates,
where x grows to the right and y grows DOWN."""
import math


def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def midpoint(p1, p2):
    """Point halfway between p1 and p2."""
    return (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2


def tilt_angle(p1, p2):
    """Angle (radians) of the line through p1 and p2, measured left-to-right.

    The points are ordered by x first, so swapping p1 and p2 gives the same
    answer and the result stays within -90..90 degrees (no 180 degree flips).
    Because y points down, a positive angle means a clockwise tilt on screen.
    """
    left, right = sorted((p1, p2))
    return math.atan2(right[1] - left[1], right[0] - left[0])


def transform_point(point, scale, angle, translation):
    """Scale, then rotate (radians, about the origin), then translate a point."""
    x = point[0] * scale
    y = point[1] * scale
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        x * cos_a - y * sin_a + translation[0],
        x * sin_a + y * cos_a + translation[1],
    )


def transform_points(points, scale, angle, translation):
    """Apply transform_point to every point in a list."""
    return [transform_point(p, scale, angle, translation) for p in points]
