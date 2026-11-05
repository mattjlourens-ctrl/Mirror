"""Bezier curves and swept tubes, shared by the temples and the fallback bridge."""
import numpy as np
import trimesh

SECTION_POINTS = 12
SQUARENESS = 0.6  # superellipse exponent: 1 is an ellipse, smaller is more rectangular
UP = np.array((0.0, 1.0, 0.0))


def cubic_bezier(control, t):
    """Points on a cubic Bezier: Bernstein weights applied to 4 control points."""
    t = np.asarray(t, dtype=np.float64)[..., None]
    return ((1 - t) ** 3 * control[0] + 3 * (1 - t) ** 2 * t * control[1]
            + 3 * (1 - t) * t ** 2 * control[2] + t ** 3 * control[3])


def _is_flat(control, tolerance):
    """A cubic stays within 3/4 of its inner control points' distance from its chord."""
    chord = control[3] - control[0]
    inner = control[1:3] - control[0]
    off_chord = inner - np.outer(inner @ chord / (chord @ chord), chord)
    return (np.linalg.norm(off_chord, axis=1) * 0.75 <= tolerance).all()


def _flatten(control, tolerance, points):
    """Append points up to and including the curve's end, splitting by de Casteljau."""
    if _is_flat(control, tolerance):
        points.append(control[3])
        return
    ab, bc, cd = (control[:3] + control[1:]) / 2
    abc, bcd = (ab + bc) / 2, (bc + cd) / 2
    middle = (abc + bcd) / 2
    _flatten(np.array([control[0], ab, abc, middle]), tolerance, points)
    _flatten(np.array([middle, bcd, cd, control[3]]), tolerance, points)


def _without_straight_joints(points):
    directions = np.diff(points, axis=0)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bent = np.linalg.norm(np.cross(directions[:-1], directions[1:]), axis=1) > 1e-9
    return points[np.r_[True, bent, True]]


def flatten_cubics(curves, tolerance):
    """Polyline through joined cubics, with only the points the tolerance needs."""
    points = [curves[0][0]]
    for control in curves:
        _flatten(control, tolerance, points)
    return _without_straight_joints(np.array(points))


def polyline_length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def section_ring(half_a, half_b):
    """Rounded-rectangle section as (N, 2) points, counter-clockwise."""
    angles = np.linspace(0, 2 * np.pi, SECTION_POINTS, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    return np.stack((half_a * np.sign(cos) * np.abs(cos) ** SQUARENESS,
                     half_b * np.sign(sin) * np.abs(sin) ** SQUARENESS), axis=1)


def _tangents(path):
    segments = np.diff(path, axis=0)
    segments /= np.linalg.norm(segments, axis=1, keepdims=True)
    tangents = np.vstack((segments[0], segments[:-1] + segments[1:], segments[-1]))
    return tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def _transported_axes(tangents):
    """Rotation-minimising frames: each axis is the last, turned only as the path turns."""
    axis = UP - tangents[0] * (UP @ tangents[0])
    axis /= np.linalg.norm(axis)
    axes = [axis]
    for before, after in zip(tangents, tangents[1:]):
        # Rodrigues' rotation taking `before` onto `after`, in cross-product form.
        turn = np.cross(before, after)
        axis = axis + np.cross(turn, axis) + np.cross(turn, np.cross(turn, axis)) / (1 + before @ after)
        axis = axis / np.linalg.norm(axis)
        axes.append(axis)
    return np.array(axes)


def sweep_section(path, ring, scales=None):
    """Closed tube around a path; the ring's first axis starts as close to +Y as it can."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        raise ValueError("a swept path needs at least two points")
    scales = np.ones(len(path)) if scales is None else np.asarray(scales, dtype=np.float64)
    tangents = _tangents(path)
    axes = _transported_axes(tangents)
    others = np.cross(tangents, axes)

    count = len(ring)
    rings = (path[:, None] + scales[:, None, None] * (ring[:, 0, None] * axes[:, None]
                                                      + ring[:, 1, None] * others[:, None]))
    vertices = np.vstack((rings.reshape(-1, 3), path[0], path[-1]))

    faces = []
    for level in range(len(path) - 1):
        for i in range(count):
            a, b = level * count + i, level * count + (i + 1) % count
            faces += [(a, a + count, b), (b, a + count, b + count)]
    start_center, end_center = len(vertices) - 2, len(vertices) - 1
    last = (len(path) - 1) * count
    for i in range(count):
        faces.append((start_center, i, (i + 1) % count))
        faces.append((end_center, last + (i + 1) % count, last + i))

    tube = trimesh.Trimesh(vertices, faces, process=False)
    if tube.volume < 0:
        tube.invert()
    return tube
