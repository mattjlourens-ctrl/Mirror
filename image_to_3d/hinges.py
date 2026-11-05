"""Hinges and nose supports: simple shapes placed using the traced outline.

Only the positions come from the photo. The shapes and sizes are simplified
stand-ins, because the real parts sit behind the frame and can't be seen.
"""
from dataclasses import dataclass, fields
import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, box
from shapely.ops import nearest_points
import trimesh

from image_to_3d.curvature import wrap_offset
from image_to_3d.sweep import UP, flatten_cubics, section_ring, sweep_section

HINGE_BAND_FRACTION = 0.025   # how far in from the outer edge counts as the end piece
PIVOT_CLEARANCE_CM = 0.03     # gap between the barrel and the frame's rear surface
BARREL_SECTIONS = 16

PAD_SIZE_CM = (0.5, 0.9, 0.22)  # ellipsoid diameters along X, Y, Z before tilting
PAD_TILT_DEGREES = 20.0
PAD_ARM_LENGTH_CM = 0.45
PAD_ARM_RADIUS_CM = 0.05
PAD_NASAL_SHIFT_CM = 0.15
PAD_RIM_INSET_CM = 0.1
PAD_HEIGHT_BELOW_CENTER = 0.2   # as a fraction of the opening's height
NO_OPENING_PAD_X_FRACTION = 0.1

BRIDGE_ARCH_CM = 0.4
BRIDGE_HEIGHT_CM = 0.4
BRIDGE_RIM_OVERLAP_CM = 0.15
BRIDGE_FLATNESS_CM = 0.005


@dataclass(frozen=True)
class HingeParameters:
    """Simplified hinge shapes, in centimetres."""
    barrel_radius: float = 0.13
    barrel_height: float = 0.45
    block_width: float = 0.3     # mounting block size across X
    block_height: float = 0.4    # mounting block size across Y
    anchor_inset: float = 0.2    # how far in from the outer edge the hinge sits

    def validate(self):
        values = [getattr(self, item.name) for item in fields(self)]
        if not all(math.isfinite(value) for value in values) or min(values) < 0:
            raise ValueError("hinge sizes must be non-negative numbers")
        if min(self.barrel_radius, self.barrel_height, self.block_width, self.block_height) == 0:
            raise ValueError("hinge barrel and block sizes must be positive")


@dataclass(frozen=True)
class Hinge:
    barrel: trimesh.Trimesh     # vertical cylinder around the pivot
    block: trimesh.Trimesh      # joins the frame's rear surface to the barrel
    pivot: np.ndarray           # where the temple starts
    block_axis: np.ndarray      # unit vector from the frame into the barrel

    @property
    def mesh(self):
        return trimesh.util.concatenate([self.block, self.barrel])


def _on_material(polygons_cm, point):
    """The point itself if it's on the frame, otherwise the closest point that is."""
    frame = MultiPolygon(polygons_cm)
    if frame.contains(Point(point)):
        return np.array(point, dtype=np.float64)
    return np.array(nearest_points(frame, Point(point))[0].coords[0])


def estimate_hinge_anchor(polygons_cm, side, params=HingeParameters()):
    """(x, y) on the frame where a hinge mounts: mid-height of the outermost end piece."""
    bounds = MultiPolygon(polygons_cm).bounds
    outer_x = bounds[2] if side > 0 else bounds[0]
    band = HINGE_BAND_FRACTION * (bounds[2] - bounds[0])
    x_min, x_max = sorted((outer_x, outer_x - side * band))
    end_piece = MultiPolygon(polygons_cm).intersection(box(x_min, bounds[1], x_max, bounds[3]))
    _, y_min, _, y_max = end_piece.bounds
    return _on_material(polygons_cm, (outer_x - side * params.anchor_inset, (y_min + y_max) / 2))


def _surface_normal(x, frame_width, depth):
    """Forward unit normal of z = f(x)."""
    slope = -8 * depth * x / frame_width ** 2
    normal = np.array((-slope, 0.0, 1.0))
    return normal / np.linalg.norm(normal)


def _upright_frame(direction, origin):
    """4x4 matrix with local Z along direction and local Y as near to +Y as possible."""
    y_axis = UP - direction * (UP @ direction)
    y_axis /= np.linalg.norm(y_axis)
    matrix = np.eye(4)
    matrix[:3, 0] = np.cross(y_axis, direction)
    matrix[:3, 1] = y_axis
    matrix[:3, 2] = direction
    matrix[:3, 3] = origin
    return matrix


def build_hinge(anchor, side, frame_width, thickness, depth, params=HingeParameters()):
    """A vertical barrel behind the frame and a block joining it to the rear surface.

    The block runs along the frame's normal at the anchor. The frame keeps curving
    back past the anchor, so the pivot clears the frame's outermost rear edge too.
    """
    params.validate()
    x, y = anchor
    rear_z = float(wrap_offset(x, frame_width, depth)) - thickness
    outer_rear_z = float(wrap_offset(side * frame_width / 2, frame_width, depth)) - thickness
    backwards = -_surface_normal(x, frame_width, depth)
    rear_point = np.array((x, y, rear_z))
    reach = (rear_z - outer_rear_z + params.barrel_radius + PIVOT_CLEARANCE_CM) / -backwards[2]
    pivot = rear_point + backwards * reach

    start = rear_point - backwards * thickness / 2  # inside the frame
    length = np.linalg.norm(pivot - start)
    block = trimesh.creation.box((params.block_width, params.block_height, length),
                                 transform=_upright_frame(backwards, (start + pivot) / 2))
    # Trimesh's cylinder runs along Z; a quarter turn about X stands it up along Y.
    barrel = trimesh.creation.cylinder(params.barrel_radius, params.barrel_height,
                                       sections=BARREL_SECTIONS)
    barrel.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, (1, 0, 0)))
    barrel.apply_translation(pivot)
    return Hinge(barrel=barrel, block=block, pivot=pivot, block_axis=backwards)


def _nearest_opening(polygons_cm, side):
    """The opening on this side of the frame that's closest to the centre, if any."""
    candidates = [ring for polygon in polygons_cm for ring in polygon.interiors
                  if ring.centroid.x * side > 0]
    return min(candidates, key=lambda ring: abs(ring.centroid.x), default=None)


def estimate_nose_pad_anchor(polygons_cm, side):
    """(x, y) on the rim, on the nasal side of the lens opening and below its centre."""
    opening = _nearest_opening(polygons_cm, side)
    if opening is None:
        bounds = MultiPolygon(polygons_cm).bounds
        target = (side * NO_OPENING_PAD_X_FRACTION * (bounds[2] - bounds[0]),
                  (bounds[1] + bounds[3]) / 2)
        return _on_material(polygons_cm, target)

    center_x = opening.centroid.x
    _, y_min, _, y_max = opening.bounds
    y = opening.centroid.y - PAD_HEIGHT_BELOW_CENTER * (y_max - y_min)
    crossings = shapely.get_coordinates(opening.intersection(LineString([(center_x, y), (0, y)])))
    edge_x = crossings[np.argmin(np.abs(crossings[:, 0] - center_x)), 0]
    return _on_material(polygons_cm, (edge_x - side * PAD_RIM_INSET_CM, y))


def build_nose_pad(anchor, side, frame_width, thickness, depth):
    """A flattened ellipsoid on a short arm from the rim, tilted to face the nose."""
    x, y = anchor
    rear_z = float(wrap_offset(x, frame_width, depth)) - thickness
    center = np.array((x - side * PAD_NASAL_SHIFT_CM, y, rear_z - PAD_ARM_LENGTH_CM))
    arm = trimesh.creation.cylinder(PAD_ARM_RADIUS_CM, sections=8,
                                    segment=((x, y, rear_z + thickness / 2), center))
    pad = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    pad.apply_scale(PAD_SIZE_CM)
    tilt = trimesh.transformations.rotation_matrix(math.radians(side * PAD_TILT_DEGREES), UP)
    pad.apply_transform(tilt)
    pad.apply_translation(center)
    return trimesh.util.concatenate([arm, pad])


def build_fallback_bridge(polygons_cm, thickness, depth):
    """An arched bar between rims when the photo has no bridge; None if it has one."""
    left = [polygon for polygon in polygons_cm if polygon.bounds[2] < 0]
    right = [polygon for polygon in polygons_cm if polygon.bounds[0] > 0]
    spans_center = any(polygon.bounds[0] <= 0 <= polygon.bounds[2] for polygon in polygons_cm)
    if spans_center or not left or not right:
        return None

    left_rim = max(left, key=lambda polygon: polygon.bounds[2])
    right_rim = min(right, key=lambda polygon: polygon.bounds[0])
    bounds = MultiPolygon(polygons_cm).bounds
    frame_width = bounds[2] - bounds[0]
    left_edge, right_edge = nearest_points(left_rim, right_rim)
    start = np.array(left_edge.coords[0]) - (BRIDGE_RIM_OVERLAP_CM, 0)
    end = np.array(right_edge.coords[0]) + (BRIDGE_RIM_OVERLAP_CM, 0)

    xs = start[0] + (end[0] - start[0]) * np.array((0, 1 / 3, 2 / 3, 1))
    ys = start[1] + (end[1] - start[1]) * np.array((0, 1 / 3, 2 / 3, 1))
    ys += np.array((0, 4 / 3, 4 / 3, 0)) * BRIDGE_ARCH_CM  # a cubic rises 3/4 of this
    zs = wrap_offset(xs, frame_width, depth) - thickness / 2
    path = flatten_cubics([np.stack((xs, ys, zs), axis=1)], BRIDGE_FLATNESS_CM)
    return sweep_section(path, section_ring(BRIDGE_HEIGHT_CM / 2, thickness * 0.4))
