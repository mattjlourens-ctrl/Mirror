"""Procedural temple arms. Nothing here comes from the photo except the hinge position.

Each arm's centreline is two joined cubic Beziers, in coordinates relative to the
hinge with +X pointing away from the head (the left arm is mirrored at the end):

  hinge --------- first curve --------- ear bend ---- second curve ---- tip

The two curves share a tangent at the bend, so the arm has no kink. The first
stays level; the second drops by `ear_drop` and curves in by `curvature`.
"""
from dataclasses import dataclass
import math

import numpy as np

from image_to_3d.sweep import (cubic_bezier, flatten_cubics, polyline_length, section_ring,
                               sweep_section)

FLATNESS_TOLERANCE_CM = 0.005
LENGTH_SOLVER_ITERATIONS = 16
LENGTH_SAMPLES = 256
ROUNDED_TIP_CM = 0.4
ROUNDED_TIP_RINGS = (0.0, 0.6, 0.9, 1.0)  # fractions of the rounded length
MIN_TIP_SCALE = 0.35
MIN_RING_SPACING_CM = 0.1
MAX_SECTION_CM = 3.0


@dataclass(frozen=True)
class TempleParameters:
    """One arm's measurements, in centimetres.

    None of these can be measured from a frontal photo. The defaults follow
    GlassesDimensions in glasses_3d.py and clear Mirror's head proxy on a 14 cm frame.
    """
    length: float = 14.0        # along the centreline, hinge to tip, like a printed size
    width: float = 0.55         # section height (Y at the hinge)
    thickness: float = 0.25     # section depth, across the arm
    splay: float = 1.6          # outward offset at the ear bend, to clear the head
    curvature: float = 0.4      # how far the tip curves back in from the bend
    ear_drop: float = 1.5       # how far the tip falls below the hinge
    bend_fraction: float = 0.8  # where the ear bend starts, as a fraction of the reach

    def validate(self):
        values = (self.length, self.width, self.thickness, self.splay, self.curvature,
                  self.ear_drop, self.bend_fraction)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("temple parameters must be finite numbers")
        if self.length <= 0:
            raise ValueError("temple length must be positive")
        if not (0 < self.width <= MAX_SECTION_CM and 0 < self.thickness <= MAX_SECTION_CM):
            raise ValueError(f"temple width and thickness must be in (0, {MAX_SECTION_CM}] cm")
        limit = self.length / 2
        if not 0 <= self.curvature <= limit:
            raise ValueError("temple curvature must be between 0 and half the temple length")
        if not 0 <= self.ear_drop <= limit:
            raise ValueError("temple ear drop must be between 0 and half the temple length")
        if abs(self.splay) > limit:
            raise ValueError("temple splay must be within half the temple length")
        if not 0.5 <= self.bend_fraction <= 0.95:
            raise ValueError("temple bend fraction must be in 0.5..0.95")


def _centerline_curves(params, reach):
    """The two Bezier control polygons for an arm reaching `reach` cm back."""
    bend_z = reach * params.bend_fraction
    ear_z = reach - bend_z
    first_slope = params.splay / bend_z
    second_slope = -params.curvature / ear_z
    # Sharing this direction at the bend keeps the two curves tangent there.
    at_bend = np.array(((first_slope + second_slope) / 2, 0.0, -1.0))

    hinge = np.zeros(3)
    bend = np.array((params.splay, 0.0, -bend_z))
    tip = np.array((params.splay - params.curvature, -params.ear_drop, -reach))
    first = np.array((hinge, hinge + np.array((first_slope, 0.0, -1.0)) * bend_z / 3,
                      bend - at_bend * bend_z / 3, bend))
    # Ending on a slope 1.5x the average drop makes the arm hook down, not just slant.
    at_tip = np.array((second_slope, -1.5 * params.ear_drop / ear_z, -1.0))
    second = np.array((bend, bend + at_bend * ear_z / 3, tip - at_tip * ear_z / 3, tip))
    return first, second


def _reach_for_length(params):
    """Solve for the reach that makes the centreline `length` long: fixed-point scaling."""
    reach = params.length
    samples = np.linspace(0, 1, LENGTH_SAMPLES)
    for _ in range(LENGTH_SOLVER_ITERATIONS):
        length = sum(polyline_length(cubic_bezier(curve, samples))
                     for curve in _centerline_curves(params, reach))
        reach *= params.length / length
    return reach


def temple_centerline(params):
    """Polyline from the hinge (origin) to the tip, in hinge-relative coordinates."""
    params.validate()
    return flatten_cubics(_centerline_curves(params, _reach_for_length(params)),
                          FLATNESS_TOLERANCE_CM)


def _with_rounded_tip(path, tip_length):
    """Insert rings over the last stretch of the path; returns (path, ring scales)."""
    spans = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance = np.r_[0.0, np.cumsum(spans)]
    start = distance[-1] - tip_length
    tip_distances = start + np.array(ROUNDED_TIP_RINGS) * tip_length
    # Rings almost on top of each other would fold through one another at a bend.
    keep = distance < start - MIN_RING_SPACING_CM
    all_distances = np.r_[distance[keep], tip_distances]
    points = np.stack([np.interp(all_distances, distance, path[:, axis]) for axis in range(3)],
                      axis=1)
    scales = np.r_[np.ones(keep.sum()),
                   np.maximum(np.sqrt(1 - np.array(ROUNDED_TIP_RINGS) ** 2), MIN_TIP_SCALE)]
    return points, scales


def build_temple(hinge, side, params=TempleParameters()):
    """Arm mesh from a hinge point; side is +1 for screen-right, -1 for screen-left."""
    if side not in (-1, 1):
        raise ValueError("side must be +1 (screen-right) or -1 (screen-left)")
    path = temple_centerline(params)
    tip_length = min(ROUNDED_TIP_CM, params.length / 4)
    path, scales = _with_rounded_tip(path, tip_length)
    temple = sweep_section(path, section_ring(params.width / 2, params.thickness / 2), scales)
    if side == -1:
        temple.vertices = temple.vertices * (-1, 1, 1)
        temple.invert()
    temple.apply_translation(hinge)
    return temple
