"""A procedurally built 3D wireframe glasses model and its renderer.

Accessory-local coordinates (what every glasses model is expressed in once
its `orientation` and `units_to_cm` are applied):
  - origin: centre of the frame front, between the lenses, at lens mid-height
  - X: towards the screen-right lens
  - Y: up
  - Z: forward, away from the face (towards the camera when looking at it)
  - units: centimetres
These match the face-local axes in head_pose.py, so a model at its origin
sits on the attachment point facing the same way as the face.
"""
from dataclasses import dataclass, field
from functools import cached_property
import math

import cv2
import numpy as np

from accessory_transform import Attachment, project_accessory
from occlusion import sample_lines, visible_line_segments

# Spacing of the depth-tested samples along each edge.
SAMPLE_SPACING_CM = 0.25

MAGENTA = (255, 0, 255)  # BGR


@dataclass(frozen=True)
class WireframeModel:
    """Geometry plus the metadata a future .glb/.obj loader would also supply."""
    name: str
    vertices: np.ndarray        # (N, 3) float, model units, model axes
    edges: np.ndarray           # (M, 2) int, pairs of vertex indices
    units_to_cm: float = 1.0
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))  # model -> accessory axes
    color: tuple = MAGENTA      # style metadata (BGR)
    line_thickness: int = 2

    @cached_property
    def line_samples(self):
        """Edge samples for occlusion; computed once per model, on first use."""
        return sample_lines(self.vertices, self.edges,
                            SAMPLE_SPACING_CM / self.units_to_cm)


@dataclass(frozen=True)
class GlassesDimensions:
    """Frame measurements in centimetres."""
    frame_width: float = 14.0    # outer edge to outer edge of the lenses
    lens_height: float = 4.0
    bridge_width: float = 1.8    # gap between the lenses
    temple_length: float = 14.0  # how far the arms reach back from the hinges
    wrap_depth: float = 0.5      # how far the outer lens edges curve back
    ear_drop: float = 1.5        # how far the arm tips bend down behind the ears
    temple_spread: float = 1.6   # how far each arm splays outwards by the ear bend
    lens_segments: int = 24      # points per lens outline

    @property
    def lens_width(self):
        return (self.frame_width - self.bridge_width) / 2

    def validate(self):
        if min(self.frame_width, self.lens_height, self.bridge_width,
               self.temple_length) <= 0:
            raise ValueError("glasses dimensions must be positive")
        if self.bridge_width >= self.frame_width:
            raise ValueError("bridge_width must be smaller than frame_width")
        if self.lens_segments < 8:
            raise ValueError("lens_segments must be at least 8")


# Calibration for the default model: lens centres level with the eye corners
# (0.6 cm below landmark 168) and the frame front 0.5 cm in front of the
# nose bridge so it doesn't pass through the nose.
DEFAULT_GLASSES_ATTACHMENT = Attachment(offset_cm=(0.0, -0.6, 0.5))


def _lens_outline(center_x, dims):
    """Rounded-rectangle (superellipse) lens outline, wrapped around the face."""
    half_width = dims.lens_width / 2
    half_height = dims.lens_height / 2
    frame_half_width = dims.frame_width / 2
    points = []
    for i in range(dims.lens_segments):
        t = 2 * math.pi * i / dims.lens_segments
        # |cos|^0.5 instead of cos squares the corners off an ellipse.
        x = center_x + half_width * math.copysign(abs(math.cos(t)) ** 0.5, math.cos(t))
        y = half_height * math.copysign(abs(math.sin(t)) ** 0.5, math.sin(t))
        z = -dims.wrap_depth * (x / frame_half_width) ** 2
        points.append((x, y, z))
    return points


def _nearest(vertices, start, stop, target_xy):
    """Index (within start..stop) of the vertex closest to target_xy in X/Y."""
    squared_distances = ((vertices[start:stop, :2] - target_xy) ** 2).sum(axis=1)
    return start + int(np.argmin(squared_distances))


def build_glasses_model(dims=GlassesDimensions()):
    """Build the wireframe glasses once; the result is reused every frame."""
    dims.validate()
    lens_offset_x = (dims.bridge_width + dims.lens_width) / 2
    n = dims.lens_segments

    vertices = _lens_outline(-lens_offset_x, dims) + _lens_outline(lens_offset_x, dims)
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(n + i, n + (i + 1) % n) for i in range(n)]
    vertices_array = np.array(vertices)

    # Attach the bridge and temples to existing lens vertices so the frame
    # is one connected wireframe.
    bridge_y = dims.lens_height * 0.3
    hinge_y = dims.lens_height * 0.25
    half_bridge = dims.bridge_width / 2
    half_frame = dims.frame_width / 2
    left_inner = _nearest(vertices_array, 0, n, (-half_bridge, bridge_y))
    right_inner = _nearest(vertices_array, n, 2 * n, (half_bridge, bridge_y))
    left_hinge = _nearest(vertices_array, 0, n, (-half_frame, hinge_y))
    right_hinge = _nearest(vertices_array, n, 2 * n, (half_frame, hinge_y))

    # Bridge: arches slightly up over the nose.
    bridge_top = len(vertices)
    vertices.append((0.0, bridge_y + 0.4, 0.0))
    edges += [(left_inner, bridge_top), (bridge_top, right_inner)]

    # Temples: back from each hinge (-Z), splaying outwards to clear the side
    # of the head, then bending down at the ear.
    for hinge, outwards in ((left_hinge, -1.0), (right_hinge, 1.0)):
        hx, hy, hz = vertices[hinge]
        ear_x = hx + outwards * dims.temple_spread
        bend = len(vertices)
        vertices.append((ear_x, hy, hz - 0.85 * dims.temple_length))
        vertices.append((ear_x, hy - dims.ear_drop, hz - dims.temple_length))
        edges += [(hinge, bend), (bend, bend + 1)]

    return WireframeModel(
        name="basic wireframe glasses",
        vertices=np.array(vertices, dtype=np.float64),
        edges=np.array(edges, dtype=np.int32),
    )


def projected_segments(model, accessory_to_face, pose):
    """(K, 2, 2) int32 pixel segments for edges whose ends are both drawable."""
    pixels = project_accessory(model.vertices, accessory_to_face, pose)
    if pixels is None:
        return None
    segments = pixels[model.edges]  # (M, 2 ends, 2 coords)
    # Drop edges touching a point behind the camera, or absurdly far off-screen.
    drawable = np.isfinite(segments).all(axis=(1, 2)) & \
        (np.abs(segments) < 1e5).all(axis=(1, 2))
    return np.round(segments[drawable]).astype(np.int32)


def draw_wireframe(frame, model, accessory_to_face, pose, depth_buffer=None):
    """Draw the model; returns False (and draws nothing) without a valid pose.

    With a depth buffer (see occlusion.py), parts behind the head are hidden.
    """
    if pose is None:
        return False
    if depth_buffer is None:
        segments = projected_segments(model, accessory_to_face, pose)
    else:
        segments = visible_line_segments(model.line_samples,
                                         pose.transform @ accessory_to_face,
                                         pose.camera_matrix, depth_buffer)
        on_canvas = (np.abs(segments) < 1e5).all(axis=(1, 2))
        segments = np.round(segments[on_canvas]).astype(np.int32)
    if segments is None or len(segments) == 0:
        return False
    cv2.polylines(frame, list(segments), False, model.color,
                  model.line_thickness, cv2.LINE_AA)
    return True
