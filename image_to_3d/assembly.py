"""Build, validate and export the complete glasses model.

Every component is in the same coordinates, matching the accessory-local axes in
glasses_3d.py and head_pose.py:
  - origin: centre of the frame's front view, on the front surface at x = 0
  - X: towards the screen-right lens
  - Y: up
  - Z: forward, away from the face; the frame curves back and the temples run to -Z
  - units: centimetres in memory, metres in the GLB (load with units_to_cm=100)

Parts that are meant to overlap, like a hinge in the frame, stay separate shells,
so the model is an assembly and not one watertight solid.
"""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import trimesh

from image_to_3d.appearance import textured_frame_mesh
from image_to_3d.curvature import build_curved_frame
from image_to_3d.errors import MeshBuildError
from image_to_3d.hinges import (HingeParameters, build_fallback_bridge, build_hinge,
                                build_nose_pad, estimate_hinge_anchor, estimate_nose_pad_anchor)
from image_to_3d.mesh import CM_TO_M, scale_polygons_to_cm, write_glb
from image_to_3d.temples import TempleParameters, build_temple
from image_to_3d.validation import validate_components

FRONT_FRAME = "front_frame"
LEFT_TEMPLE = "left_temple"
RIGHT_TEMPLE = "right_temple"
LEFT_HINGE = "left_hinge"
RIGHT_HINGE = "right_hinge"
LEFT_NOSE_PAD = "left_nose_pad"
RIGHT_NOSE_PAD = "right_nose_pad"
BRIDGE = "bridge"

MAX_CURVATURE_FRACTION = 0.25  # of the frame width: beyond this the edge slope passes 45 degrees
MAX_FRAME_THICKNESS_CM = 2.0

FRAME_MATERIAL = dict(baseColorFactor=(30, 30, 34, 255), metallicFactor=0.0, roughnessFactor=0.35)
METAL_MATERIAL = dict(baseColorFactor=(170, 170, 175, 255), metallicFactor=0.9,
                      roughnessFactor=0.3)
PAD_MATERIAL = dict(baseColorFactor=(225, 225, 215, 255), metallicFactor=0.0, roughnessFactor=0.5)
MATERIALS = {FRONT_FRAME: FRAME_MATERIAL, LEFT_TEMPLE: FRAME_MATERIAL, RIGHT_TEMPLE: FRAME_MATERIAL,
             BRIDGE: FRAME_MATERIAL, LEFT_HINGE: METAL_MATERIAL, RIGHT_HINGE: METAL_MATERIAL,
             LEFT_NOSE_PAD: PAD_MATERIAL, RIGHT_NOSE_PAD: PAD_MATERIAL}
# Which of FrameAppearance.materials() each component takes.
MATERIAL_ROLES = {FRONT_FRAME: "front", LEFT_TEMPLE: "temple", RIGHT_TEMPLE: "temple",
                  BRIDGE: "frame", LEFT_HINGE: "hinge", RIGHT_HINGE: "hinge",
                  LEFT_NOSE_PAD: "pad", RIGHT_NOSE_PAD: "pad"}


@dataclass(frozen=True)
class GlassesParameters:
    """Everything the photo can't tell us, in centimetres."""
    frame_width: float = 14.0      # outer edge to outer edge, as in GlassesDimensions
    frame_thickness: float = 0.4   # front to rear along Z
    frame_curvature: float = 0.5   # how far the outer edges sit behind the centre
    left_temple: TempleParameters = TempleParameters()
    right_temple: TempleParameters = TempleParameters()
    hinge: HingeParameters = HingeParameters()
    nose_pads: bool = False        # most acetate frames have none, so they are opt-in

    def validate(self):
        values = (self.frame_width, self.frame_thickness, self.frame_curvature)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frame width, thickness and curvature must be finite numbers")
        if self.frame_width <= 0:
            raise ValueError("frame width must be positive")
        if not 0 < self.frame_thickness <= MAX_FRAME_THICKNESS_CM:
            raise ValueError(f"frame thickness must be in (0, {MAX_FRAME_THICKNESS_CM}] cm")
        if not 0 <= self.frame_curvature <= MAX_CURVATURE_FRACTION * self.frame_width:
            raise ValueError("frame curvature must be between 0 and a quarter of the frame width")
        self.left_temple.validate()
        self.right_temple.validate()
        self.hinge.validate()


@dataclass(frozen=True)
class GlassesModel:
    components: dict         # {name: mesh in centimetres}, in a fixed order
    report: object           # ValidationReport
    openings: int            # enclosed gaps preserved from the silhouette
    nose_support: str        # what the nose rests on
    left_pivot: np.ndarray
    right_pivot: np.ndarray


def _contacts(components):
    """Pairs that are meant to overlap, among the components present."""
    pairs = [(FRONT_FRAME, LEFT_HINGE), (FRONT_FRAME, RIGHT_HINGE),
             (LEFT_HINGE, LEFT_TEMPLE), (RIGHT_HINGE, RIGHT_TEMPLE),
             (FRONT_FRAME, LEFT_NOSE_PAD), (FRONT_FRAME, RIGHT_NOSE_PAD), (FRONT_FRAME, BRIDGE)]
    return [pair for pair in pairs if pair[0] in components and pair[1] in components]


def _nose_support(components, has_bridge):
    bridge = "generated bridge" if has_bridge else "bridge from the photo"
    return f"{bridge} and nose pads" if LEFT_NOSE_PAD in components else bridge


def assemble_glasses(polygons, params=GlassesParameters()):
    """Build every component from traced outline polygons; raises if the result is invalid."""
    params.validate()
    polygons_cm = scale_polygons_to_cm(polygons, params.frame_width)
    shape = (params.frame_width, params.frame_thickness, params.frame_curvature)

    hinges = {side: build_hinge(estimate_hinge_anchor(polygons_cm, side, params.hinge), side,
                                *shape, params.hinge) for side in (-1, 1)}
    components = {
        FRONT_FRAME: build_curved_frame(polygons_cm, params.frame_thickness, params.frame_curvature),
        LEFT_TEMPLE: build_temple(hinges[-1].pivot, -1, params.left_temple),
        RIGHT_TEMPLE: build_temple(hinges[1].pivot, 1, params.right_temple),
        LEFT_HINGE: hinges[-1].mesh,
        RIGHT_HINGE: hinges[1].mesh,
    }
    bridge = build_fallback_bridge(polygons_cm, params.frame_thickness, params.frame_curvature)
    if bridge is not None:
        components[BRIDGE] = bridge
    if params.nose_pads:
        for name, side in ((LEFT_NOSE_PAD, -1), (RIGHT_NOSE_PAD, 1)):
            components[name] = build_nose_pad(estimate_nose_pad_anchor(polygons_cm, side), side,
                                              *shape)

    report = validate_components(
        components, _contacts(components),
        check_self=[name for name in (FRONT_FRAME, LEFT_TEMPLE, RIGHT_TEMPLE, BRIDGE)
                    if name in components])
    if not report.ok:
        raise MeshBuildError("The assembled model is invalid: " + "; ".join(report.problems))
    return GlassesModel(components=components, report=report,
                        openings=sum(len(polygon.interiors) for polygon in polygons_cm),
                        nose_support=_nose_support(components, bridge is not None),
                        left_pivot=hinges[-1].pivot, right_pivot=hinges[1].pivot)


def _export_mesh(name, mesh, appearance):
    """A copy in metres, carrying its material, and the texture for the front frame."""
    if appearance is not None and name == FRONT_FRAME:
        metres = textured_frame_mesh(mesh, appearance)
        metres.visual.material.name = name
    else:
        metres = mesh.copy()
        values = MATERIALS[name] if appearance is None else appearance.materials()[MATERIAL_ROLES[name]]
        material = trimesh.visual.material.PBRMaterial(name=name, **values)
        metres.visual = trimesh.visual.TextureVisuals(material=material)
    metres.apply_scale(CM_TO_M)
    return metres


def export_model_glb(model, path, appearance=None):
    """Write one GLB with a named node per component; returns the path written.

    With a FrameAppearance, the front frame carries the embedded photo texture and
    the other parts take colours estimated from it. Without one, every part gets
    the fixed materials above.
    """
    scene = trimesh.Scene()
    for name, mesh in model.components.items():
        scene.add_geometry(_export_mesh(name, mesh, appearance), node_name=name, geom_name=name)
    return write_glb(scene, Path(path))
