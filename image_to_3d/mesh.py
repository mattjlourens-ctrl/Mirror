"""Turn a glasses mask into a shallow extruded mesh and export it as GLB.

Mesh coordinates match the accessory-local axes in glasses_3d.py:
  - origin: centre of the silhouette, on the front face
  - X: towards the screen-right lens
  - Y: up
  - Z: forward; the mesh extends back from Z = 0 to Z = -thickness
  - units: centimetres (the exported GLB is in metres, as glTF specifies)
"""
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
from shapely import affinity, make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
import trimesh

from image_to_3d.errors import MeshBuildError

# Masks are resampled to this longest side, so outlines don't depend on image size.
WORKING_SIZE_PX = 1024
OUTLINE_TOLERANCE_PX = 1.0  # how far simplified outlines may stray from the mask
MIN_POLYGON_AREA_PX = 16.0
CM_TO_M = 0.01


@dataclass(frozen=True)
class FrameDimensions:
    """Mesh measurements in centimetres."""
    frame_width: float = 14.0  # outer edge to outer edge, as in GlassesDimensions
    thickness: float = 0.4     # front to back depth of the extrusion

    def validate(self):
        for value in (self.frame_width, self.thickness):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("frame width and thickness must be positive numbers")


def _resample_to_working_size(mask):
    """Crop the mask to its content and scale it to the working resolution."""
    ys, xs = np.nonzero(mask)
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = cropped.shape
    scale = WORKING_SIZE_PX / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if scale < 1:
        # Keep any covered pixel: thin rims must not break, which would merge
        # lens openings into the background.
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        binary = resized > 0
    else:
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_LINEAR)
        binary = resized > 127
    return np.pad(binary.astype(np.uint8) * 255, 1)


def _simplified_ring(contour):
    ring = cv2.approxPolyDP(contour, OUTLINE_TOLERANCE_PX, True).reshape(-1, 2)
    return ring if len(ring) >= 3 else None


def _polygons_from_contours(contours, hierarchy):
    """Pair each outer contour with its holes; RETR_CCOMP gives two levels."""
    polygons = []
    for index, contour in enumerate(contours):
        _, _, first_hole, parent = hierarchy[index]
        if parent != -1:
            continue
        outer = _simplified_ring(contour)
        if outer is None:
            continue
        holes = []
        hole_index = first_hole
        while hole_index != -1:
            hole = _simplified_ring(contours[hole_index])
            if hole is not None:
                holes.append(hole)
            hole_index = hierarchy[hole_index][0]
        polygons.append(Polygon(outer, holes))
    return polygons


def mask_to_polygons(mask):
    """Trace a mask into valid, disjoint polygons, with openings as interior rings.

    Coordinates are working-resolution pixels (Y down), left to right by part.
    """
    if not mask.any():
        return []
    contours, hierarchy = cv2.findContours(_resample_to_working_size(mask), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:  # everything rounded away at the working resolution
        return []
    traced = _polygons_from_contours(contours, hierarchy[0])
    # Contours run through pixel centres, half a pixel inside the true edge of
    # the frame on both sides; growing by that much restores the rim's width.
    grown = [make_valid(polygon).buffer(0.5, join_style="mitre") for polygon in traced]
    merged = unary_union(grown)
    parts = merged.geoms if isinstance(merged, MultiPolygon) else [merged]
    kept = [part for part in parts if part.area >= MIN_POLYGON_AREA_PX]
    return sorted(kept, key=lambda part: part.bounds[0])


def polygons_to_mesh(polygons, dimensions=FrameDimensions()):
    """Extrude polygons into one mesh scaled and placed per the module docstring."""
    dimensions.validate()
    if not polygons:
        raise MeshBuildError("The silhouette has no area to build a mesh from")

    min_x, min_y, max_x, max_y = MultiPolygon(polygons).bounds
    scale = dimensions.frame_width / (max_x - min_x)
    # Negative Y scale flips image-down to mesh-up.
    to_mesh = [scale, 0, 0, -scale, -scale * (min_x + max_x) / 2, scale * (min_y + max_y) / 2]

    parts = [trimesh.creation.extrude_polygon(affinity.affine_transform(polygon, to_mesh),
                                              height=dimensions.thickness, engine="earcut")
             for polygon in polygons]
    mesh = trimesh.util.concatenate(parts)
    mesh.apply_translation((0, 0, -dimensions.thickness))
    if not mesh.is_volume:
        raise MeshBuildError("The extruded silhouette is not a closed, consistent volume")
    return mesh


def build_frame_mesh(mask, dimensions=FrameDimensions()):
    """Build the shallow frame mesh directly from a mask."""
    return polygons_to_mesh(mask_to_polygons(mask), dimensions)


def export_glb(mesh, path):
    """Write the mesh, in metres, to a .glb file; returns the path written."""
    path = Path(path)
    if path.suffix.lower() != ".glb":
        raise ValueError(f"Output file must end in .glb: {path}")
    if mesh.is_empty:
        raise MeshBuildError("Cannot export an empty mesh")

    metres = mesh.copy()
    metres.apply_scale(CM_TO_M)
    data = metres.export(file_type="glb")
    if not data:
        raise MeshBuildError("Trimesh produced an empty GLB")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
